import discord
import re
import requests
import time
import asyncio
import glob
import os
import shutil
import uuid
import subprocess
from typing import List

from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ------------------------------------------------------------
# LLMManager – Qdrant‑backed knowledge + GitHub‑Wiki importer
# ------------------------------------------------------------

class LLMManager(commands.Cog):
    """Interact with a local Ollama LLM using a Qdrant knowledge base."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9999999999)
        self.config.register_global(
            model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333"
        )
        self.collection = "fus_wiki"
        self.q_client: QdrantClient | None = None
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 384‑dim vectors

    # ---------- helpers -------------------------------------------------

    async def ensure_qdrant(self):
        if self.q_client is None:
            url = await self.config.qdrant_url()
            self.q_client = QdrantClient(url=url)

    def _vec(self, text: str) -> List[float]:
        return self.embedder.encode(text).tolist()

    # ---------- low‑level Qdrant ops ------------------------------------

    def _ensure_collection_sync(self):
        try:
            self.q_client.get_collection(self.collection)
        except Exception:
            self.q_client.recreate_collection(
                collection_name=self.collection,
                vectors_config={"size": 384, "distance": "Cosine"}
            )

    def _upsert_sync(self, tag: str, content: str, source: str):
        self._ensure_collection_sync()
        payload = {"tag": tag, "content": content, "source": source}
        self.q_client.upsert(
            collection_name=self.collection,
            points=[{
                "id": uuid.uuid4().int & ((1 << 64) - 1),  # 64‑bit random id
                "vector": self._vec(content),
                "payload": payload
            }]
        )

    def _scroll_sync(self):
        """Return only the list of Records; scroll gives (points, next_offset)."""
        points, _ = self.q_client.scroll(
            collection_name=self.collection,
            with_payload=True,
            limit=1000
        )
        return points

    # ---------- commands: knowledge management -------------------------

    @commands.command()
    async def llmknow(self, ctx, tag: str, *, content: str):
        """Add manual knowledge under *tag*."""
        await self.ensure_qdrant()
        await asyncio.get_running_loop().run_in_executor(None, self._upsert_sync, tag.lower(), content, "manual")
        await ctx.send(f"Added manual info under '{tag.lower()}'.")

    @commands.command()
    async def llmknowshow(self, ctx):
        await self.ensure_qdrant()
        pts = await asyncio.get_running_loop().run_in_executor(None, self._scroll_sync)
        if not pts:
            return await ctx.send("No knowledge entries stored.")

        header, footer, max_len = "```\n", "```", 2000
        chunks, cur = [], header
        for pt in pts:
            pid = getattr(pt, "id", None)
            payload = getattr(pt, "payload", {})
            if not pid or not payload:
                continue
            snippet = payload.get("content", "")[:300].replace("\n", " ")
            line = f"[{pid}] ({payload.get('tag','NoTag')},src={payload.get('source','?')}): {snippet}\n"
            if len(cur) + len(line) > max_len - len(footer):
                chunks.append(cur + footer)
                cur = header + line
            else:
                cur += line
        if cur != header:
            chunks.append(cur + footer)
        for ch in chunks:
            await ctx.send(ch)

    @commands.command()
    async def llmknowdelete(self, ctx, doc_id: int):
        await self.ensure_qdrant()
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.q_client.delete(collection_name=self.collection, points=[doc_id])
        )
        await ctx.send(f"Deleted entry {doc_id}.")

    @commands.command()
    async def llmknowdeletetag(self, ctx, tag: str):
        await self.ensure_qdrant()
        filt = {"must": [{"key": "tag", "match": {"value": tag.lower()}}]}
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.q_client.delete(collection_name=self.collection, filter=filt)
        )
        await ctx.send(f"Deleted entries with tag '{tag.lower()}'.")

    # ---------- GitHub wiki import -------------------------------------

    @commands.command()
    async def importwiki(self, ctx, repo: str = "https://github.com/Kvitekvist/FUS.wiki.git"):
        """Clone / pull GitHub wiki and store pages in Qdrant (source=wiki)."""
        await self.ensure_qdrant()
        base = str(cog_data_path(self)); os.makedirs(base, exist_ok=True)
        clone_dir = os.path.join(base, "wiki")

        # purge old wiki entries
        filt = {"must": [{"key": "source", "match": {"value": "wiki"}}]}
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.q_client.delete(collection_name=self.collection, filter=filt)
        )

        # clone/pull repo
        if os.path.isdir(os.path.join(clone_dir, ".git")):
            subprocess.run(["git", "-C", clone_dir, "pull"], check=False)
        else:
            shutil.rmtree(clone_dir, ignore_errors=True)
            subprocess.run(["git", "clone", repo, clone_dir], check=True)
        await ctx.send("Wiki repo updated. Importing …")

        import markdown, bs4
        md_files = glob.glob(os.path.join(clone_dir, "*.md"))
        for path in md_files:
            text = open(path, encoding="utf-8").read()
            html = markdown.markdown(text)
            soup = bs4.BeautifulSoup(html, "html.parser")
            tags = ", ".join({h.get_text(strip=True) for h in soup.find_all(re.compile("^h[1-3]$"))})
            plain = soup.get_text(" ", strip=True)
            await asyncio.get_running_loop().run_in_executor(None, self._upsert_sync, tags or os.path.basename(path), plain, "wiki")
        await ctx.send(f"Wiki import done ({len(md_files)} pages).")

    # ---------- LLM querying -------------------------------------------

    async def _ollama_chat(self, prompt: str):
        model = await self.config.model(); api = await self.config.api_url()
        rsp = requests.post(f"{api}/api/chat", json={"model": model, "messages": [{"role": "user", "content": prompt}]})
        rsp.raise_for_status(); return rsp.json()["message"]["content"]

    async def ask_with_context(self, question: str) -> str:
        await self.ensure_qdrant()
        vec = self._vec(question)
        hits = self.q_client.search(self.collection, query_vector=vec, limit=5)
        if not hits:
            return "No relevant information found."
        ctx = "\n\n".join(f"[{h.id}] {h.payload.get('content','')[:500]}" for h in hits)
        prompt = (
            f"Context:\n{ctx}\n\nQuestion: {question}\n\n"
            "Answer concisely and include Markdown links when relevant."
        )
        return await asyncio.get_running_loop().run_in_executor(None, self._ollama_chat, prompt)

    @commands.command()
    async def askllm(self, ctx, *, question: str):
        ans = await self.ask_with_context(question)
        await ctx.send(ans)

    # ---------- util commands -----------------------------------------

    @commands.command()
    async def setmodel(self, ctx, model):
        await self.config.model.set(model)
        await ctx.send(f"Model set to {model}")

    @commands.command()
    async def setapi(self, ctx, url):
        await self.config.api_url.set(url.rstrip("/"))
        await ctx.send("API URL updated")

    @commands.command()
    async def setqdrant(self, ctx, url):
        await self.config.qdrant_url.set
