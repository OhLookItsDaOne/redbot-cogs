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
    """Interact with a local Ollama LLM through a Qdrant knowledge base."""

    # ---------- initialisation ----------------------------------------

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9999999999)
        self.config.register_global(
            model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333",
        )
        self.collection = "fus_wiki"
        self.q_client: QdrantClient | None = None
        # small & fast SBERT variant → 384‑D vectors
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # ---------- helpers ------------------------------------------------

    async def ensure_qdrant(self):
        if self.q_client is None:
            url = await self.config.qdrant_url()
            self.q_client = QdrantClient(url=url)

    def _vec(self, txt: str) -> List[float]:
        return self.embedder.encode(txt).tolist()

    # ---------- low‑level Qdrant ops -----------------------------------

    def _ensure_collection_sync(self):
        try:
            self.q_client.get_collection(self.collection)
        except Exception:
            self.q_client.recreate_collection(
                collection_name=self.collection,
                vectors_config={"size": 384, "distance": "Cosine"},
            )

    def _upsert_sync(self, tag: str, content: str, source: str):
        self._ensure_collection_sync()
        self.q_client.upsert(
            self.collection,
            [
                {
                    "id": uuid.uuid4().int & ((1 << 64) - 1),
                    "vector": self._vec(content),
                    "payload": {"tag": tag, "content": content, "source": source},
                }
            ],
        )

    # helper: delete by FILTER (collect ids → delete)
    def _delete_by_filter_sync(self, filt: dict):
        ids: list[int] = []
        offset = None
        while True:
            pts, offset = self.q_client.scroll(
                collection_name=self.collection,
                with_payload=False,
                limit=1000,
                scroll_filter=filt,
                offset=offset,
            )
            ids.extend(p.id for p in pts)
            if offset is None:
                break
        if ids:
            self.q_client.delete(self.collection, ids)

    # ---------- commands: knowledge management ------------------------

    @commands.command()
    async def llmknow(self, ctx, tag: str, *, content: str):
        """Add manual knowledge under **tag**."""
        await self.ensure_qdrant()
        await asyncio.get_running_loop().run_in_executor(
            None, self._upsert_sync, tag.lower(), content, "manual"
        )
        await ctx.send(f"Added manual info under '{tag.lower()}'.")

    @commands.command()
    async def llmknowshow(self, ctx):
        """List all knowledge entries (chunks ≤ 2000 chars)."""
        await self.ensure_qdrant()
        points, _ = self.q_client.scroll(self.collection, with_payload=True, limit=1000)
        if not points:
            return await ctx.send("No knowledge entries stored.")

        header, footer, max_len = "```\n", "```", 2000
        chunk, out = header, []
        for pt in points:
            pl = pt.payload or {}
            snippet = pl.get("content", "")[:300].replace("\n", " ")
            line = f"[{pt.id}] ({pl.get('tag','NoTag')},src={pl.get('source','?')}): {snippet}\n"
            if len(chunk) + len(line) > max_len - len(footer):
                out.append(chunk + footer)
                chunk = header + line
            else:
                chunk += line
        if chunk != header:
            out.append(chunk + footer)
        for ch in out:
            await ctx.send(ch)

    @commands.command()
    async def llmknowdelete(self, ctx, doc_id: int):
        """Delete entry by **ID**."""
        await self.ensure_qdrant()
        await asyncio.get_running_loop().run_in_executor(None, lambda: self.q_client.delete(self.collection, [doc_id]))
        await ctx.send(f"Deleted entry {doc_id}.")

    @commands.command()
    async def llmknowdeletetag(self, ctx, tag: str):
        """Delete every entry whose *tag* matches."""
        await self.ensure_qdrant()
        filt = {"must": [{"key": "tag", "match": {"value": tag.lower()}}]}
        await asyncio.get_running_loop().run_in_executor(None, self._delete_by_filter_sync, filt)
        await ctx.send(f"Deleted entries with tag '{tag.lower()}'.")

    # ---------- GitHub Wiki import ------------------------------------

    @commands.command()
    async def importwiki(self, ctx, repo: str = "https://github.com/Kvitekvist/FUS.wiki.git"):
        """Clone / pull GitHub Wiki and (re‑)import into Qdrant (source=wiki)."""
        await self.ensure_qdrant()
        data_dir = str(cog_data_path(self)); os.makedirs(data_dir, exist_ok=True)
        clone_dir = os.path.join(data_dir, "wiki")

        # 1) purge old wiki docs
        filt = {"must": [{"key": "source", "match": {"value": "wiki"}}]}
        await asyncio.get_running_loop().run_in_executor(None, self._delete_by_filter_sync, filt)

        # 2) clone or pull
        if os.path.isdir(os.path.join(clone_dir, ".git")):
            subprocess.run(["git", "-C", clone_dir, "pull"], check=False)
        else:
            shutil.rmtree(clone_dir, ignore_errors=True)
            subprocess.run(["git", "clone", repo, clone_dir], check=True)
        await ctx.send("Wiki repo updated. Importing …")

        import markdown, bs4
        md_files = glob.glob(os.path.join(clone_dir, "*.md"))
        count = 0
        for path in md_files:
            text = open(path, encoding="utf-8").read()
            html = markdown.markdown(text)
            soup = bs4.BeautifulSoup(html, "html.parser")
            tags = ", ".join({h.get_text(strip=True) for h in soup.find_all(re.compile("^h[1-3]$"))})
            plain = soup.get_text(" ", strip=True)
            await asyncio.get_running_loop().run_in_executor(None, self._upsert_sync, tags or os.path.basename(path), plain, "wiki")
            count += 1
        await ctx.send(f"Wiki import done ({count} pages).")

    # ---------- LLM querying ------------------------------------------

    async def _ollama_chat(self, prompt: str) -> str:
        model = await self.config.model(); api = await self.config.api_url()
        resp = requests.post(f"{api}/api/chat", json={"model": model, "messages": [{"role": "user", "content": prompt}]})
        resp.raise_for_status(); return resp.json()["message"]["content"]

    async def ask_with_context(self, question: str) -> str:
        await self.ensure_qdrant()
        hits = self.q_client.search(self.collection, query_vector=self._vec(question), limit=5)
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
        await ctx.send(await self.ask_with_context(question))

    # ---------- util commands -----------------------------------------

    @commands.command()
    async def setmodel(self, ctx, model):
        await self.config.model.set(model); await ctx.send(f"Model set to {model}")

    @commands.command()
    async def setapi(self, ctx, url):
        await self.config.api_url.set(url.rstrip("/")); await ctx.send("API URL updated")

    @commands.command()
    async def setqdrant(self, ctx, url):
        await self.config.qdrant_url.set(url.rstrip("/")); self.q_client = None; await ctx.send("Qdrant URL updated")

    # ---------- bot events -------------------------------------------

    @commands.Cog.listener()
    async def on_ready(self):
        print("LLMManager cog loaded.")


def setup(bot):
    bot.add_cog(LLMManager(bot))
