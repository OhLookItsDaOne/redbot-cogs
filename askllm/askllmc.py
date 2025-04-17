import asyncio
import glob
import os
import re
import shutil
import subprocess
import uuid
from typing import List

import discord
import requests
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# -----------------------------------------------------------------------------
# Helper – ensure that a PyPI package is available inside the Docker container
# -----------------------------------------------------------------------------

def _ensure_pkg(module: str, pip_name: str | None = None):
    try:
        __import__(module)
    except ModuleNotFoundError:
        subprocess.check_call(["python", "-m", "pip", "install", pip_name or module])
        __import__(module)

# Only wiki‑import needs these
_ensure_pkg("markdown")
_ensure_pkg("bs4", "beautifulsoup4")
import markdown  # noqa: E402 – imported after _ensure_pkg
import bs4       # noqa: E402 – imported after _ensure_pkg

# -----------------------------------------------------------------------------
# LLMManager – Qdrant‑backed KB  +  local Ollama chat
# -----------------------------------------------------------------------------


class LLMManager(commands.Cog):
    """Interact with a local Ollama LLM via a Qdrant knowledge base."""

    # ------------------------------------------------------------------
    # Init / basic helpers
    # ------------------------------------------------------------------

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9876543210)
        self.config.register_global(
            model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333",
        )
        self.collection = "fus_wiki"
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 384‑D vectors
        self.q_client: QdrantClient | None = None

    async def ensure_qdrant(self):
        if self.q_client is None:
            url = await self.config.qdrant_url()
            self.q_client = QdrantClient(url=url)

    # embedding
    def _vec(self, txt: str) -> List[float]:
        return self.embedder.encode(txt).tolist()

    # ------------------------------------------------------------------
    # Low‑level Qdrant helpers (sync – run in executor)
    # ------------------------------------------------------------------

    def _ensure_collection(self):
        try:
            self.q_client.get_collection(self.collection)
        except Exception:
            self.q_client.recreate_collection(
                collection_name=self.collection,
                vectors_config={"size": 384, "distance": "Cosine"},
            )

    def _upsert_sync(self, tag: str, content: str, source: str):
        self._ensure_collection()
        self.q_client.upsert(
            self.collection,
            [
                {
                    "id": uuid.uuid4().int & ((1 << 64) - 1),
                    "vector": self._vec(f"{tag}. {content}"),
                    "payload": {"tag": tag, "content": content, "source": source},
                }
            ],
        )

    def _collect_ids_sync(self, filt: dict) -> List[int]:
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
        return ids

    # ------------------------------------------------------------------
    # Commands – knowledge management
    # ------------------------------------------------------------------

    @commands.command()
    async def llmknow(self, ctx, tag: str, *, content: str):
        """Add manual knowledge under **tag**."""
        await self.ensure_qdrant()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._upsert_sync, tag.lower(), content, "manual")
        await ctx.send(f"Added manual info under '{tag.lower()}'.")

    @commands.command()
    async def llmknowshow(self, ctx):
        """List every stored entry (chunked ≤ 2000 chars)."""
        await self.ensure_qdrant()
        pts, _ = self.q_client.scroll(self.collection, with_payload=True, limit=1000)
        if not pts:
            return await ctx.send("No knowledge entries stored.")

        chunks, cur = [], "```\n"
        for p in pts:
            pl = p.payload or {}
            snippet = pl.get("content", "")[:280].replace("\n", " ")
            line = f"[{p.id}] ({pl.get('tag','NoTag')}, {pl.get('source','?')}): {snippet}\n"
            if len(cur) + len(line) > 1990:
                chunks.append(cur + "```")
                cur = "```\n" + line
            else:
                cur += line
        chunks.append(cur + "```")
        for ch in chunks:
            await ctx.send(ch)

    @commands.command()
    async def llmknowdelete(self, ctx, doc_id: int):
        await self.ensure_qdrant()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self.q_client.delete(self.collection, [doc_id]))
        await ctx.send(f"Deleted entry {doc_id}.")

    @commands.command()
    async def llmknowdeletetag(self, ctx, tag: str):
        await self.ensure_qdrant()
        loop = asyncio.get_running_loop()
        filt = {"must": [{"key": "tag", "match": {"value": tag.lower()}}]}
        ids = await loop.run_in_executor(None, self._collect_ids_sync, filt)
        if ids:
            await loop.run_in_executor(None, lambda: self.q_client.delete(self.collection, ids))
        await ctx.send(f"Deleted entries with tag '{tag.lower()}'.")

    # ------------------------------------------------------------------
    # GitHub‑Wiki import (clone / pull)
    # ------------------------------------------------------------------

    @commands.command()
    async def importwiki(self, ctx, repo: str = "https://github.com/Kvitekvist/FUS.wiki.git"):
        await self.ensure_qdrant()
        loop = asyncio.get_running_loop()
        data_dir = str(cog_data_path(self)); os.makedirs(data_dir, exist_ok=True)
        clone_dir = os.path.join(data_dir, "wiki")

        # purge previous wiki docs
        filt = {"must": [{"key": "source", "match": {"value": "wiki"}}]}
        ids = await loop.run_in_executor(None, self._collect_ids_sync, filt)
        if ids:
            await loop.run_in_executor(None, lambda: self.q_client.delete(self.collection, ids))

        # git clone / pull
        if os.path.isdir(os.path.join(clone_dir, ".git")):
            subprocess.run(["git", "-C", clone_dir, "pull"], check=False)
        else:
            shutil.rmtree(clone_dir, ignore_errors=True)
            subprocess.run(["git", "clone", repo, clone_dir], check=True)
        await ctx.send("Wiki repo updated – importing …")

        md_files = glob.glob(os.path.join(clone_dir, "*.md"))
        if not md_files:
            return await ctx.send("No markdown pages found – aborting.")

        def _import_page(path: str):
            text = open(path, encoding="utf-8").read()
            html = markdown.markdown(text)
            soup = bs4.BeautifulSoup(html, "html.parser")
            tags = ", ".join({h.get_text(strip=True) for h in soup.find_all(re.compile("^h[1-3]$"))})
            plain = soup.get_text(" ", strip=True)
            self._upsert_sync(tags or os.path.basename(path), plain, "wiki")

        for fp in md_files:
            await loop.run_in_executor(None, _import_page, fp)
        await ctx.send(f"Wiki import done ({len(md_files)} pages).")

    # ------------------------------------------------------------------
    # Init / reset collection
    # ------------------------------------------------------------------

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def initcollection(self, ctx):
        await self.ensure_qdrant()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self.q_client.recreate_collection(
                collection_name=self.collection,
                vectors_config={"size": 384, "distance": "Cosine"},
            ),
        )
        await ctx.send(f"Collection **{self.collection}** recreated.")

    # ------------------------------------------------------------------
    # Ollama chat helper (sync –> run_in_executor)
    # ------------------------------------------------------------------

    def _ollama_chat_sync(self, api: str, model: str, prompt: str) -> str:
        resp = requests.post(
            f"{api}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "(empty response)")

    # ------------------------------------------------------------------
    # Main Q&A logic
    # ------------------------------------------------------------------

    async def _answer(self, question: str) -> str:
        await self.ensure_qdrant()
        # simple heuristic: if GPU型号 mentioned but "resolution" missing → add it
        gpu_words = ("3060", "3070", "3080", "3090", "4060", "4070", "4080", "4090")
        if any(w in question for w in gpu_words) and "resolution" not in question.lower():
            question += " resolution"

        hits = self.q_client.search(self.collection, query_vector=self._vec(question), limit=5)
        if not hits:
            return "No relevant information found."
        ctx_txt = "\n\n".join(f"[{h.id}] {h.payload.get('content','')[:600]}" for h in hits)
        prompt = (
            f"Context:\n{ctx_txt}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely and include Markdown links when relevant."
        )
        api, model = await self.config.api_url(), await self.config.model()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._ollama_chat_sync, api, model, prompt)

    # ------------------------------------------------------------------
    # Public commands / event hooks
    # ------------------------------------------------------------------

    @commands.command(name="askllm")
    async def askllm_cmd(self, ctx, *, question: str):
        async with ctx.typing():
            answer = await self._answer(question)
        await ctx.send(answer)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return
        if self.bot.user.mentioned_in(message):
            q = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
            if q:
                async with message.channel.typing():
                    ans = await self._answer(q)
                await message.channel.send(ans)

    # ------------------------------------------------------------------
    # Simple config setters
    # ------------------------------------------------------------------

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
        await self.config.qdrant_url.set(url.rstrip("/"))
        self.q_client = None  # reconnect next time
        await ctx.send("Qdrant URL updated")

    # ------------------------------------------------------------------
    # on_ready log
    # ------------------------------------------------------------------

    @commands.Cog.listener()
    async def on_ready(self):
        print("LLMManager cog loaded.")


async def setup(bot):
    await bot.add_cog(LLMManager
