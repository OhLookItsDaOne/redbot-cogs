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

# ------------------------------------------------------------
# helper: ensure that a PyPI package is available (Docker safe)
# ------------------------------------------------------------
def _ensure_pkg(module: str, pip_name: str | None = None):
    """Install *module* inside the running container if missing."""
    try:
        __import__(module)
    except ModuleNotFoundError:
        subprocess.check_call(
            ["python", "-m", "pip", "install", pip_name or module],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        __import__(module)


# Markdown → HTML + HTML parsing are only required for the wiki‑importer
_ensure_pkg("markdown")
_ensure_pkg("bs4", "beautifulsoup4")
from markdown import markdown                # noqa: E402 (imported after _ensure_pkg)
from bs4 import BeautifulSoup                # noqa: E402


# ------------------------------------------------------------
# LLMManager – Qdrant‑backed KB  +  local Ollama chat
# ------------------------------------------------------------
class LLMManager(commands.Cog):
    """Interact with a local Ollama LLM via a Qdrant knowledge base."""

    # ------------------------------------------------------------------
    # Init / basic helpers
    # ------------------------------------------------------------------
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9876543210123)
        self.config.register_global(
            model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333",
        )

        self.collection = "fus_wiki"
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 384‑D vectors
        self.q_client: QdrantClient | None = None
        self._last_manual_id: int | None = None  # ID des letzten !llmknow‑Eintrags

    # ---------- helpers ------------------------------------------------
    async def ensure_qdrant(self):
        if self.q_client is None:
            url = await self.config.qdrant_url()
            self.q_client = QdrantClient(url=url)

    def _vec(self, txt: str) -> List[float]:
        """Return SBERT embedding for *txt*."""
        return self.embedder.encode(txt).tolist()

    # ---------- low‑level Qdrant ops (sync, run in executor) -----------
    def _ensure_collection(self):
        """Create collection if missing."""
        try:
            self.q_client.get_collection(self.collection)
        except Exception:
            self.q_client.recreate_collection(
                collection_name=self.collection,
                vectors_config={"size": 384, "distance": "Cosine"},
            )

    def _upsert_sync(self, tag: str, content: str, source: str) -> int:
        """Insert/update doc and return its point‑ID."""
        self._ensure_collection()
        new_id = uuid.uuid4().int & ((1 << 64) - 1)  # 64‑bit random
        self.q_client.upsert(
            self.collection,
            [
                {
                    "id": new_id,
                    "vector": self._vec(f"{tag}. {content}"),
                    "payload": {"tag": tag, "content": content, "source": source},
                }
            ],
        )
        return new_id

    def _collect_ids_sync(self, filt: dict) -> List[int]:
        """Collect all point‑IDs matching *filt*."""
        ids, offset = [], None
        while True:
            pts, offset = self.q_client.scroll(
                self.collection, limit=1000, with_payload=False,
                scroll_filter=filt, offset=offset,
            )
            ids.extend(p.id for p in pts)
            if offset is None:
                break
        return ids

    # ------------------------------------------------------------------
    # Commands – knowledge management
    # ------------------------------------------------------------------
    @commands.command()
    async def llmknow(self, ctx: commands.Context, tag: str, *, content: str):
        """Add manual knowledge under **tag**."""
        await self.ensure_qdrant()
        loop = asyncio.get_running_loop()
        new_id = await loop.run_in_executor(None, self._upsert_sync, tag.lower(), content, "manual")
        self._last_manual_id = new_id
        await ctx.send(f"Added manual info under '{tag.lower()}' (ID {new_id}).")

    @commands.command()
    async def llmknowshow(self, ctx: commands.Context):
        """List stored entries (chunked ≤ 2 000 chars)."""
        await self.ensure_qdrant()
        pts, _ = self.q_client.scroll(self.collection, with_payload=True, limit=1000)
        if not pts:
            return await ctx.send("No knowledge entries stored.")

        header, footer, max_len = "```\n", "```", 2000
        chunk, out = header, []
        for p in pts:
            pl = p.payload or {}
            snippet = pl.get("content", "")[:280].replace("\n", " ")
            line = f"[{p.id}] ({pl.get('tag','NoTag')},src={pl.get('source','?')}): {snippet}\n"
            if len(chunk) + len(line) > max_len - len(footer):
                out.append(chunk + footer)
                chunk = header + line
            else:
                chunk += line
        out.append(chunk + footer)
        for ch in out:
            await ctx.send(ch)

    @commands.command()
    async def llmknowdelete(self, ctx: commands.Context, doc_id: int):
        """Delete entry by **ID**."""
        await self.ensure_qdrant()
        await asyncio.get_running_loop().run_in_executor(None, lambda: self.q_client.delete(self.collection, [doc_id]))
        await ctx.send(f"Deleted entry {doc_id}.")

    @commands.command()
    async def llmknowdeletetag(self, ctx: commands.Context, tag: str):
        """Delete every entry whose *tag* matches (case‑insensitive)."""
        await self.ensure_qdrant()
        filt = {"must": [{"key": "tag", "match": {"value": tag.lower()}}]}
        ids = await asyncio.get_running_loop().run_in_executor(None, self._collect_ids_sync, filt)
        if ids:
            await asyncio.get_running_loop().run_in_executor(None, lambda: self.q_client.delete(self.collection, ids))
        await ctx.send(f"Deleted entries with tag '{tag.lower()}'.")

    @commands.command()
    async def llmknowdeletelast(self, ctx: commands.Context):
        """Delete the most recently added manual entry (this session)."""
        if self._last_manual_id is None:
            return await ctx.send("No manual entry recorded this session.")
        await self.ensure_qdrant()
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.q_client.delete(self.collection, [self._last_manual_id])
        )
        await ctx.send(f"Deleted last manual entry (ID {self._last_manual_id}).")
        self._last_manual_id = None

    # ------------------------------------------------------------------
    # GitHub‑Wiki import
    # ------------------------------------------------------------------
    @commands.command()
    async def importwiki(
        self,
        ctx: commands.Context,
        repo: str = "https://github.com/Kvitekvist/FUS.wiki.git",
    ):
        """Clone / pull GitHub Wiki and import (source = wiki)."""
        await self.ensure_qdrant()
        data_dir = str(cog_data_path(self)); os.makedirs(data_dir, exist_ok=True)
        clone_dir = os.path.join(data_dir, "wiki")

        # 1) purge previous wiki docs
        filt = {"must": [{"key": "source", "match": {"value": "wiki"}}]}
        ids = await asyncio.get_running_loop().run_in_executor(None, self._collect_ids_sync, filt)
        if ids:
            await asyncio.get_running_loop().run_in_executor(None, lambda: self.q_client.delete(self.collection, ids))

        # 2) git clone / pull
        if os.path.isdir(os.path.join(clone_dir, ".git")):
            subprocess.run(["git", "-C", clone_dir, "pull"], check=False)
        else:
            shutil.rmtree(clone_dir, ignore_errors=True)
            subprocess.run(["git", "clone", repo, clone_dir], check=True)
        await ctx.send("Wiki repo updated – importing …")

        # 3) iterate *.md files
        md_files = glob.glob(os.path.join(clone_dir, "*.md"))
        if not md_files:
            return await ctx.send("No markdown pages found – aborting.")

        def _import_page(path: str):
            text = open(path, encoding="utf-8").read()
            html = markdown(text)
            soup = BeautifulSoup(html, "html.parser")
            tags = ", ".join({h.get_text(strip=True) for h in soup.find_all(re.compile(r"^h[1-3]$"))})
            plain = soup.get_text(" ", strip=True)
            self._upsert_sync(tags or os.path.basename(path), plain, "wiki")

        loop = asyncio.get_running_loop()
        for fp in md_files:
            await loop.run_in_executor(None, _import_page, fp)
        await ctx.send(f"Wiki import done ({len(md_files)} pages).")

    # ------------------------------------------------------------------
    # Init / reset collection
    # ------------------------------------------------------------------
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def initcollection(self, ctx: commands.Context):
        """Recreate the Qdrant collection from scratch."""
        await self.ensure_qdrant()
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.q_client.recreate_collection(
                collection_name=self.collection,
                vectors_config={"size": 384, "distance": "Cosine"},
            ),
        )
        await ctx.send(f"Collection **{self.collection}** recreated.")

    # ------------------------------------------------------------------
    # Ollama chat helper (sync → run_in_executor)
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
        return resp.json().get("message", {}).get("content", "<no response>")

    # ------------------------------------------------------------------
    # Main Q&A logic
    # ------------------------------------------------------------------
    async def _answer(self, question: str) -> str:
        await self.ensure_qdrant()
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
    async def askllm_cmd(self, ctx: commands.Context, *, question: str):
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
    async def setmodel(self, ctx: commands.Context, model):
        await self.config.model.set(model)
        await ctx.send(f"Model set to {model}")

    @commands.command()
    async def setapi(self, ctx: commands.Context, url):
        await self.config.api_url.set(url.rstrip("/"))
        await ctx.send("API URL updated")

    @commands.command()
    async def setqdrant(self, ctx: commands.Context, url):
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
    """Red‑DiscordBot loading hook"""
    await bot.add_cog(LLMManager(bot))
