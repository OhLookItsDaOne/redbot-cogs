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
# Helper – auto‑install a PyPI package inside the container if it is missing
# -----------------------------------------------------------------------------

def _ensure_pkg(module: str, pip_name: str | None = None):
    try:
        __import__(module)
    except ModuleNotFoundError:
        subprocess.check_call(["python", "-m", "pip", "install", pip_name or module])
        __import__(module)

# Wiki‑import & chat‑learning need these two only for that part
_ensure_pkg("markdown")
_ensure_pkg("bs4", "beautifulsoup4")
import markdown  # noqa: E402 – imported after ensure
import bs4       # noqa: E402 – imported after ensure

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

    # -------------- internal helpers ----------------------------------

    async def ensure_qdrant(self):
        if self.q_client is None:
            url = await self.config.qdrant_url()
            self.q_client = QdrantClient(url=url)

    def _vec(self, txt: str) -> List[float]:
        return self.embedder.encode(txt).tolist()

    # -------------- low‑level Qdrant helpers (sync, run in executor) ---

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
            [{
                "id": uuid.uuid4().int & ((1 << 64) - 1),
                "vector": self._vec(f"{tag}. {content}"),
                "payload": {"tag": tag, "content": content, "source": source},
            }],
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
        """List all KB entries (chunked ≤ 2000 chars)."""
        await self.ensure_qdrant()
        pts, _ = self.q_client.scroll(self.collection, with_payload=True, limit=1000)
        if not pts:
            return await ctx.send("No knowledge entries stored.")

        header, footer, max_len = "```\n", "```", 2000
        chunk, out = header, []
        for p in pts:
            pl = p.payload or {}
            snippet = pl.get("content", "")[:280].replace("\n", " ")
            line = f"[{p.id}] ({pl.get('tag','NoTag')}, {pl.get('source','?')}): {snippet}\n"
            if len(chunk) + len(line) > max_len - len(footer):
                out.append(chunk + footer)
                chunk = header + line
            else:
                chunk += line
        out.append(chunk + footer)
        for ch in out:
            await ctx.send(ch)

    @commands.command()
    async def llmknowdelete(self, ctx, doc_id: int):
        """Delete entry by **ID**."""
        await self.ensure_qdrant()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self.q_client.delete(self.collection, [doc_id]))
        await ctx.send(f"Deleted entry {doc_id}.")

    @commands.command()
    async def llmknowdeletetag(self, ctx, tag: str):
        """Delete every entry that has *tag*."""
        await self.ensure_qdrant()
        loop = asyncio.get_running_loop()
        filt = {"must": [{"key": "tag", "match": {"value": tag.lower()}}]}
        ids = await loop.run_in_executor(None, self._collect_ids_sync, filt)
        if ids:
            await loop.run_in_executor(None, lambda: self.q_client.delete(self.collection, ids))
        await ctx.send(f"Deleted entries with tag '{tag.lower()}'.")

    @commands.command()
    async def llmknowdeletelast(self, ctx, tag: str = "chat"):
        """Delete **latest** snippet stored under *tag* (default `chat`)."""
        await self.ensure_qdrant()
        loop = asyncio.get_running_loop()
        filt = {"must": [{"key": "tag", "match": {"value": tag.lower()}}]}
        ids = await loop.run_in_executor(None, self._collect_ids_sync, filt)
        if not ids:
            return await ctx.send(f"No entry found for tag '{tag}'.")
        last_id = max(ids)
        await loop.run_in_executor(None, lambda: self.q_client.delete(self.collection, [last_id]))
        await ctx.send(f"Deleted latest entry with tag '{tag}' (ID {last_id}).")

    # ------------------------------------------------------------------
    # Learn from recent chat messages
    # ------------------------------------------------------------------

    async def _summarise_chunks(self, text: str) -> List[str]:
        """Split big text into ≤ 300‑char bullets via the LLM."""
        api, model = await self.config.api_url(), await self.config.model()
        prompt = (
            "Split the following text into self‑contained bullet points (each ≤ 300 characters).\n"
            "Return one bullet per line without any numbering.\n\n" + text
        )
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, self._ollama_chat_sync, api, model, prompt)
        bullets = [b.strip("•- ") for b in raw.split("\n") if b.strip()]
        return bullets[:20]

    @commands.command()
    async def learn(self, ctx, limit: int = 50, tag: str = "chat"):
        """Learn from the last **limit** user messages in this channel."""
        await self.ensure_qdrant()
        limit = max(5, min(200, limit))
        msgs = [
            m.content for m in await ctx.channel.history(limit=limit, oldest_first=False).flatten()
            if not m.author.bot and m.content
        ]
        if not msgs:
            return await ctx.send("No user messages found to learn from.")

        combined = "\n".join(reversed(msgs))
        bullets = await self._summarise_chunks(combined) if len(combined) > 1200 else [combined]

        loop = asyncio.get_running_loop()
        for bl in bullets:
            await loop.run_in_executor(None, self._upsert_sync, tag.lower(), bl, "learn")
        await ctx.send(f"Learned {len(bullets)} snippet(s) under tag '{tag}'.")

    # ------------------------------------------------------------------
    # GitHub‑Wiki import
    # ------------------------------------------------------------------

    @commands.command()
    async def importwiki(self, ctx, repo: str = "https://github.com/Kvitekvist/FUS.wiki.git"):
        """Clone / pull a GitHub Wiki and re‑import it (source=wiki)."""
        await self.ensure_qdrant()
        loop = asyncio.get_running_loop()
        data_dir = str(cog_data_path(self)); os.makedirs(data_dir, exist_ok=True)
        clone_dir = os.path.join(data_dir, "wiki")

        # delete old wiki docs
        filt = {"must": [{"key": "source", "match": {"value": "wiki"}}]}
        ids = await loop.run_in_executor(None, self._collect_ids_sync, filt)
        if ids:
            await loop.run_in_executor(None, lambda: self.q_client.delete(self.collection, ids))

        # clone or pull
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
            soup = bs4.BeautifulSoup(markdown.markdown(text), "html.parser")
            tags = ", ".join({h.get_text(strip=True) for h in soup.find_all(re.compile("^h[1-3]$"))})
            plain = soup.get_text(" ", strip=True)
            self._upsert_sync(tags or os.path.basename(path), plain, "wiki")

        for fp in md_files:
            await loop.run_in_executor(None, _import_page, fp)
        await ctx.send(f"Wiki import done ({len(md_files)} pages).")

    # ------------------------------------------------------------------
    # Collection initialisation / reset
    # ------------------------------------------------------------------

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def initcollection(self, ctx):
        """Create the Qdrant collection (or recreate it)."""
        await self.ensure_qdrant()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ensure_collection)
        await ctx.send(f"Collection '{self.collection}' is ready.")

    # ------------------------------------------------------------------
    # LLM querying helpers
    # ------------------------------------------------------------------

    def _ollama_chat_sync(self, api: str, model: str, prompt: str) -> str:
        """Blocking call to local Ollama (non‑stream)."""
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
        return resp.json()["message"]["content"]

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
        model, api = await self.config.model(), await self.config.api_url()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._ollama_chat_sync, api, model, prompt)

    # ------------------------------------------------------------------
    # Commands – ask the LLM
    # ------------------------------------------------------------------

    @commands.command()
    async def askllm(self, ctx, *, question: str):
        await ctx.typing()
        await ctx.send(await self.ask_with_context(question))

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return
        if self.bot.user.mentioned_in(message):
            q = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
            if q:
                await message.channel.typing()
                await message.channel.send(await self.ask
