# askllmc.py  –  hybrid Qdrant + local-Ollama   (Red-DiscordBot cog)

import asyncio
import os
import re
import subprocess
import uuid
import json
from typing import List

import discord
import requests
from redbot.core import commands, Config
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, http

# ---------------------------------------------------------------------------
# helper – install missing PyPI packages on-the-fly (safe in Docker)
# ---------------------------------------------------------------------------
def _ensure_pkg(mod: str, pip_name: str | None = None):
    try:
        __import__(mod)
    except ModuleNotFoundError:
        subprocess.check_call(
            ["python", "-m", "pip", "install", pip_name or mod],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        __import__(mod)

_ensure_pkg("sentence_transformers")
_ensure_pkg("qdrant_client")


# ----------------------------------------------------------------------------
# LLMManager
# ----------------------------------------------------------------------------
class LLMManager(commands.Cog):
    """Interact with a local Ollama LLM through a Qdrant knowledge base."""

    # --------------------------------------------------------------------
    # init
    # --------------------------------------------------------------------
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9876543210123)
        self.config.register_global(
            model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333",
            auto_channels=[],
        )

        self.collection = "fus_wiki"

        # stronger model (768-Dim)
        self.embedder = SentenceTransformer("all-mpnet-base-v2")
        self.vec_dim = self.embedder.get_sentence_embedding_dimension()

        self.q_client: QdrantClient | None = None
        self._last_manual_id: int | None = None  # last !llmknow id (session)
        # track last hits for image sending
        self._last_ranked_hits: List = []

    # --------------------------------------------------------------------
    # basic helpers
    # --------------------------------------------------------------------
    async def ensure_qdrant(self):
        if self.q_client is None:
            self.q_client = QdrantClient(url=await self.config.qdrant_url())

    def _vec(self, txt: str) -> List[float]:
        return self.embedder.encode(txt).tolist()

    # --------------------------------------------------------------------
    # upsert manual entry (extract only true image URLs)
    # --------------------------------------------------------------------
    def _upsert_sync(self, tag: str, content: str, source: str) -> int:
        self._ensure_collection()
        # 1) Extract only explicit markdown image URLs (png/jpg/gif/webp)
        image_urls = re.findall(
            r'!\[.*?\]\((https?://[^\s\)]+\.(?:png|jpe?g|gif|webp)(?:\?[^)\s]*)?)\)',
            content,
            flags=re.IGNORECASE,
        )
        # 2) Remove image markup but keep normal links inline
        txt = re.sub(r'!\[.*?\]\((https?://[^\s\)]+)\)', '', content)
        txt = re.sub(r'\[([^\]]+)\]\((https?://[^\s\)]+)\)', r'\1: \2', txt).strip()
        # 3) Build vector and payload
        pid = uuid.uuid4().int & ((1 << 64) - 1)
        vec = self._vec(f"{tag}. {tag}. {txt}")
        payload = {"tag": tag, "content": txt, "source": source}
        if image_urls:
            payload["images"] = image_urls
        # 4) Upsert
        self.q_client.upsert(
            self.collection,
            [{"id": pid, "vector": vec, "payload": payload}],
        )
        return pid

    # --------------------------------------------------------------------
    # low-level Qdrant helpers (sync → executor)
    # --------------------------------------------------------------------
    def _ensure_collection(self, force: bool = False):
        """Create / recreate collection so its vector-dim matches the embedder."""
        try:
            info = self.q_client.get_collection(self.collection)
            size = info.config.params.vectors.size  # type: ignore
            if size != self.vec_dim or force:
                raise ValueError("dim mismatch → recreate")
        except Exception:
            self.q_client.recreate_collection(
                collection_name=self.collection,
                vectors_config={
                    "size": self.vec_dim,
                    "distance": "Cosine",
                    "hnsw_config": {"m": 16, "ef_construct": 200},
                },
                optimizers_config={
                    "default_segment_number": 4,
                    "indexing_threshold": 256,
                },
                wal_config={
                    "wal_capacity_mb": 1024,
                },
                payload_indexing_config={
                    "enable": True,
                    "field_schema": {
                        "tag":     {"type": "keyword"},
                        "source":  {"type": "keyword"},
                        "content": {"type": "text"},    # ← Volltextsuche auf content
                    },
                },
                compression_config={
                    "type": "ProductQuantization",
                    "params": {"segments": 8, "subvector_size": 2},
                },
            )

    def _collect_ids_sync(self, filt: dict) -> List[int]:
        ids, offset = [], None
        while True:
            pts, offset = self.q_client.scroll(
                self.collection,
                limit=1000,
                with_payload=False,
                scroll_filter=filt,
                offset=offset,
            )
            ids.extend(p.id for p in pts)
            if offset is None:
                break
        return ids

    # --------------------------------------------------------------------
    # Commands: add/remove/move image URLs
    # --------------------------------------------------------------------
    @commands.command(name="llmknowaddimg")
    @commands.has_permissions(administrator=True)
    async def llmknowaddimg(self, ctx, doc_id: int, url: str):
        """Adds an image URL to an entry—only if it looks like an actual image file."""
        if not re.search(r'\.(?:png|jpe?g|gif|webp)(?:\?.*)?$', url, flags=re.IGNORECASE):
            return await ctx.send("⚠️ The provided URL does not appear to be an image.")
        await self.ensure_qdrant()
        pts = self.q_client.retrieve(self.collection, [doc_id], with_payload=True)
        if not pts:
            return await ctx.send(f"Entry {doc_id} not found.")
        payload = pts[0].payload or {}
        images = payload.get("images", [])
        if url in images:
            return await ctx.send("⚠️ This image URL is already stored.")
        images.append(url)
        self.q_client.set_payload(
            collection_name=self.collection,
            payload={"images": images},
            points=[doc_id],
        )
        await ctx.send(f"✅ Image URL added to entry {doc_id}.")

    @commands.command(name="llmknowrmimg")
    @commands.has_permissions(administrator=True)
    async def llmknowrmimg(self, ctx, doc_id: int, url: str):
        """Removes an image URL from an existing entry."""
        await self.ensure_qdrant()
        pts = self.q_client.retrieve(self.collection, [doc_id], with_payload=True)
        if not pts:
            return await ctx.send(f"Entry {doc_id} not found.")
        payload = pts[0].payload or {}
        images = payload.get("images", [])
        if url not in images:
            return await ctx.send("URL not present.")
        images.remove(url)
        self.q_client.set_payload(
            collection_name=self.collection,
            payload={"images": images},
            points=[doc_id],
        )
        await ctx.send(f"✅ Image URL removed from entry {doc_id}.")

    @commands.command(name="llmknowmvimg")
    @commands.has_permissions(administrator=True)
    async def llmknowmvimg(self, ctx, doc_id: int, from_pos: int, to_pos: int):
        """Moves an image URL within an entry's payload."""
        await self.ensure_qdrant()
        pts = self.q_client.retrieve(self.collection, [doc_id], with_payload=True)
        if not pts:
            return await ctx.send(f"Entry {doc_id} not found.")
        images = pts[0].payload.get("images", [])
        if not (1 <= from_pos <= len(images)):
            return await ctx.send(f"Invalid from_pos: {from_pos}. Only {len(images)} images.")
        url = images.pop(from_pos - 1)
        to_pos = max(1, min(to_pos, len(images) + 1))
        images.insert(to_pos - 1, url)
        self.q_client.set_payload(
            collection_name=self.collection,
            payload={"images": images},
            points=[doc_id],
        )
        await ctx.send(f"✅ Image moved from position {from_pos} to {to_pos}.")

    # --------------------------------------------------------------------
    # learn command (unchanged)
    # --------------------------------------------------------------------
    @commands.command(name="learn")
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, num: int):
        # … your existing learn implementation …
        ...

    # --------------------------------------------------------------------
    # Manual-knowledge commands
    # --------------------------------------------------------------------
    @commands.command()
    async def llmknow(self, ctx, tag: str, *, content: str):
        """Add manual knowledge under **tag**."""
        await self.ensure_qdrant()
        loop = asyncio.get_running_loop()
        new_id = await loop.run_in_executor(None, self._upsert_sync, tag.lower(), content, "manual")
        self._last_manual_id = new_id
        await ctx.send(f"Added manual info under '{tag.lower()}' (ID {new_id}).")

    @commands.command(name="llmknowshow")
    async def llmknowshow(self, ctx):
        """Show entries with clearly visible image lists."""
        await self.ensure_qdrant()
        pts, _ = self.q_client.scroll(self.collection, with_payload=True, limit=1000)
        if not pts:
            return await ctx.send("No knowledge entries stored.")
        hdr, ftr, maxlen = "```\n", "```", 2000
        cur, chunks = hdr, []
        for p in pts:
            pl = p.payload or {}
            line = f"[{p.id}] ({pl.get('tag','NoTag')},{pl.get('source','?')}): "
            line += pl.get("content","")[:200].replace('\n',' ')
            images = pl.get("images", [])
            if images:
                line += "\n  → Images:\n" + "\n".join(f"    • {u}" for u in images)
            line += "\n"
            if len(cur) + len(line) > maxlen - len(ftr):
                chunks.append(cur + ftr)
                cur = hdr + line
            else:
                cur += line
        chunks.append(cur + ftr)
        for c in chunks:
            await ctx.send(c)

    @commands.command()
    async def llmknowdelete(self, ctx, doc_id: int):
        await self.ensure_qdrant()
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.q_client.delete(self.collection, [doc_id])
        )
        await ctx.send(f"Deleted entry {doc_id}.")

    @commands.command()
    async def llmknowdeletetag(self, ctx, tag: str):
        await self.ensure_qdrant()
        filt = {"must": [{"key": "tag", "match": {"value": tag.lower()}}]}
        ids = await asyncio.get_running_loop().run_in_executor(None, self._collect_ids_sync, filt)
        if ids:
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: self.q_client.delete(self.collection, ids)
            )
        await ctx.send(f"Deleted entries with tag '{tag.lower()}'.")

    @commands.command()
    async def llmknowdeletelast(self, ctx):
        if self._last_manual_id is None:
            return await ctx.send("No manual entry recorded this session.")
        await self.ensure_qdrant()
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.q_client.delete(self.collection, [self._last_manual_id])
        )
        await ctx.send(f"Deleted last manual entry (ID {self._last_manual_id}).")
        self._last_manual_id = None

    # --------------------------------------------------------------------
    # main Q&A
    # --------------------------------------------------------------------
    async def _answer(self, question: str) -> str:
        await self.ensure_qdrant()
        loop = asyncio.get_running_loop()

        # 1) Heuristic query expansion
        aug_q = question
        ql = question.lower()
        if "virtual" in ql and "desktop" in ql and "resolution" not in ql:
            aug_q += " resolution"

        # 2) Build keyword list for optional boost
        clean = re.sub(r"[^\w\s]", " ", aug_q.lower())
        kws = [w for w in clean.split() if len(w) > 2]

        # 3) Manual-only vector search + keyword SHOULD-boost
        manual_filter = {"must": [{"key": "source", "match": {"value": "manual"}}]}
        if kws:
            manual_filter["should"] = (
                [{"key": "tag",     "match": {"value": k}} for k in kws] +
                [{"key": "content", "match": {"value": k}} for k in kws]
            )

        manual_hits = await loop.run_in_executor(
            None,
            lambda: self.q_client.search(
                collection_name=self.collection,
                query_vector=self._vec(aug_q),
                query_filter=manual_filter,
                limit=5,
                with_payload=True,
            ),
        )

        if not manual_hits:
            return "No relevant information found."

        # 4) Remember only these hits for image sending
        self._last_ranked_hits = manual_hits

        # 5) Build context & prompt
        ctx = "\n\n".join(
            " ".join(h.payload["content"].split()[:200]) for h in manual_hits
        )
        final_prompt = (
            "Use **only** the facts below to answer. If the facts are insufficient, say so.\n\n"
            f"### Facts ###\n{ctx}\n\n"
            f"### Question ###\n{question}\n\n"
            "### Answer ###"
        )
        return await loop.run_in_executor(
            None,
            self._ollama_chat_sync,
            await self.config.api_url(),
            await self.config.model(),
            final_prompt,
        )

    @commands.command(name="askllm")
    async def askllm_cmd(self, ctx, *, question: str):
        async with ctx.typing():
            ans = await self._answer(question)
        await ctx.send(ans)
        # Only send images for manual hits actually used
        for h in getattr(self, "_last_ranked_hits", []):
            for url in h.payload.get("images", []):
                embed = discord.Embed()
                embed.set_image(url=url)
                await ctx.send(embed=embed)

    @commands.Cog.listener()
    async def on_message(self, m: discord.Message):
        if m.author.bot or not m.guild:
            return

        autolist = await self.config.auto_channels()
        if self.bot.user.mentioned_in(m) or m.content.startswith("!askllm"):
            q = m.clean_content.replace(f"@{self.bot.user.display_name}", "").strip()
        elif m.channel.id in autolist:
            q = m.content.strip()
        else:
            return
        if not q:
            return

        # Reset last hits
        self._last_ranked_hits = []

        # Get answer
        try:
            async with m.channel.typing():
                ans = await self._answer(q)
        except http.exceptions.ResponseHandlingException as e:
            return await m.channel.send(f"⚠️ Could not connect: {e}")

        await m.channel.send(ans)

        # Only send images for manual hits actually used
        for h in self._last_ranked_hits:
            for url in h.payload.get("images", []):
                embed = discord.Embed()
                embed.set_image(url=url)
                await m.channel.send(embed=embed)

    # --------------------------------------------------------------------
    # simple setters
    # --------------------------------------------------------------------
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
        self.q_client = None
        await ctx.send("Qdrant URL updated")  # reconnect next time

    # --------------------------------------------------------------------
    # on_ready
    # --------------------------------------------------------------------
    @commands.Cog.listener()
    async def on_ready(self):
        print("LLMManager cog loaded.")


async def setup(bot):
    await bot.add_cog(LLMManager(bot))
