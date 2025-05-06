from __future__ import annotations
"""
FusRoh Cog – SkyrimVR Mod‑List Helper
====================================
Red‑DiscordBot cog that teams **Ollama (gemma3:12b)** with a
self‑hosted **Qdrant** vector store so your Discord bot can answer
SkyrimVR‑mod‑list support questions from a curated knowledge base.

Key points
----------
* Commands: ``!fusknow``, ``!fusshow``, ``!fusknowdel``, ``!learn``,
  ``!autotype``.
* Auto‑replies in designated channels (or when mentioned).
* **Hallucination guard** – replies "I’m not sure" unless KB provides
  a strong match.
* On start‑up Qdrant collection is (re‑)created to avoid schema drift.
"""

import logging
import textwrap
import time
from typing import Any, Dict, List

import aiohttp
from redbot.core import Config, commands
from redbot.core.bot import Red
from redbot.core.commands import BadArgument

logger = logging.getLogger("red.fusrohcog")
DEFAULT_COLLECTION = "fusroh_support"
EMBEDDING_DIM = 768  # Gemma default – change if you swap model

# ---------------------------------------------------------------------------
# Qdrant minimal async wrapper
# ---------------------------------------------------------------------------

class QdrantClient:
    """Tiny async HTTP wrapper for the Qdrant REST API."""

    def __init__(self, url: str, collection: str = DEFAULT_COLLECTION):
        self.base = url.rstrip("/")
        self.collection = collection

    # ------------------------- low‑level helper -------------------------
    async def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, **kwargs) as resp:
                if resp.status >= 400:
                    txt = await resp.text()
                    raise RuntimeError(f"Qdrant {method} {path} failed: {resp.status} {txt}")
                return await resp.json()

    # ---------------------- collection management ----------------------
    async def recreate_collection(self):
        """Drop existing collection (if any) and create a fresh one."""
        # Delete if present
        try:
            await self._request("DELETE", f"/collections/{self.collection}")
        except RuntimeError:
            pass  # 404 – not present, that’s fine
        # Create new
        schema = {"vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"}}
        await self._request("PUT", f"/collections/{self.collection}", json=schema)
        logger.info("Re‑created Qdrant collection %s", self.collection)

    async def ensure_collection(self):
        try:
            await self._request("GET", f"/collections/{self.collection}")
        except RuntimeError:
            await self.recreate_collection()

    # ----------------------------- CRUD -------------------------------
    async def upsert(self, point_id: int, vector: List[float], payload: Dict[str, Any]):
        await self.ensure_collection()
        data = {"points": [{"id": point_id, "vector": vector, "payload": payload}]}
        await self._request("PUT", f"/collections/{self.collection}/points", json=data)

    async def delete_point(self, point_id: int):
        await self.ensure_collection()
        data = {"points": [point_id]}
        await self._request("DELETE", f"/collections/{self.collection}/points", json=data)

    async def search(self, vector: List[float], limit: int = 5):
        await self.ensure_collection()
        body = {
            "vector": vector,
            "limit": limit,
            "with_payload": True,
            "score_threshold": 0.25,
        }
        resp = await self._request("POST", f"/collections/{self.collection}/points/search", json=body)
        return resp.get("result", [])

    async def scroll(self, limit: int = 10, offset: int = 0):
        await self.ensure_collection()
        params = {"limit": limit, "offset": offset, "with_payload": True}
        resp = await self._request("GET", f"/collections/{self.collection}/points", params=params)
        return resp.get("result", [])

# ---------------------------------------------------------------------------
# Cog
# ---------------------------------------------------------------------------

class FusRohCog(commands.Cog):
    """Attach with `[p]load FusRoh`"""

    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=0xFUSFUS)
        self.config.register_global(
            model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333",
            autotype_channels=[],
        )
        self._qdrant: QdrantClient | None = None

    # --------------------------- helpers ---------------------------
    async def _get_qdrant(self) -> QdrantClient:
        if self._qdrant is None:
            self._qdrant = QdrantClient(await self.config.qdrant_url())
            await self._qdrant.recreate_collection()  # Start clean each boot
        return self._qdrant

    async def _create_embedding(self, text: str) -> List[float]:
        url = await self.config.api_url()
        payload = {
            "model": await self.config.model(),
            "prompt": text,
            "stream": False,
            "options": {"type": "embedding"},
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{url.rstrip('/')}/api/embeddings", json=payload) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"Embeddings error {resp.status}: {await resp.text()}")
                return (await resp.json())["embedding"]

    async def _generate_reply(self, context: List[Dict[str, str]]) -> str:
        url = await self.config.api_url()
        sys_prompt = (
            "You are a helpful support assistant for the SkyrimVR mod‑list community. "
            "If no answer is found in the Knowledge section, say ‘I’m not sure’."
        )
        payload = {
            "model": await self.config.model(),
            "stream": False,
            "messages": [{"role": "system", "content": sys_prompt}, *context],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{url.rstrip('/')}/api/chat", json=payload) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"Chat error {resp.status}: {await resp.text()}")
                return (await resp.json())["message"]["content"]

    async def _chunk_send(self, ctx: commands.Context, text: str):
        for i in range(0, len(text), 1990):
            await ctx.send(f"```{text[i:i+1990]}```")

    @staticmethod
    def _pid() -> int:
        return int(time.time() * 1000)

    # ----------------------------- commands -----------------------------
    @commands.command()
    async def fusknow(self, ctx: commands.Context, *, text: str):
        """Add *text* to the knowledge base."""
        vec = await self._create_embedding(text)
        pid = self._pid()
        await (await self._get_qdrant()).upsert(pid, vec, {"text": text, "author": str(ctx.author)})
        await ctx.send(f"✅ Saved with id `{pid}`")

    @commands.command()
    async def fusshow(self, ctx: commands.Context, count: int = 10, offset: int = 0):
        """Show recent entries."""
        rows = await (await self._get_qdrant()).scroll(count, offset)
        if not rows:
            await ctx.send("No entries.")
            return
        out = [f"• **{r['id']}** – {r['payload']['text'][:150]}…" for r in rows]
        await self._chunk_send(ctx, "\n".join(out))

    @commands.command()
    async def fusknowdel(self, ctx: commands.Context, point_id: int):
        await (await self._get_qdrant()).delete_point(point_id)
        await ctx.send("🗑️ Deleted.")

    @commands.command()
    async def learn(self, ctx: commands.Context, count: int = 5):
        if not 1 <= count <= 20:
            raise BadArgument("Count 1‑20.")
        msgs = [m async for m in ctx.channel.history(limit=count+1) if m.id != ctx.message.id]
        msgs.reverse()
        bundle = "\n".join(f"{m.author.display_name}: {m.clean_content}" for m in msgs)
        pid = self._pid()
        await (await self._get_qdrant()).upsert(pid, await self._create_embedding(bundle), {"text": bundle, "learned": True})
        await ctx.send(f"📚 Learned `{pid}`.")

    @commands.command()
    async def autotype(self, ctx: commands.Context, mode: str | None = None):
        cid = ctx.channel.id
        autos = await self.config.autotype_channels()
        if mode is None:
            await ctx.send(f"Auto‑typing is **{'on' if cid in autos else 'off'}** here.")
            return
        if mode.lower() == "on":
            if cid not in autos:
                autos.append(cid)
                await self.config.autotype_channels.set(autos)
            await ctx.send("Auto‑typing enabled.")
        elif mode.lower() == "off":
            if cid in autos:
                autos.remove(cid)
                await self.config.autotype_channels.set(autos)
            await ctx.send("Auto‑typing disabled.")
        else:
            raise BadArgument("Use on/off.")

    # --------------------------- listener ---------------------------
    @commands.Cog.listener()
    async def on_message_without_command(self, message):
        if (not message.guild) or message.author.bot or message.author == self.bot.user:
            return
        ctx = await self.bot.get_context(message)
        if ctx.valid:
            return
        autos = await self.config.autotype_channels()
        if message.channel.id not in autos and self.bot.user not in message.mentions:
            return

        history = [m async for m in message.channel.history(limit=5)]
        history.reverse()
        context = [{"role": "user" if m.author == message.author else "assistant", "content": m.clean_content} for m in history]

        emb = await self._create_embedding(message.clean_content)
        hits = await (await self._get_qdrant()).search(emb)
        if hits:
            kb = "\n\n".join(f"* {h['payload']['text']}" for h in hits)
            context.append({"role": "system", "content": f"Knowledge:\n{kb}"})

        try:
            reply = await self._generate_reply(context)
        except Exception as e:
            logger.exception("LLM error: %s", e)
            return
        if "i’m not sure" in reply.lower():
            return
        await message.channel.send(reply)

# ---------------------------------------------------------------------------
# Red loader
# ---------------------------------------------------------------------------

async def setup(bot: Red):
    await bot.add_cog(FusRohCog(bot))
