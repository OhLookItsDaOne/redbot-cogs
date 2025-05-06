from __future__ import annotations
"""
FusRoh Cog – SkyrimVR Mod‑List Helper
====================================
Red‑DiscordBot cog that teams **Ollama (gemma3:12b)** with a
self‑hosted **Qdrant** vector store so your Discord bot can answer
SkyrimVR‑mod‑list support questions from a curated knowledge base.

*Commands*
----------
``!fusknow`` ‑ add text to KB  •  ``!fusshow`` ‑ list entries  •
``!fusknowdel`` ‑ delete point  •  ``!learn`` ‑ ingest chat  •
``!autotype`` ‑ toggle channel auto‑reply  •  **NEW** ``!fuswipe`` ‑
*owner‑only* command that **removes every collection** from your
Qdrant instance and recreates the default one (handy when schema or
vector size changed).

Hallucination guard: replies *“I’m not sure”* unless a KB hit above the
score threshold is provided.
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
EMBEDDING_DIM = 768  # adjust for other embedding models

# ───────────────────────────────────────────────────────────────────────────
# Qdrant helper
# ───────────────────────────────────────────────────────────────────────────

class QdrantClient:
    def __init__(self, url: str, collection: str = DEFAULT_COLLECTION):
        self.base = url.rstrip("/")
        self.collection = collection

    # ---- low‑level request wrapper ----
    async def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, **kwargs) as resp:
                if resp.status >= 400:
                    txt = await resp.text()
                    raise RuntimeError(f"Qdrant {method} {path} failed: {resp.status} {txt}")
                return await resp.json()

    # ---- collection management ----
    async def recreate_collection(self):
        # drop if exists
        try:
            await self._request("DELETE", f"/collections/{self.collection}")
        except RuntimeError:
            pass
        schema = {"vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"}}
        await self._request("PUT", f"/collections/{self.collection}", json=schema)
        logger.info("Re‑created collection %s", self.collection)

    async def ensure_collection(self):
        try:
            await self._request("GET", f"/collections/{self.collection}")
        except RuntimeError:
            await self.recreate_collection()

    # ---- owner tools ----
    async def drop_all_collections(self):
        data = await self._request("GET", "/collections")
        names = [c["name"] for c in data.get("result", {}).get("collections", [])]
        for name in names:
            await self._request("DELETE", f"/collections/{name}")
        logger.warning("Dropped all collections: %s", ", ".join(names))

    # ---- CRUD ----
    async def upsert(self, pid: int, vec: List[float], payload: Dict[str, Any]):
        await self.ensure_collection()
        body = {"points": [{"id": pid, "vector": vec, "payload": payload}]}
        await self._request("PUT", f"/collections/{self.collection}/points", json=body)

    async def delete_point(self, pid: int):
        await self.ensure_collection()
        await self._request("DELETE", f"/collections/{self.collection}/points", json={"points": [pid]})

    async def search(self, vec: List[float], limit: int = 5):
        await self.ensure_collection()
        body = {
            "vector": vec,
            "limit": limit,
            "with_payload": "true",
            "score_threshold": 0.25,
        }
        res = await self._request("POST", f"/collections/{self.collection}/points/search", json=body)
        return res.get("result", [])

    async def scroll(self, limit: int = 10, offset: int = 0):
        await self.ensure_collection()
        params = {"limit": limit, "offset": offset, "with_payload": "true"}
        res = await self._request("GET", f"/collections/{self.collection}/points", params=params)
        return res.get("result", [])

# ───────────────────────────────────────────────────────────────────────────
# The Red Cog
# ───────────────────────────────────────────────────────────────────────────

class FusRohCog(commands.Cog):
    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=0xF0F5F5)
        self.config.register_global(
            model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333",
            autotype_channels=[],
        )
        self._qdrant: QdrantClient | None = None

    # ---- helpers ----
    async def _qd(self) -> QdrantClient:
        if self._qdrant is None:
            self._qdrant = QdrantClient(await self.config.qdrant_url())
            await self._qdrant.ensure_collection()
        return self._qdrant

    async def _embed(self, text: str) -> List[float]:
        payload = {"model": await self.config.model(), "prompt": text, "stream": False, "options": {"type": "embedding"}}
        async with aiohttp.ClientSession() as s:
            async with s.post(f"{(await self.config.api_url()).rstrip('/')}/api/embeddings", json=payload) as r:
                if r.status >= 400:
                    raise RuntimeError(await r.text())
                return (await r.json())["embedding"]

    async def _chat(self, messages):
        sys = "You are a helpful SkyrimVR‑mod‑list assistant. If nothing relevant is in Knowledge, reply ‘I’m not sure’."
        payload = {"model": await self.config.model(), "stream": False, "messages": [{"role": "system", "content": sys}, *messages]}
        async with aiohttp.ClientSession() as s:
            async with s.post(f"{(await self.config.api_url()).rstrip('/')}/api/chat", json=payload) as r:
                if r.status >= 400:
                    raise RuntimeError(await r.text())
                return (await r.json())["message"]["content"]

    async def _chunk(self, ctx, txt):
        for i in range(0, len(txt), 1990):
            await ctx.send(f"```{txt[i:i+1990]}```")

    # ---- commands ----
    @commands.command()
    async def fusknow(self, ctx, *, text: str):
        pid = int(time.time()*1000)
        await (await self._qd()).upsert(pid, await self._embed(text), {"text": text, "author": str(ctx.author)})
        await ctx.send(f"✅ Saved id `{pid}`")

    @commands.command()
    async def fusshow(self, ctx, count: int = 10, offset: int = 0):
        rows = await (await self._qd()).scroll(count, offset)
        if not rows:
            return await ctx.send("No entries.")
        await self._chunk(ctx, "\n".join(f"• **{r['id']}** – {r['payload']['text'][:150]}…" for r in rows))

    @commands.command()
    async def fusknowdel(self, ctx, point_id: int):
        await (await self._qd()).delete_point(point_id)
        await ctx.send("🗑️ Deleted.")

    @commands.command()
    async def learn(self, ctx, count: int = 5):
        if not 1 <= count <= 20:
            raise BadArgument("Count 1‑20")
        msgs = [m async for m in ctx.channel.history(limit=count+1) if m.id != ctx.message.id]
        msgs.reverse()
        bundle = "\n".join(f"{m.author.display_name}: {m.clean_content}" for m in msgs)
        pid = int(time.time()*1000)
        await (await self._qd()).upsert(pid, await self._embed(bundle), {"text": bundle, "learned": True})
        await ctx.send(f"📚 Learned `{pid}`")

    @commands.command()
    async def autotype(self, ctx, mode: str | None = None):
        cid = ctx.channel.id
        autos = await self.config.autotype_channels()
        if mode is None:
            state = 'on' if cid in autos else 'off'
            return await ctx.send(f"Auto‑typing is **{state}** here.")
        mode = mode.lower()
        if mode == "on":
            if cid not in autos:
                autos.append(cid)
                await self.config.autotype_channels.set(autos)
            await ctx.send("Auto‑typing enabled.")
        elif mode == "off":
            if cid in autos:
                autos.remove(cid)
                await self.config.autotype_channels.set(autos)
            await ctx.send("Auto‑typing disabled.")
        else:
            raise BadArgument("Use on/off")

    # -------- owner‑only wipe --------
    @commands.is_owner()
    @commands.command()
    async def fuswipe(self, ctx):
        """**OWNER ONLY** – delete *all* Qdrant collections and recreate an empty default one."""
        await (await self._qd()).drop_all_collections()
        await (await self._qd()).recreate_collection()
        await ctx.send("💥 All Qdrant collections wiped and fresh KB created.")

    # -------- listener --------
    @commands.Cog.listener()
    async def on_message_without_command(self, message):
        if not message.guild or message.author.bot or message.author == self.bot.user:
            return
        ctx = await self.bot.get_context(message)
        if ctx.valid:
            return
        autos = await self.config.autotype_channels()
        if message.channel.id not in autos and self.bot.user not in message.mentions:
            return
        hist = [m async for m in message.channel.history(limit=5)]
        hist.reverse()
        context = [{"role": "user" if m.author == message.author else "assistant", "content": m.clean_content} for m in hist]
        hits = await (await self._qd()).search(await self._embed(message.clean_content))
        if hits:
            kb = "\n\n".join(f"* {h['payload']['text']}" for h in hits)
            context.append({"role": "system", "content": f"Knowledge:\n{kb}"})
        try:
            reply = await self._chat(context)
        except Exception as e:
            logger.exception("LLM fail: %s", e)
            return
        if "i’m not sure" in reply.lower():
            return
        await message.channel.send(reply)

# ───────────────────────────────────────────────────────────────────────────
# Red loader
# ───────────────────────────────────────────────────────────────────────────

async def setup(bot: Red):
    await bot.add_cog(FusRohCog(bot))
