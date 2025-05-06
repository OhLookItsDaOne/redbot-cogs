from __future__ import annotations
"""
FusRoh Cog – SkyrimVR Mod‑List Helper
====================================
Red‑DiscordBot cog that pairs **Ollama** chat (gemma3:12b by default)
with a local **Sentence‑Transformers** embedder and **Qdrant** vector
store so the bot can answer SkyrimVR‑mod‑list support questions.

**Embedding model now:** **intfloat/e5‑large‑v2** (1024‑dim vectors).
This model delivers top‑tier retrieval quality.  It is downloaded
automatically on first run and will use GPU if `torch.cuda.is_available()`;
otherwise it runs happily on CPU (≈1 GB RAM load, encode ~200 ms / text
on Ryzen‑class CPUs).

Commands: `!fusknow`, `!fusshow`, `!fusknowdel`, `!learn`, `!autotype`,
`!fuswipe`.

> *The database is **not wiped** on restart.  `!fuswipe` stays available
> if you ever want a clean slate.*
"""

import logging
import time
from typing import Any, Dict, List

import aiohttp
from redbot.core import Config, commands
from redbot.core.bot import Red
from redbot.core.commands import BadArgument

# ── Local embedding model ────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import torch  # type: ignore
except ImportError:  # pragma: no cover – optional dep
    SentenceTransformer = None  # type: ignore
    torch = None  # type: ignore

EMBED_MODEL = "intfloat/e5-large-v2"
EMBED_DIM = 1024

logger = logging.getLogger("red.fusrohcog")
DEFAULT_COLLECTION = "fusroh_support"

# ─────────────────────────────────────────────────────────────────────────
# Qdrant helper
# ─────────────────────────────────────────────────────────────────────────

class QdrantClient:
    def __init__(self, url: str, collection: str = DEFAULT_COLLECTION):
        self.base = url.rstrip("/")
        self.collection = collection

    async def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, **kwargs) as resp:
                if resp.status >= 400:
                    txt = await resp.text()
                    raise RuntimeError(f"Qdrant {method} {path} {resp.status}: {txt}")
                return await resp.json()

    # collection management
    async def recreate_collection(self):
        try:
            await self._request("DELETE", f"/collections/{self.collection}")
        except RuntimeError:
            pass
        schema = {"vectors": {"size": EMBED_DIM, "distance": "Cosine"}}
        await self._request("PUT", f"/collections/{self.collection}", json=schema)
        logger.info("Created collection %s (%d‑dims)", self.collection, EMBED_DIM)

    async def ensure(self):
        try:
            await self._request("GET", f"/collections/{self.collection}")
        except RuntimeError:
            await self.recreate_collection()

    async def drop_all(self):
        res = await self._request("GET", "/collections")
        for c in [c["name"] for c in res.get("result", {}).get("collections", [])]:
            await self._request("DELETE", f"/collections/{c}")
        logger.warning("Dropped all collections")

    # CRUD
    async def upsert(self, pid: int, vec: List[float], payload: Dict[str, Any]):
        await self._request("PUT", f"/collections/{self.collection}/points", json={"points": [{"id": pid, "vector": vec, "payload": payload}]})

    async def delete(self, pid: int):
        await self._request("DELETE", f"/collections/{self.collection}/points", json={"points": [pid]})

    async def search(self, vec: List[float], limit: int = 5):
        res = await self._request("POST", f"/collections/{self.collection}/points/search", json={
            "vector": vec,
            "limit": limit,
            "with_payload": True,
            "score_threshold": 0.25,
        })
        return res.get("result", [])

    async def scroll(self, limit: int = 10, offset: int = 0):
        try:
            res = await self._request("GET", f"/collections/{self.collection}/points", params={"limit": limit, "offset": offset, "with_payload": True})
        except RuntimeError as exc:
            if "404" in str(exc):
                return []
            raise
        return res.get("result", [])

# ─────────────────────────────────────────────────────────────────────────
# Cog
# ─────────────────────────────────────────────────────────────────────────

class FusRohCog(commands.Cog):
    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=0xF0F5F5)
        self.config.register_global(
            chat_model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333",
            autotype_channels=[],
        )
        self._qd: QdrantClient | None = None
        self._st: SentenceTransformer | None = None

    # ---------- helpers ----------
    async def _qd_client(self) -> QdrantClient:
        if self._qd is None:
            self._qd = QdrantClient(await self.config.qdrant_url())
            await self._qd.ensure()  # ensure once, never wipe automatically
        return self._qd

    async def _embed(self, text: str) -> List[float]:
        if SentenceTransformer is None:
            raise RuntimeError("Please `pip install sentence-transformers` to use this cog.")
        if self._st is None:
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            self._st = SentenceTransformer(EMBED_MODEL, device=device)
            logger.info("Loaded %s on %s", EMBED_MODEL, device)
        return self._st.encode(text, convert_to_numpy=True).tolist()

    async def _chat(self, messages):
        sys = "You are a helpful SkyrimVR‑mod‑list assistant. If no answer from Knowledge, reply ‘I’m not sure’."
        payload = {"model": await self.config.chat_model(), "stream": False, "messages": [{"role": "system", "content": sys}, *messages]}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{(await self.config.api_url()).rstrip('/')}/api/chat", json=payload) as resp:
                if resp.status >= 400:
                    raise RuntimeError(await resp.text())
                return (await resp.json())["message"]["content"]

    async def _chunk_send(self, ctx, text):
        for i in range(0, len(text), 1990):
            await ctx.send(f"```{text[i:i+1990]}```")

    # ---------- commands ----------
    @commands.command()
    async def fusknow(self, ctx, *, text: str):
        pid = int(time.time() * 1000)
        await (await self._qd_client()).upsert(pid, await self._embed(text), {"text": text, "author": str(ctx.author)})
        await ctx.send(f"✅ Saved `{pid}`")

    @commands.command()
    async def fusshow(self, ctx, count: int = 10, offset: int = 0):
        rows = await (await self._qd_client()).scroll(count, offset)
        if not rows:
            return await ctx.send("No entries.")
        await self._chunk_send(ctx, "\n".join(f"• **{r['id']}** – {r['payload']['text'][:150]}…" for r in rows))

    @commands.command()
    async def fusknowdel(self, ctx, point_id: int):
        await (await self._qd_client()).delete(point_id)
        await ctx.send("🗑️ Deleted.")

    @commands.command()
    async def learn(self, ctx, count: int = 5):
        if not 1 <= count <= 20:
            raise BadArgument("Count 1‑20")
        msgs = [m async for m in ctx.channel.history(limit=count + 1) if m.id != ctx.message.id]
        msgs.reverse()
        bundle = "\n".join(f"{m.author.display_name}: {m.clean_content}" for m in msgs)
        pid = int(time.time() * 1000)
        await (await self._qd_client()).upsert(pid, await self._embed(bundle), {"text": bundle, "learned": True})
        await ctx.send(f"📚 Learned `{pid}`")

    @commands.command()
    async def autotype(self, ctx, mode: str | None = None):
        cid = ctx.channel.id
        autos = await self.config.autotype_channels()
        if mode is None:
            return await ctx.send(f"Auto‑typing is **{'on' if cid in autos else 'off'}** here.")
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

    @commands.is_owner()
    @commands.command()
    async def fuswipe(self, ctx):
        qd = await self._qd_client()
        await qd.drop_all()
        await qd.recreate_collection()
        await ctx.send("💥 Qdrant wiped and fresh collection created.")

    # ---------- listener ----------
        # ---------- listener ----------
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

        history = [m async for m in message.channel.history(limit=5)]
        history.reverse()
        ctx_msgs = [{"role": "user" if m.author == message.author else "assistant", "content": m.clean_content} for m in history]
        hits = await (await self._qd_client()).search(await self._embed(message.clean_content))
        if hits:
            kb = "

".join(f"* {h['payload']['text']}" for h in hits)
            ctx_msgs.append({"role": "system", "content": f"Knowledge:
{kb}"})

        try:
            reply = await self._chat(ctx_msgs)
        except Exception as exc:
            logger.exception("Chat error: %s", exc)
            return
        if "i’m not sure" in reply.lower():
            return
        await message.channel.send(reply)

# ── loader ───────────────────────────────────────────────────────────────
async def setup(bot: Red):
    await bot.add_cog(FusRohCog(bot))
