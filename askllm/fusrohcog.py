from __future__ import annotations
"""
FusRoh Cog – SkyrimVR Mod‑List Helper
====================================
Connects a Red‑DiscordBot to **Ollama** for chat (gemma3:12b by default)
plus **Qdrant** for long‑term knowledge storage.  Now supports *local*
embeddings via **Sentence‑Transformers** so you are no longer limited to
models that expose an /embeddings endpoint.

Commands (unchanged)
--------------------
`!fusknow`, `!fusshow`, `!fusknowdel`, `!learn`, `!autotype`, `!fuswipe`

New ‑‑ Settings
---------------
* ``[p]set fusroh embedder local`` – switch to local ST embeddings.
* ``[p]set fusroh embedder ollama`` – use Ollama’s /embeddings route
  (requires a model that supports it, e.g. *nomic‑embed‑text*).

Local default model: **all‑mpnet‑base‑v2** (768‑dim).  GPU is used if
`torch.cuda.is_available()`.
"""

import logging
import textwrap
import time
from typing import Any, Dict, List

import aiohttp
from redbot.core import Config, commands
from redbot.core.bot import Red
from redbot.core.commands import BadArgument

# local embedding
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # missing dependency – we’ll warn later
    SentenceTransformer = None  # type: ignore

logger = logging.getLogger("red.fusrohcog")
DEFAULT_COLLECTION = "fusroh_support"
LOCAL_EMBED_MODEL = "all-mpnet-base-v2"
LOCAL_EMBED_DIM = 768

# ───────────────────────────────────────────────────────────────────────────
# Qdrant helper
# ───────────────────────────────────────────────────────────────────────────

class QdrantClient:
    def __init__(self, url: str, collection: str = DEFAULT_COLLECTION):
        self.base = url.rstrip("/")
        self.collection = collection

    async def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base}{path}"
        async with aiohttp.ClientSession() as s:
            async with s.request(method, url, **kwargs) as r:
                if r.status >= 400:
                    txt = await r.text()
                    raise RuntimeError(f"Qdrant {method} {path} {r.status}: {txt}")
                return await r.json()

    # collections
    async def recreate_collection(self, dim: int):
        try:
            await self._request("DELETE", f"/collections/{self.collection}")
        except RuntimeError:
            pass
        schema = {"vectors": {"size": dim, "distance": "Cosine"}}
        await self._request("PUT", f"/collections/{self.collection}", json=schema)
        logger.info("Created collection %s (%d‑dims)", self.collection, dim)

    async def ensure(self, dim: int):
        try:
            await self._request("GET", f"/collections/{self.collection}")
        except RuntimeError:
            await self.recreate_collection(dim)

    async def drop_all(self):
        res = await self._request("GET", "/collections")
        for c in [c["name"] for c in res.get("result", {}).get("collections", [])]:
            await self._request("DELETE", f"/collections/{c}")
        logger.warning("Dropped all collections")

    # points
    async def upsert(self, pid: int, vec: List[float], payload: Dict[str, Any]):
        body = {"points": [{"id": pid, "vector": vec, "payload": payload}]}
        await self._request("PUT", f"/collections/{self.collection}/points", json=body)

    async def delete(self, pid: int):
        await self._request("DELETE", f"/collections/{self.collection}/points", json={"points": [pid]})

    async def search(self, vec: List[float], limit: int = 5):
        body = {
            "vector": vec,
            "limit": limit,
            "with_payload": True,
            "score_threshold": 0.25,
        }
        res = await self._request("POST", f"/collections/{self.collection}/points/search", json=body)
        return res.get("result", [])

    async def scroll(self, limit: int = 10, offset: int = 0):
        try:
            res = await self._request(
                "GET", f"/collections/{self.collection}/points",
                params={"limit": limit, "offset": offset, "with_payload": True},
            )
        except RuntimeError as exc:
            if "404" in str(exc):
                return []
            raise
        return res.get("result", [])

# ───────────────────────────────────────────────────────────────────────────
# Cog
# ───────────────────────────────────────────────────────────────────────────

class FusRohCog(commands.Cog):
    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=0xF0F5F5)
        self.config.register_global(
            chat_model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333",
            embedder="local",  # or "ollama"
            autotype_channels=[],
        )
        self._qd: QdrantClient | None = None
        self._st: SentenceTransformer | None = None

    # ---------- utility ----------
    async def _get_dim(self) -> int:
        emb = await self.config.embedder()
        if emb == "local":
            return LOCAL_EMBED_DIM
        return 768  # typical Ollama embed models – could be dynamic

    async def _qd_client(self) -> QdrantClient:
        if self._qd is None:
            self._qd = QdrantClient(await self.config.qdrant_url())
            await self._qd.ensure(await self._get_dim())
        return self._qd

    async def _embed(self, text: str) -> List[float]:
        if (await self.config.embedder()) == "local":
            if SentenceTransformer is None:
                raise RuntimeError("sentence‑transformers not installed. pip install sentence-transformers")
            if self._st is None:
                self._st = SentenceTransformer(LOCAL_EMBED_MODEL, device="cuda" if SentenceTransformer and SentenceTransformer._hf_backend.exists("cuda") else "cpu")
            vec = self._st.encode(text, convert_to_numpy=True)
            return vec.tolist()
        # else use Ollama embeddings API
        payload = {"model": await self.config.chat_model(), "prompt": text, "stream": False, "options": {"type": "embedding"}}
        async with aiohttp.ClientSession() as s:
            async with s.post(f"{(await self.config.api_url()).rstrip('/')}/api/embeddings", json=payload) as r:
                if r.status >= 400:
                    raise RuntimeError(f"Ollama embeddings error {r.status}: {await r.text()}")
                return (await r.json())["embedding"]

    async def _chat(self, msgs):
        sys = "You are a helpful SkyrimVR‑mod‑list assistant. If no answer from Knowledge, reply ‘I’m not sure’."
        payload = {"model": await self.config.chat_model(), "stream": False, "messages": [{"role": "system", "content": sys}, *msgs]}
        async with aiohttp.ClientSession() as s:
            async with s.post(f"{(await self.config.api_url()).rstrip('/')}/api/chat", json=payload) as r:
                if r.status >= 400:
                    raise RuntimeError(await r.text())
                return (await r.json())["message"]["content"]

    async def _chunk_send(self, ctx, txt):
        for i in range(0, len(txt), 1990):
            await ctx.send(f"```{txt[i:i+1990]}```")

    # ---------- commands ----------
    @commands.command()
    async def fusknow(self, ctx, *, text: str):
        vec = await self._embed(text)
        pid = int(time.time()*1000)
        await (await self._qd_client()).upsert(pid, vec, {"text": text, "author": str(ctx.author)})
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
        msgs = [m async for m in ctx.channel.history(limit=count+1) if m.id != ctx.message.id]
        msgs.reverse()
        chunk = "\n".join(f"{m.author.display_name}: {m.clean_content}" for m in msgs)
        pid = int(time.time()*1000)
        await (await self._qd_client()).upsert(pid, await self._embed(chunk), {"text": chunk, "learned": True})
        await ctx.send(f"📚 Learned `{pid}`")

    @commands.command()
    async def autotype(self, ctx, mode: str | None = None):
        cid = ctx.channel.id
        autos = await self.config.autotype_channels()
        if mode is None:
            return await ctx.send(f"Auto‑typing is **{'on' if cid in autos else 'off'}** here.")
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
            raise BadArgument("Use on/off")

    @commands.is_owner()
    @commands.command()
    async def fuswipe(self, ctx):
        qd = await self._qd_client()
        await qd.drop_all()
        await qd.recreate_collection(await self._get_dim())
        await ctx.send("💥 Qdrant wiped and fresh collection created.")

    # listener
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
        ctx_msgs = [{"role": "user" if m.author == message.author else "assistant", "content": m.clean_content} for m in hist]
        hits = await (await self._qd_client()).search(await self._embed(message.clean_content))
        if hits:
            kb = "\n\n".join(f"* {h['payload']['text']}" for h in hits)
            ctx_msgs.append({"role": "system", "content": f"Knowledge:\n{kb}"})
        try:
            reply = await self._chat(ctx_msgs)
        except Exception as e:
            logger.exception("Chat fail: %s", e)
            return
        if "i’m not sure" in reply.lower():
            return
        await message.channel.send(reply)

# loader
async def setup(bot: Red):
    await bot.add_cog(FusRohCog(bot))
