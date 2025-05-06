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
import numpy as np
import aiohttp
from redbot.core import Config, commands
from redbot.core.bot import Red
from redbot.core.commands import BadArgument
from sentence_transformers import CrossEncoder
# ── Local embedding model ────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import torch  # type: ignore
except ImportError:  # pragma: no cover – optional dep
    SentenceTransformer = None  # type: ignore
    torch = None  # type: ignore

EMBED_MODEL = "intfloat/e5-large-v2"
EMBED_DIM = 1024
# ------------------------------------------------------------------
# Konstante ganz OBEN in der Datei (neben EMBED_DIM etc.)
HIT_THRESHOLD = 0.30
# ------------------------------------------------------------------
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
        res = await self._request(
            "POST",
            f"/collections/{self.collection}/points/search",
            json={
                "vector": vec,
                "limit":  limit,
                "with_payload": True,
                "with_vectors": True,          #  <‑ ergänzen
                "score_threshold": 0.25,
            },
        )
        return res.get("result", [])
        
    async def scroll(self, limit: int = 10, offset: int = 0):
        body = {"limit": limit, "with_payload": True}
        if offset:
            body["offset"] = offset
        try:
            res = await self._request(
                "POST",
                f"/collections/{self.collection}/points/scroll",
                json=body,
            )
        except RuntimeError as exc:
            if "404" in str(exc):
                return []
            raise

        data = res.get("result", {})
        # new ⇣ pick the actual list of points
        if isinstance(data, dict) and "points" in data:
            return data["points"]
        return data  # fallback for older API shapes

# ─────────────────────────────────────────────────────────────────────────
# Cog
# ─────────────────────────────────────────────────────────────────────────

class FusRohCog(commands.Cog):
    def __init__(self, bot: Red):
        self.bot = bot
        # ---------- Red‑Config ----------
        self.config = Config.get_conf(self, identifier=0xF0F5F5)
        self.config.register_global(
            chat_model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333",
            autotype_channels=[],
        )
        # ---------- Vektor‑ & LLM‑Runtime ----------
        self._qd: QdrantClient | None = None
        self._st: SentenceTransformer | None = None
        # Cross‑Encoder initialisieren **hier**, nicht ganz oben
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self._ce = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=device,
        )
        logger.info("Loaded Cross‑Encoder on %s", device)
        
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

    # ---------- kleine Ausgabe‑Hilfe ----------
    async def _chunk_send(self, ctx: commands.Context, text: str):
        """Schickt langen Text in 2000‑Zeichen‑Chunks an Discord."""
        for i in range(0, len(text), 1990):
            await ctx.send(f"```{text[i:i+1990]}```")
     # ---------- Hilfsfunktionen ----------
    @staticmethod
    def chunk_text(text: str, tokens: int = 120, overlap: int = 20):
        words = text.split()
        step  = tokens - overlap
        for i in range(0, len(words), step):
            yield " ".join(words[i : i + tokens])

    @staticmethod
    def mmr(query_vec, doc_vecs, k: int = 3, λ: float = 0.7):
        """Maximal Marginal Relevance – gibt die Indexliste der besten k Vektoren zurück."""
        query = np.asarray(query_vec)
        sim   = [np.dot(query, d) / (np.linalg.norm(query) * np.linalg.norm(d)) for d in doc_vecs]

        selected, selected_ids = [], set()
        for _ in range(k):
            mmr_scores = [
                λ * s - (1 - λ) * max(
                    np.dot(d, doc_vecs[j]) / (np.linalg.norm(d) * np.linalg.norm(doc_vecs[j]))
                    for j in selected_ids
                ) if i not in selected_ids else -1
                for i, (s, d) in enumerate(zip(sim, doc_vecs))
            ]
            nxt = int(np.argmax(mmr_scores))
            selected.append(nxt)
            selected_ids.add(nxt)
        return selected

    # ---------- commands ----------
    @commands.command()
    async def fusknow(self, ctx, *, text: str):
        for chunk in self.chunk_text(text):
            pid  = int(time.time()*1000)
            vec  = await self._embed(chunk)
            meta = {"author": str(ctx.author), "source": ctx.message.jump_url}
            await (await self._qd_client()).upsert(pid, vec, {"text": chunk, **meta})
        await ctx.send("✅ Added.")

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
    @commands.Cog.listener()
    async def on_message_without_command(self, message):
        # ---- Vorbedingungen -------------------------------------------------
        if not message.guild or message.author.bot or message.author == self.bot.user:
            return
        ctx = await self.bot.get_context(message)
        if ctx.valid:
            return

        autos = await self.config.autotype_channels()
        if message.channel.id not in autos and self.bot.user not in message.mentions:
            return

        # ---- Chat‑Historie als Kontext --------------------------------------
        history = [m async for m in message.channel.history(limit=5)]
        history.reverse()
        ctx_msgs = [
            {
                "role": "user" if m.author == message.author else "assistant",
                "content": m.clean_content,
            }
            for m in history
        ]

        # ---- Vektor‑Suche ----------------------------------------------------
        query_vec = await self._embed(message.clean_content)
        hits = await (await self._qd_client()).search(query_vec, limit=8)

        # 1. Score‑Schwelle
        hits = [h for h in hits if h["score"] >= HIT_THRESHOLD]
        if not hits:
            await message.channel.send("I’m not sure.")
            return

        # 2. MMR‑Diversität (Top‑5)
        doc_vecs = [h["vector"] for h in hits]          # <- vectors braucht 'with_vectors':True
        best_idx = self.mmr(query_vec, doc_vecs, k=5, λ=0.7)
        hits     = [hits[i] for i in best_idx]

        # 3. Cross‑Encoder‑Rerank (Top‑2 höchster Präzision)
        pairs   = [(message.clean_content, h["payload"]["text"]) for h in hits]
        scores  = self._ce.predict(pairs)
        hits    = [h for h, s in sorted(zip(hits, scores), key=lambda x: -x[1])][:2]

        # ---- Knowledge‑Block für Gemma --------------------------------------
        kb = "\n\n".join(f"* {h['payload']['text'][:300]}…" for h in hits)
        ctx_msgs.append({"role": "system", "content": f"Knowledge:\n{kb}"})

        # ---- LLM‑Antwort -----------------------------------------------------
        try:
            reply = await self._chat(ctx_msgs)
        except Exception as exc:
            logger.exception("Chat error: %s", exc)
            return

        if "i’m not sure" in reply.lower():
            await message.channel.send(reply)
            return

        await message.channel.send(reply)

# ── loader ───────────────────────────────────────────────────────────────
async def setup(bot: Red):
    await bot.add_cog(FusRohCog(bot))
