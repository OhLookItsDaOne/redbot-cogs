from __future__ import annotations
import logging
import re
import time
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
from redbot.core import Config, commands
from redbot.core.bot import Red
from redbot.core.commands import BadArgument
from sentence_transformers import CrossEncoder
try:
    from sentence_transformers import SentenceTransformer 
    import torch 
except ImportError:
    SentenceTransformer = None
    torch = None 

EMBED_MODEL = "intfloat/e5-large-v2"
EMBED_DIM = 1024
logger = logging.getLogger("red.fusrohcog")
DEFAULT_COLLECTION = "fusroh_support"

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
                    raise RuntimeError(
                        f"Qdrant {method} {path} {resp.status}: {txt}"
                    )
                return await resp.json()
                
    async def recreate_collection(self):
        try:
            await self._request("DELETE", f"/collections/{self.collection}")
        except RuntimeError:
            pass
        schema = {"vectors": {"size": EMBED_DIM, "distance": "Dot"}}
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

    async def upsert(self, pid: int, vec: List[float], payload: Dict[str, Any]):
        await self._request(
            "PUT",
            f"/collections/{self.collection}/points",
            json={"points": [{"id": pid, "vector": vec, "payload": payload}]},
        )

    async def delete(self, pid: int):
        await self._request(
            "DELETE", f"/collections/{self.collection}/points", json={"points": [pid]}
        )

    async def search(
        self,
        vec: List[float],
        limit: int = 5,
        qfilter: Optional[Dict[str, Any]] = None,
    ):
        body = {
            "vector": vec,
            "limit": limit,
            "with_payload": True,
            "with_vectors": True,
        }
        if qfilter:
            body["filter"] = qfilter
        res = await self._request(
            "POST", f"/collections/{self.collection}/points/search", json=body
        )
        return res.get("result", [])

    async def scroll(self, limit: int = 10, offset: int = 0, qfilter: dict | None = None):
        body = {"limit": limit, "with_payload": True}
        if offset:
            body["offset"] = offset
        if qfilter:
            body["filter"] = qfilter
        try:
            res = await self._request(
                "POST", f"/collections/{self.collection}/points/scroll", json=body
            )
        except RuntimeError as exc:
            if "404" in str(exc):
                return []
            raise

        data = res.get("result", {})
        if isinstance(data, dict) and "points" in data:
            return data["points"]
        return data

class FusRohCog(commands.Cog):
    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=0xF0F5F5)
        self.config.register_global(
            chat_model="deepseek-r1:14b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333",
            autotype_channels=[],
            vec_thr=0.25,
            ce_thr=0.20,
        )
        self._qd: QdrantClient | None = None
        self._st: SentenceTransformer | None = None
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self._ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
        logger.info("Loaded Cross‑Encoder on %s", device)

    async def _qd_client(self) -> QdrantClient:
        if self._qd is None:
            self._qd = QdrantClient(await self.config.qdrant_url())
            await self._qd.ensure()
        return self._qd

    async def _embed(self, text: str) -> List[float]:
        if SentenceTransformer is None:
            raise RuntimeError("Please `pip install sentence-transformers` to use this cog.")
        if self._st is None:
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            self._st = SentenceTransformer(EMBED_MODEL, device=device)
            logger.info("Loaded %s on %s", EMBED_MODEL, device)
        return self._st.encode(text, convert_to_numpy=True).tolist()

    async def _chunk_send(self, ctx: commands.Context, text: str):
        for i in range(0, len(text), 1990):
            await ctx.send(f"```{text[i:i+1990]}```")

    @staticmethod
    def chunk_text(text: str, tokens: int = 80, overlap: int = 15):
        words = text.split()
        step = tokens - overlap
        for i in range(0, len(words), step):
            yield " ".join(words[i : i + tokens])

    @staticmethod
    def clean_discord_text(txt: str) -> str:
        txt = re.sub(r'^>.*$', '', txt, flags=re.M)          
        txt = re.sub(r'^#+\\s+', '', txt, flags=re.M)        
        txt = re.sub(r'^\\s*\\w{2,20}:\\s*', '', txt)        
        return txt.strip()

    @staticmethod
    def mmr(query_vec, doc_vecs, k: int = 3, λ: float = 0.7):
        query = np.asarray(query_vec)
        sim = [np.dot(query, d) / (np.linalg.norm(query) * np.linalg.norm(d)) for d in doc_vecs]
        selected, selected_ids = [], set()
        for _ in range(min(k, len(doc_vecs))):
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

    async def _vec_thr(self) -> float:
        return await self.config.vec_thr()

    async def _ce_thr(self) -> float:
        return await self.config.ce_thr()

    @staticmethod
    def _split_tags(raw: str) -> tuple[list[str], str]:
        """
        [tag1 tag2] eigentlicher Text  →  (['tag1','tag2'], 'eig. Text')
        """
        m = re.match(r"\s*\[([^]]+)]\s*(.+)", raw, flags=re.S)
        if not m:
            return [], raw.strip()
        tag_str, body = m.groups()
        tags = re.split(r"[,\s]+", tag_str.strip())
        return [t.lower() for t in tags if t], body.strip()

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def fusknow(self, ctx, *, text: str):
        """Fügt Wissen hinzu – Tags optional in eckigen Klammern."""
        doc_id = int(time.time() * 1000)
        tags, clean = self._split_tags(text)
        for chunk in self.chunk_text(clean):
            pid = int(time.time() * 1000)
            vec = await self._embed(chunk)
            payload = {
                "text": chunk,
                "author": str(ctx.author),
                "source": ctx.message.jump_url,
                "doc_id": doc_id,            
            }
            if tags:
                payload["tags"] = tags
            await (await self._qd_client()).upsert(pid, vec, payload)
        await ctx.send("✅ Added.")
        
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def fusshow(self, ctx, count: int = 10, offset: int = 0, *, flag: str = ""):
        full = "--full" in flag
        rows = await (await self._qd_client()).scroll(count, offset)
        if not rows:
            return await ctx.send("No entries.")
    
        lines = []
        for r in rows:
            txt = r["payload"]["text"]
            if not full:
                txt = txt[:150] + ("…" if len(txt) > 150 else "")
            lines.append(f"• **{r['id']}** – {txt}")
        await self._chunk_send(ctx, "\n".join(lines))
    @commands.command()
    async def fusget(self, ctx, point_id: int):
        qd = await self._qd_client()
        res = await qd.scroll(limit=1, offset=0)         
        res = [p for p in res if p["id"] == point_id]
        if not res:
            return await ctx.send("ID not found.")
        payload = res[0]["payload"]
        text = payload["text"]
        links = "\n".join(payload.get("links", []))
        await self._chunk_send(ctx, f"**{point_id}**\n{text}\n{links}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def fusknowdel(self, ctx, point_id: int):
        await (await self._qd_client()).delete(point_id)
        await ctx.send("🗑️ Deleted.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, count: int = 5):
        if not 1 <= count <= 20:
            raise BadArgument("Count 1‑20")
        msgs = [
            m async for m in ctx.channel.history(limit=count + 1)
            if m.id != ctx.message.id
        ]
        msgs.reverse()

        clean_lines, links = [], []
        for m in msgs:
            clean_lines.append(self.clean_discord_text(m.clean_content))
            links += re.findall(r'https?://\S+', m.content)  

        bundle = "\n".join(clean_lines)                   
        payload = {"text": bundle, "learned": True}
        if links:
            payload["links"] = links

        pid = int(time.time() * 1000)
        await (await self._qd_client()).upsert(
            pid, await self._embed(bundle), payload
        )
        await ctx.send(f"📚 Learned `{pid}`")

    @commands.command()
    async def autotype(self, ctx, mode: str | None = None):
        cid = ctx.channel.id
        autos = await self.config.autotype_channels()
        if mode is None:
            return await ctx.send(
                f"Auto‑typing is **{'on' if cid in autos else 'off'}** here."
            )
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

    @commands.command()
    async def fusthreshold(
        self, ctx, vec: float | None = None, ce: float | None = None
    ):
        """`!fusthreshold` → zeigt;  `!fusthreshold 0.22 0.18` → setzt."""
        if vec is None and ce is None:
            v, c = await self._vec_thr(), await self._ce_thr()
            await ctx.send(f"Vector‑Thr **{v:.2f}**, CE‑Thr **{c:.2f}**")
            return
        if vec is not None:
            if not 0 < vec < 1:
                raise BadArgument("vec zwischen 0 und 1")
            await self.config.vec_thr.set(vec)
        if ce is not None:
            if not 0 <= vec < 1:
                raise BadArgument("vec zwischen 0 und 1 (0.0 erlaubt)")
            await self.config.ce_thr.set(ce)
        await ctx.send("✅ Schwellen gespeichert.")

    @commands.is_owner()
    @commands.command()
    async def fuswipe(self, ctx):
        qd = await self._qd_client()
        await qd.drop_all()
        await qd.recreate_collection()
        await ctx.send("💥 Qdrant wiped and fresh collection created.")
        
    async def _chat(self, messages):
        sys_prompt = (
            "You are a SkyrimVR-support assistant using only the provided Knowledge.\n"
            "1) Read the Knowledge facts and pick the relevant ones.\n"
            "2) Formulate a concise answer without repeating all facts verbatim.\n"
            "3) If you lack enough information, reply exactly “I’m not sure.”\n"
            "Cite each fact you use with [#] based on its position in the Knowledge block."
            
        )
        body = {
            "model": await self.config.chat_model(),
            "stream": False,
            "messages": [{"role": "system", "content": sys_prompt}, *messages],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{(await self.config.api_url()).rstrip('/')}/api/chat", json=body
            ) as resp:
                if resp.status >= 400:
                    raise RuntimeError(await resp.text())
                data = await resp.json()
        return data["message"]["content"]    
    
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
        qd = await self._qd_client()
        history = [m async for m in message.channel.history(limit=5)]
        history.reverse()
        ctx_msgs = [
            {"role": "user" if m.author == message.author else "assistant", "content": m.clean_content}
            for m in history
        ]
        query_vec = await self._embed(message.clean_content)
        all_points = await qd.scroll(limit=100)
        alle_tags = {t for p in all_points for t in p["payload"].get("tags", [])}
        await message.channel.send(f"🏷️ Verfügbare Tags in DB: {sorted(alle_tags)}")
        word_tokens = set(re.findall(r"[A-Za-z0-9_\-]+", message.clean_content.lower()))
        want_tags = [t for t in word_tokens if t in alle_tags]
        await message.channel.send(f"🎯 Gesuchte Tags aus Frage: {want_tags}")
        hits = await qd.search(query_vec, limit=8)
        for h in hits:
            txt = h["payload"]["text"][:60].replace("\n", " ")
            tags = h["payload"].get("tags", [])
            score = round(h.get("score", 0), 3)
            await message.channel.send(f"id={h['id']} score={score} tags={tags} text=\"{txt}…\"")
    
        vec_thr = await self._vec_thr()
        hits = [h for h in hits if h["score"] >= vec_thr]
        if not hits:
            return await message.channel.send("I’m not sure.")
        selected = None
        prompt_note = ""
        if want_tags:
            points = await qd.scroll(limit=200)
            # exakte Treffer: alle Tags müssen drin sein
            direct = [
                p for p in points
                if all(t in p["payload"].get("tags", []) for t in want_tags)
            ]
            if direct:
                selected = direct
                prompt_note = f"Treffer mit allen Tags: {', '.join(want_tags)}."
            else:
                # Fallback: mindestens ein Tag
                fallback = [
                    p for p in points
                    if any(t in p["payload"].get("tags", []) for t in want_tags)
                ]
                if fallback:
                    selected = fallback
                    prompt_note = (
                        f"Keine exakten Tag-Treffer; nutze Einträge mit einem der Tags: "
                        f"{', '.join(want_tags)}."
                    )
        
        # 2) Vektor-Fallback nur, wenn überhaupt keine Tag-Treffer
        if selected is None:
            # Wir haben ja schon das erste `hits = await qd.search(...)`
            # und den Debug ausgegeben – also nur noch filtern:
            vec_thr = await self._vec_thr()
            hits = [h for h in hits if h["score"] >= vec_thr]
            if not hits:
                return await message.channel.send("I’m not sure.")
            selected = hits
            prompt_note = "Nutze semantisch ähnliche Einträge (Vektorsuche)."
        
        # 3) Chunks laden und Knowledge-Block bauen
        kb_texts = []
        for h in selected:
            doc_id = h["payload"]["doc_id"]
            chunks = await qd.scroll(
                limit=100,
                qfilter={"must":[{"key":"doc_id","match":{"value":doc_id}}]},
            )
            chunks.sort(key=lambda c: c["id"])
            kb_texts.append(" ".join(c["payload"]["text"] for c in chunks))
        
        kb = prompt_note + "\n\n" + "\n\n".join(f"* {txt[:500]}…" for txt in kb_texts)
        ctx_msgs.append({"role":"system","content":f"Knowledge:\n{kb}"})
        
        # 4) Einmalige LLM-Abfrage und Antwort versenden
        raw = await self._chat(ctx_msgs)
        reply = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
        await message.channel.send(reply or "I’m not sure.")

async def setup(bot: Red):
    await bot.add_cog(FusRohCog(bot))
