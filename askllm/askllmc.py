import subprocess
import importlib

def _ensure_pkg(mod: str, pip_name: str | None = None):
    try:
        importlib.import_module(mod)
    except ModuleNotFoundError:
        subprocess.check_call([
            "python", "-m", "pip", "install", pip_name or mod
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        importlib.import_module(mod)

for pkg in ["nltk", "spacy", "rake_nltk", "cachetools", "sentence_transformers", "qdrant_client", "rank_bm25"]:
    _ensure_pkg(pkg)

import spacy
try:
    spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')

import asyncio, re, uuid, json
from datetime import datetime
from typing import List, Optional, Dict
import discord
from discord import Embed
import requests
import nltk
from nltk.corpus import wordnet as wn
from rake_nltk import Rake
from cachetools import TTLCache
from redbot.core import commands, Config
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

# Download necessary NLTK corpora
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class LLMManager(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9876543210123)
        self.config.register_global(
            model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333",
            vector_threshold=0.3,
            recency_months=6,
            auto_channels=[],
            threshold=0.6,
        )
        self.embedder = SentenceTransformer("all-mpnet-base-v2")
        self.vec_dim = self.embedder.get_sentence_embedding_dimension()
        self.q_client: Optional[QdrantClient] = None
        self.bm25: Optional[BM25Okapi] = None
        self._bm25_texts: List[str] = []
        self.rake = Rake()
        import nltk.data
        self.rake.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle').tokenize
        self.nlp = spacy.load('en_core_web_sm')
        self.cache = TTLCache(maxsize=100, ttl=3600)
        self._TAG_STOPWORDS = set(["the","a","an","to","in","on","with","for","of","and","or","i","you","we","they","it","my","your","our"])
        self._last_ranked_hits: List = []
        self._last_manual_id: Optional[int] = None

    async def ensure_qdrant(self):
        if not self.q_client:
            url = await self.config.qdrant_url()
            self.q_client = QdrantClient(url=url)
            self._ensure_collection()

    def _ensure_collection(self, force: bool = False):
        try:
            info = self.q_client.get_collection("fus_wiki")
            size = info.config.params.vectors.size
            if size != self.vec_dim or force:
                raise ValueError
        except Exception:
            self.q_client.recreate_collection(
                collection_name="fus_wiki",
                vectors_config={"size": self.vec_dim, "distance": "Cosine", "hnsw_config": {"m": 16, "ef_construct": 200}},
                optimizers_config={"default_segment_number": 4, "indexing_threshold": 256},
                payload_indexing_config={"enable": True, "field_schema": {"tag": {"type": "keyword"}, "source": {"type": "keyword"}, "content": {"type": "text"}, "created_at": {"type": "keyword"}}},
                compression_config={"type": "ProductQuantization", "params": {"segments": 8, "subvector_size": 2}},
                wal_config={"wal_capacity_mb": 1024},
            )

    def _embed(self, text: str) -> List[float]:
        return self.embedder.encode(text).tolist()

    def _expand_synonyms(self, terms: List[str]) -> List[str]:
        syns = set(terms)
        for term in terms:
            for syn in wn.synsets(term):
                for lem in syn.lemmas(): syns.add(lem.name().lower())
        return list(syns)

    def _extract_phrases(self, text: str) -> List[str]:
        self.rake.extract_keywords_from_text(text)
        return [p.lower().replace(" ", "_") for p in self.rake.get_ranked_phrases()[:5]]

    def _guess_tags(self, text: str) -> List[str]:
        words = re.findall(r"\w+", text.lower())
        seen, uniq = set(), []
        for w in words:
            if w not in self._TAG_STOPWORDS and len(w) > 2 and w not in seen:
                uniq.append(w); seen.add(w)
        return uniq[:5]

    def _dynamic_limit(self, question: str) -> int:
        l = len(question.split())
        return 10 if l < 5 else 40 if l < 20 else 100

    def _token_overlap(self, a: str, b: str) -> float:
        ta = set(re.findall(r"\w+", a.lower()))
        tb = set(re.findall(r"\w+", b.lower()))
        return len(ta & tb) / len(tb) if tb else 0.0

    async def _retrieve(self, question: str) -> List:
        toks = [tok.text.lower() for tok in self.nlp(question) if tok.is_alpha]
        tags = self._expand_synonyms(toks + self._extract_phrases(question))[:10]
        q_vec = self._embed(question)
        limit = self._dynamic_limit(question)
        bm25_scores = {i: s for i, s in enumerate(self.bm25.get_scores(question.split()))} if self.bm25 else {}
        await self.ensure_qdrant()
        filt = {"should": [{"key": "tag", "match": {"value": t}} for t in tags]} if tags else None
        hits = self.q_client.search("fus_wiki", query_vector=q_vec, limit=limit, with_payload=True, query_filter=filt)
        if not hits and filt:
            hits = self.q_client.search("fus_wiki", query_vector=q_vec, limit=limit, with_payload=True)
        recency_months = await self.config.recency_months()
        results = []
        from datetime import datetime as _dt
        for idx, h in enumerate(hits):
            pl = h.payload or {}
            created = pl.get('created_at')
            rec = 0.0
            if created:
                age = (_dt.utcnow() - _dt.fromisoformat(created)).days / 30
                rec = max(0, 1 - age / recency_months)
            ov = self._token_overlap(question, pl.get('content', ''))
            bm25 = bm25_scores.get(idx, 0)
            score = (h.score or 0) + 0.1 * (pl.get('source') == 'manual') + 0.5 * ov + 0.2 * rec + 0.1 * bm25
            results.append((h, score))
        results.sort(key=lambda x: x[1], reverse=True)
        self._last_ranked_hits = [h for h, _ in results[:5]]
        return [h for h, _ in results[:8]]

    async def _ask_llm(self, facts: List[str], question: str) -> str:
        prompt = (
            "Use only these facts to answer. If none apply, say 'I don't know'.\nFacts:\n" +
            "\n\n".join(facts) +
            f"\n\nQuestion: {question}\nAnswer:"
        )
        api, model = await self.config.api_url(), await self.config.model()
        r = requests.post(f"{api.rstrip('/')}/api/chat", json={"model": model, "messages": [{"role": "user", "content": prompt}]}, timeout=120)
        r.raise_for_status()
        try:
            return r.json().get('message', {}).get('content', '').strip()
        except ValueError:
            return r.text.strip()

    async def _get_recent_context(self, channel: discord.TextChannel, before: discord.Message, n: int = 10) -> str:
        lines: List[str] = []
        async for m in channel.history(limit=n * 5, before=before):
            if not m.content.strip(): continue
            role = "Bot" if m.author.bot else "User"
            lines.append(f"{role}: {m.content.strip()}")
            if len(lines) >= n: break
        return "\n".join(lines[::-1])

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild: return
        autolist = await self.config.auto_channels()
        if self.bot.user.mentioned_in(message):
            q = message.clean_content.replace(f"@{self.bot.user.display_name}", "").strip()
        elif message.channel.id in autolist:
            q = message.content.strip()
        else:
            return
        if not q: return
        ctx_txt = await self._get_recent_context(message.channel, before=message)
        async with message.channel.typing():
            hits = await self._retrieve(q)
            facts = [h.payload.get('content', '') for h in hits]
            ans = await self._ask_llm(facts, q)
        await message.channel.send(ans)
        thr = await self.config.vector_threshold()
        for h in self._last_ranked_hits:
            if h.payload.get('source') != 'manual': continue
            if getattr(h, 'score', 0) >= thr:
                for url in h.payload.get('images', []):
                    await message.channel.send(embed=Embed().set_image(url=url))

    @commands.command(name="askllm")
    async def askllm(self, ctx: commands.Context, *, question: str):
        ctx_txt = await self._get_recent_context(ctx.channel, before=ctx.message)
        async with ctx.typing():
            hits = await self._retrieve(question)
            facts = [h.payload.get('content', '') for h in hits]
            ans = await self._ask_llm(facts, question)
        await ctx.send(ans)
        thr = await self.config.vector_threshold()
        for idx, h in enumerate(self._last_ranked_hits):
            if h.payload.get('source') != 'manual': continue
            if getattr(h, 'score', 0) >= thr:
                for url in h.payload.get('images', []):
                    await ctx.send(embed=Embed().set_image(url=url))

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def initcollection(self, ctx: commands.Context):
        await self.ensure_qdrant()
        await ctx.send("Recreating collection…")
        await asyncio.get_running_loop().run_in_executor(None, lambda: self._ensure_collection(force=True))
        await ctx.send("✅ Collection recreated!")

    @commands.command(name="llmknow")
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx: commands.Context, tag: str, *, content: str):
        pid = await asyncio.get_running_loop().run_in_executor(None, lambda: self.upsert_entry(tag.lower(), content, 'manual'))
        await ctx.send(f"Added entry under '{tag}' (ID {pid})")
        self._last_manual_id = pid

    @commands.command(name="llmknowshow")
    async def llmknowshow(self, ctx: commands.Context):
        """List all knowledge entries with images"""
        await self.ensure_qdrant()
        pts, _ = self.q_client.scroll("fus_wiki", with_payload=True, limit=1000)
        if not pts:
            return await ctx.send("No entries.")
        lines: List[str] = []
        for p in pts:
            pl = p.payload or {}
            entry = f"[{p.id}] ({pl.get('tag')}) {pl.get('content')}"
            imgs = pl.get("images", [])
            if imgs:
                entry += "\n→ Images:\n" + "\n".join(f"- {u}" for u in imgs)
            lines.append(entry)
        text = "\n\n".join(lines)
        for chunk in (text[i:i+1900] for i in range(0, len(text), 1900)):
            await ctx.send(f"```{chunk}```")

    @commands.command(name="llmknowaddimg")
    @commands.has_permissions(administrator=True)
    async def llmknowaddimg(self, ctx: commands.Context, doc_id: int, url: str):
        """Add an image URL to a knowledge entry"""
        if not re.search(r'\.(?:png|jpe?g|gif|webp)(?:\?.*)?$', url, flags=re.IGNORECASE):
            return await ctx.send("⚠️ That URL doesn't look like an image.")
        await self.ensure_qdrant()
        pts = self.q_client.retrieve("fus_wiki", [doc_id], with_payload=True)
        if not pts:
            return await ctx.send(f"No entry {doc_id}.")
        pl = pts[0].payload or {}
        images = pl.get("images", [])
        if url in images:
            return await ctx.send("⚠️ Already added.")
        images.append(url)
        self.q_client.set_payload("fus_wiki", {"images": images}, [doc_id])
        await ctx.send("✅ Image added.")

    @commands.command(name="llmknowrmimg")
    @commands.has_permissions(administrator=True)
    async def llmknowrmimg(self, ctx: commands.Context, doc_id: int, url: str):
        await self.ensure_qdrant()
        pts = self.q_client.retrieve("fus_wiki", [doc_id], with_payload=True)
        if not pts:
            return await ctx.send(f"No entry {doc_id}.")
        pl = pts[0].payload or {}
        imgs = pl.get('images', [])
        if url not in imgs:
            return await ctx.send("⚠️ URL not found.")
        imgs.remove(url)
        self.q_client.set_payload("fus_wiki", {"images": imgs}, [doc_id])
        await ctx.send("✅ Image removed.")

    @commands.command(name="llmknowmvimg")
    @commands.has_permissions(administrator=True)
    async def llmknowmvimg(self, ctx: commands.Context, doc_id: int, from_pos: int, to_pos: int):
        await self.ensure_qdrant()
        pts = self.q_client.retrieve("fus_wiki", [doc_id], with_payload=True)
        if not pts:
            return await ctx.send(f"No entry {doc_id}.")
        imgs = pts[0].payload.get('images', [])
        if not 1 <= from_pos <= len(imgs):
            return await ctx.send("⚠️ 'from' position out of range.")
        url = imgs.pop(from_pos - 1)
        to_pos = max(1, min(to_pos, len(imgs) + 1))
        imgs.insert(to_pos - 1, url)
        self.q_client.set_payload("fus_wiki", {"images": imgs}, [doc_id])
        await ctx.send("✅ Image reordered.")

    @commands.command(name="llmknowclearimgs")
    @commands.has_permissions(administrator=True)
    async def llmknowclearimgs(self, ctx: commands.Context, doc_id: Optional[int] = None):
        await self.ensure_qdrant()
        if doc_id is not None:
            pts = self.q_client.retrieve("fus_wiki", [doc_id], with_payload=True)
            if not pts:
                return await ctx.send(f"No entry {doc_id}.")
            self.q_client.set_payload("fus_wiki", {"images": []}, [doc_id])
            return await ctx.send(f"Cleared images from {doc_id}.")
        pts, _ = self.q_client.scroll("fus_wiki", with_payload=True, limit=1000)
        cleared = 0
        for p in pts:
            if p.payload.get('images'):
                self.q_client.set_payload("fus_wiki", {"images": []}, [p.id])
                cleared += 1
        await ctx.send(f"Cleared images from {cleared} entries.")

    @commands.command(name="llmknowdelete")
    @commands.has_permissions(administrator=True)
    async def llmknowdelete(self, ctx: commands.Context, doc_id: int):
        await self.ensure_qdrant()
        self.q_client.delete("fus_wiki", [doc_id])
        await ctx.send(f"Deleted entry {doc_id}.")

    @commands.command(name="llmknowdeletetag")
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx: commands.Context, tag: str):
        await self.ensure_qdrant()
        pts, _ = self.q_client.scroll("fus_wiki", with_payload=False, limit=1000, scroll_filter={"must":[{"key":"tag","match":{"value": tag.lower()}}]})
        ids = [p.id for p in pts]
        if ids:
            self.q_client.delete("fus_wiki", ids)
        await ctx.send(f"Deleted {len(ids)} entries tagged '{tag}'.")

    @commands.command(name="llmknowdeletelast")
    @commands.has_permissions(administrator=True)
    async def llmknowdeletelast(self, ctx: commands.Context):
        if self._last_manual_id is None:
            return await ctx.send("No recent manual entry.")
        await self.ensure_qdrant()
        self.q_client.delete("fus_wiki", [self._last_manual_id])
        await ctx.send(f"Deleted last entry {self._last_manual_id}.")
        self._last_manual_id = None

    @commands.command(name="addautochannel")
    @commands.has_permissions(administrator=True)
    async def addautochannel(self, ctx: commands.Context, channel: discord.TextChannel):
        chans = await self.config.auto_channels()
        if channel.id in chans:
            return await ctx.send("Already enabled.")
        chans.append(channel.id)
        await self.config.auto_channels.set(chans)
        await ctx.send(f"Auto-reply enabled in {channel.mention}.")

    @commands.command(name="removeautochannel")
    @commands.has_permissions(administrator=True)
    async def removeautochannel(self, ctx: commands.Context, channel: discord.TextChannel):
        chans = await self.config.auto_channels()
        if channel.id not in chans:
            return await ctx.send("Not enabled.")
        chans.remove(channel.id)
        await self.config.auto_channels.set(chans)
        await ctx.send(f"Auto-reply disabled in {channel.mention}.")

    @commands.command(name="listautochannels")
    async def listautochannels(self, ctx: commands.Context):
        chans = await self.config.auto_channels()
        if not chans:
            return await ctx.send("No auto-reply channels set.")
        await ctx.send("Auto-reply active in: " + ", ".join(f"<#{c}>" for c in chans))

    @commands.command(name="setthreshold")
    @commands.has_permissions(administrator=True)
    async def setthreshold(self, ctx: commands.Context, value: float):
        if not 0.0 <= value <= 1.0:
            return await ctx.send("Threshold must be 0.0–1.0")
        await self.config.threshold.set(value)
        await ctx.send(f"Threshold set to {value:.0%}.")

    @commands.command(name="setvectorthreshold")
    @commands.has_permissions(administrator=True)
    async def setvectorthreshold(self, ctx: commands.Context, value: float):
        if not 0.0 <= value <= 1.0:
            return await ctx.send("Vector threshold must be 0.0–1.0")
        await self.config.vector_threshold.set(value)
        await ctx.send(f"Vector threshold set to {value:.0%}.")

async def setup(bot: commands.Bot):
    await bot.add_cog(LLMManager(bot))
