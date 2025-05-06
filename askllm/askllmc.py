import subprocess
import importlib
import asyncio
import re
import uuid
import json
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
from qdrant_client import QdrantClient, http
from rank_bm25 import BM25Okapi

# Dynamically install missing packages

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

# Ensure spaCy model
import spacy
try:
    spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')

# Download necessary NLTK corpora
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class LLMManager(commands.Cog):
    """
    English LLM-based support Cog:
      • Synonym expansion (WordNet)
      • Phrase detection (RAKE)
      • Hybrid retrieval: BM25 + vector
      • Recency & overlap boosts
      • Cache popular queries
      • Auto-reply on mention or channels
      • Knowledgebase CRUD admin commands
    """

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
        self.rake = Rake()
        import nltk.data
        self.rake.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle').tokenize
        self.nlp = spacy.load('en_core_web_sm')
        self.cache = TTLCache(maxsize=100, ttl=3600)
        self._TAG_STOPWORDS = set([
            "the", "a", "an", "to", "in", "on", "with", "for",
            "of", "and", "or", "i", "you", "we", "they", "it",
            "my", "your", "our",
        ])
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
            size = info.config.params.vectors.size  # type: ignore
            if size != self.vec_dim or force:
                raise ValueError
        except Exception:
            self.q_client.recreate_collection(
                collection_name="fus_wiki",
                vectors_config={
                    "size": self.vec_dim,
                    "distance": "Cosine",
                    "hnsw_config": {"m": 16, "ef_construct": 200},
                },
                optimizers_config={
                    "default_segment_number": 4,
                    "indexing_threshold": 256,
                },
                payload_indexing_config={
                    "enable": True,
                    "field_schema": {
                        "tag": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "content": {"type": "text"},
                        "created_at": {"type": "keyword"},
                    },
                },
                compression_config={
                    "type": "ProductQuantization",
                    "params": {"segments": 8, "subvector_size": 2},
                },
                wal_config={"wal_capacity_mb": 1024},
            )

    def _embed(self, text: str) -> List[float]:
        return self.embedder.encode(text).tolist()

    def _expand_synonyms(self, terms: List[str]) -> List[str]:
        syns = set(terms)
        for term in terms:
            for syn in wn.synsets(term):
                for lem in syn.lemmas():
                    syns.add(lem.name().lower())
        return list(syns)

    def _extract_phrases(self, text: str) -> List[str]:
        self.rake.extract_keywords_from_text(text)
        return [p.lower().replace(" ", "_") for p in self.rake.get_ranked_phrases()[:5]]

    def _guess_tags(self, text: str) -> List[str]:
        words = re.findall(r"\w+", text.lower())
        seen, uniq = set(), []
        for w in words:
            if w not in self._TAG_STOPWORDS and len(w) > 2 and w not in seen:
                uniq.append(w)
                seen.add(w)
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
        hits = self.q_client.search(
            "fus_wiki", query_vector=q_vec, limit=limit,
            with_payload=True, query_filter=filt
        )
        if not hits and filt:
            hits = self.q_client.search(
                "fus_wiki", query_vector=q_vec, limit=limit,
                with_payload=True
            )
        recency_months = await self.config.recency_months()
        results = []
        for idx, h in enumerate(hits):
            pl = h.payload or {}
            created = pl.get('created_at')
            rec = 0.0
            if created:
                age = (datetime.utcnow() - datetime.fromisoformat(created)).days / 30
                rec = max(0, 1 - age / recency_months)
            ov = self._token_overlap(question, pl.get('content', ''))
            bm25 = bm25_scores.get(idx, 0)
            score = (h.score or 0) + 0.1 * (pl.get('source') == 'manual')
            score += 0.5 * ov + 0.2 * rec + 0.1 * bm25
            results.append((h, score))
        results.sort(key=lambda x: x[1], reverse=True)
        self._last_ranked_hits = [h for h, _ in results[:5]]
        return [h for h, _ in results[:8]]

    async def _ask_llm(self, facts: List[str], question: str) -> str:
        # Build prompt
        prompt = (
            "Use only these facts to answer. If none apply, say 'I don't know'.\nFacts:\n"
            + "\n\n".join(facts)
            + f"\n\nQuestion: {question}\nAnswer:"
        )
        api, model = await self.config.api_url(), await self.config.model()
        # Disable streaming to avoid chunk floods
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        r = requests.post(
            f"{api.rstrip('/')}" + "/api/chat",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        try:
            data = r.json()
            # Extract assistant content
            return data.get('message', {}).get('content', '').strip()
        except (ValueError, json.JSONDecodeError):
            # Fallback to raw text
            return r.text.strip()

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return
        # Determine if should reply
        autolist = await self.config.auto_channels()
        if self.bot.user.mentioned_in(message):
            q = message.clean_content.replace(f"@{self.bot.user.display_name}", "").strip()
        elif message.channel.id in autolist:
            q = message.content.strip()
        else:
            return
        if not q:
            return

        # Gather context and retrieve hits
        ctx_txt = await self._get_recent_context(message.channel, before=message)
        async with message.channel.typing():
            hits = await self._retrieve(q)
            facts = [h.payload.get('content', '') for h in hits]
            ans = await self._ask_llm(facts, q)

        # Truncate or chunk answer to fit Discord limits
        if len(ans) > 1900:
            for chunk in (ans[i:i+1900] for i in range(0, len(ans), 1900)):
                await message.channel.send(chunk)
        else:
            await message.channel.send(ans)

        # Send any images from top hits
        thr = await self.config.vector_threshold()
        for idx, h in enumerate(self._last_ranked_hits):
            if h.payload.get('source') != 'manual':
                continue
            score = getattr(h, 'score', 0)
            if score >= thr:
                for url in h.payload.get('images', []):
                    await message.channel.send(embed=Embed().set_image(url=url))

    @commands.command(name="askllm")
    async def askllm(self, ctx: commands.Context, *, question: str):
        ctx_txt = await self._get_recent_context(ctx.channel, before=ctx.message)
        async with ctx.typing():
            hits = await self._retrieve(question)
            facts = [h.payload.get('content', '') for h in hits]
            ans = await self._ask_llm(facts, question)
        for chunk in (ans[i:i+1900] for i in range(0, len(ans), 1900)):
            await ctx.send(chunk)
        thr = await self.config.vector_threshold()
        for h in self._last_ranked_hits:
            if h.payload.get('source') != 'manual':
                continue
            if getattr(h, 'score', 0) >= thr:
                for url in h.payload.get('images', []):
                    await ctx.send(embed=Embed().set_image(url=url))

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def initcollection(self, ctx: commands.Context):
        await self.ensure_qdrant()
        await ctx.send("Recreating collection…")
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: self._ensure_collection(force=True)
        )
        await ctx.send("✅ Collection recreated!")

    @commands.command(name="llmknow")
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx: commands.Context, tag: str, *, content: str):
        pid = await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.upsert_entry(tag.lower(), content, 'manual')
        )
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
            imgs = pl.get('images', [])
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
        images = pl.get('images', [])
        if url in images:
            return await ctx.send("⚠️ Already added.")
        images.append(url)
        self.q_client.set_payload("fus_wiki", {'images': images}, [doc_id])
        await ctx.send("✅ Image added.")

async def setup(bot: commands.Bot):
    await bot.add_cog(LLMManager(bot))
