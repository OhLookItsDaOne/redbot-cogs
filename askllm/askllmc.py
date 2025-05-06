import subprocess
import importlib
# Dynamically install missing packages
def _ensure_pkg(mod: str, pip_name: str | None = None):
    try:
        importlib.import_module(mod)
    except ModuleNotFoundError:
        subprocess.check_call([
            "python", "-m", "pip", "install", pip_name or mod
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        importlib.import_module(mod)
# Dependencies
for pkg in ["nltk", "spacy", "rake_nltk", "cachetools", "sentence_transformers", "qdrant_client", "rank_bm25"]:
    _ensure_pkg(pkg)

# Ensure spaCy model
import spacy
try:
    spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')

import asyncio
import re
import uuid
import json
from datetime import datetime, timedelta
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

# Download NLTK corpora
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # for RAKE sentence tokenization

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
            api_url="http://localhost:11434",
            qdrant_url="http://localhost:6333",
            vector_threshold=0.3,
            recency_months=6,
            auto_channels=[],
            threshold=0.6,
        )

        # Embedding & index
        self.embedder = SentenceTransformer("all-mpnet-base-v2")
        self.vec_dim = self.embedder.get_sentence_embedding_dimension()
        self.q_client: Optional[QdrantClient] = None

        # BM25 local
        self.bm25: Optional[BM25Okapi] = None
        self._bm25_texts: List[str] = []

        # Phrase & synonyms
        self.rake = Rake()
        self.rake.sentence_tokenizer = nltk.sent_tokenize
        self.nlp = spacy.load('en_core_web_sm')

        # Cache
        self.cache = TTLCache(maxsize=100, ttl=3600)

        # Tag stopwords
        self._TAG_STOPWORDS = set(["the","a","an","to","in","on","with","for","of","and","or","i","you","we","they","it","my","your","our"])

        # State
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
        good = [w for w in words if w not in self._TAG_STOPWORDS and len(w)>2]
        seen=set(); uniq=[]
        for w in good:
            if w not in seen: uniq.append(w); seen.add(w)
        return uniq[:5]

    def _dynamic_limit(self, question: str) -> int:
        l = len(question.split())
        return 10 if l<5 else 40 if l<20 else 100

    def _token_overlap(self, a: str, b: str) -> float:
        ta=set(re.findall(r"\w+",a.lower())); tb=set(re.findall(r"\w+",b.lower()))
        return len(ta&tb)/len(tb) if tb else 0.0

    async def _retrieve(self, question: str) -> List[Dict]:
        toks=[tok.text.lower() for tok in self.nlp(question) if tok.is_alpha]
        tags=self._expand_synonyms(toks + self._extract_phrases(question))[:10]
        q_vec=self._embed(question)
        limit=self._dynamic_limit(question)

        bm25_scores={i:s for i,s in enumerate(self.bm25.get_scores(question.split()))} if self.bm25 else {}
        await self.ensure_qdrant()
        filt={"should":[{"key":"tag","match":{"value":t}} for t in tags]} if tags else None
        hits=self.q_client.search("fus_wiki", query_vector=q_vec, limit=limit, with_payload=True, query_filter=filt)
        if not hits and filt: hits=self.q_client.search("fus_wiki", query_vector=q_vec, limit=limit, with_payload=True)

        recency_months=await self.config.recency_months()
        results=[]
        for idx,h in enumerate(hits):
            pl=h.payload or {}
            age_days=(datetime.utcnow()-datetime.fromisoformat(pl.get('created_at',''))).days if pl.get('created_at') else None
            rec= max(0,1-(age_days/30)/recency_months) if age_days is not None else 0
            ov=self._token_overlap(question, pl.get('content',''))
            bm25=bm25_scores.get(idx,0)
            score=(h.score or 0)+0.1*(pl.get('source')=='manual')+0.5*ov+0.2*rec+0.1*bm25
            results.append((h,score))
        results.sort(key=lambda x:x[1], reverse=True)
        self._last_ranked_hits=[h for h,_ in results[:5]]
        return [h for h,_ in results[:8]]

    async def _ask_llm(self, facts:List[str], question:str) -> str:
        prompt="Use only these facts to answer. If none apply, say 'I don't know'.\nFacts:\n"+"\n\n".join(facts)+f"\n\nQuestion: {question}\nAnswer:"
        api, model = await self.config.api_url(), await self.config.model()
        r=requests.post(f"{api}/api/chat", json={"model":model,"messages":[{"role":"user","content":prompt}]}, timeout=120)
        r.raise_for_status()
        return r.json().get('message',{}).get('content','').strip()

    async def _get_recent_context(self, channel, before, n:int=10) -> str:
        lines=[]
        async for m in channel.history(limit=n*5, before=before):
            if not m.content.strip(): continue
            role='Bot' if m.author.bot else 'User'
            lines.append(f"{role}: {m.content.strip()}")
            if len(lines)>=n: break
        return "\n".join(lines[::-1])

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild: return
        autolist=await self.config.auto_channels()
        if self.bot.user.mentioned_in(message):
            q=message.clean_content.replace(f"@{self.bot.user.display_name}","").strip()
        elif message.channel.id in autolist:
            q=message.content.strip()
        else:
            return
        if not q: return
        ctx_txt=await self._get_recent_context(message.channel, before=message)
        async with message.channel.typing():
            hits=await self._retrieve(q)
            facts=[h.payload.get('content','') for h in hits]
            ans=await self._ask_llm(facts,q)
        await message.channel.send(ans)
        thr=await self.config.vector_threshold()
        for idx,h in enumerate(self._last_ranked_hits):
            if h.payload.get('source')!='manual': continue
            if h.score and h.score>=thr:
                for url in h.payload.get('images',[]): await message.channel.send(embed=Embed().set_image(url=url))

    @commands.Cog.listener()
    async def on_ready(self): print("LLMManager cog loaded.")

    # ----- Commands -----
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def initcollection(self, ctx):
        """Recreate Qdrant collection from scratch"""
        await self.ensure_qdrant()
        await ctx.send("Recreating collection…")
        await asyncio.get_running_loop().run_in_executor(None, lambda: self._ensure_collection(force=True))
        await ctx.send("✅ Collection recreated!")

    @commands.command(name="llmknow")
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx: commands.Context, tag: str, *, content: str):
        pid = await asyncio.get_running_loop().run_in_executor(None, lambda: self.upsert_entry(tag.lower(), content, 'manual'))
        """Manually add a knowledge entry"""
        pid=await asyncio.get_running_loop().run_in_executor(None, lambda: self.upsert_entry(tag.lower(), content, 'manual'))
        await ctx.send(f"Added entry under '{tag}' (ID {pid})")
        self._last_manual_id=pid

    @commands.command(name="llmknowshow")
    async def llmknowshow(self, ctx):
        """List all knowledge entries with images"""
        await self.ensure_qdrant()
        pts,_=self.q_client.scroll('fus_wiki', with_payload=True, limit=1000)
        if not pts: return await ctx.send("No entries.")
        lines=[]
        for p in pts:
            pl=p.payload or {}
            text=f"[{p.id}] ({pl.get('tag')}) {pl.get('content')}"
            imgs=pl.get('images',[])
            if imgs:
                text+="\n Images:\n"+"\n".join(f"- {u}" for u in imgs)
            lines.append(text)
        msg="\n\n".join(lines)
        for chunk in (msg[i:i+1900] for i in range(0,len(msg),1900)):
            await ctx.send(f"```{chunk}```")

    @commands.command(name="llmknowaddimg")
    @commands.has_permissions(administrator=True)
    async def llmknowaddimg(self, ctx, doc_id: int, url: str):
        """Add an image URL to a knowledge entry"""
        if not url.lower().endswith(('.png','.jpg','.jpeg','.gif','.webp')):
            return await ctx.send("⚠️ Not an image URL.")
        await self.ensure_qdrant()
        pts=self.q_client.retrieve('fus_wiki',[doc_id], with_payload=True)
        if not pts: return await ctx.send(f"No entry {doc_id}.")
        pl=pts[0].payload or {}
        imgs=pl.setdefault('images',[])
        if url in imgs: return await ctx.send("⚠️ Already added.")
        imgs.append(url)
        self.q_client.set_payload('fus_wiki',{'images':imgs},[doc_id])
        await ctx.send("✅ Image added.")

    @commands.command(name="llmknowrmimg")
    @commands.has_permissions(administrator=True)
    async def llmknowrmimg(self, ctx, doc_id: int, url: str):
        """Remove an image URL from an entry"""
        await self.ensure_qdrant()
        pts=self.q_client.retrieve('fus_wiki',[doc_id], with_payload=True)
        if not pts: return await ctx.send(f"No entry {doc_id}.")
        pl=pts[0].payload or {}
        imgs=pl.get('images',[])
        try: imgs.remove(url)
        except ValueError: return await ctx.send("⚠️ URL not found.")
        self.q_client.set_payload('fus_wiki',{'images':imgs},[doc_id])
        await ctx.send("✅ Image removed.")

    @commands.command(name="llmknowmvimg")
    @commands.has_permissions(administrator=True)
    async def llmknowmvimg(self, ctx, doc_id: int, from_idx: int, to_idx: int):
        """Reorder an image URL in an entry"""
        await self.ensure_qdrant()
        pts=self.q_client.retrieve('fus_wiki',[doc_id], with_payload=True)
        if not pts: return await ctx.send(f"No entry {doc_id}.")
        imgs=pts[0].payload.get('images',[])
        if not 1<=from_idx<=len(imgs): return await ctx.send("⚠️ 'from' out of range.")
        url=imgs.pop(from_idx-1)
        to_idx=max(1,min(to_idx,len(imgs)+1))
        imgs.insert(to_idx-1,url)
        self.q_client.set_payload('fus_wiki',{'images':imgs},[doc_id])
        await ctx.send("✅ Image reordered.")

    @commands.command(name="llmknowclearimgs")
    @commands.has_permissions(administrator=True)
    async def llmknowclearimgs(self, ctx, doc_id: Optional[int]=None):
        """Clear images from one or all entries"""
        await self.ensure_qdrant()
        if doc_id:
            pts=self.q_client.retrieve('fus_wiki',[doc_id], with_payload=True)
            if not pts: return await ctx.send(f"No entry {doc_id}.")
            self.q_client.set_payload('fus_wiki',{'images':[]},[doc_id])
            return await ctx.send(f"Cleared images from {doc_id}.")
        pts,_=self.q_client.scroll('fus_wiki', with_payload=True, limit=1000)
        cleared=0
        for p in pts:
            if p.payload.get('images'):
                self.q_client.set_payload('fus_wiki',{'images':[]},[p.id])
                cleared+=1
        await ctx.send(f"Cleared images from {cleared} entries.")

    @commands.command(name="llmknowdelete")
    @commands.has_permissions(administrator=True)
    async def llmknowdelete(self, ctx, doc_id: int):
        """Delete a knowledge entry"""
        await self.ensure_qdrant()
        self.q_client.delete('fus_wiki',[doc_id])
        await ctx.send(f"Deleted entry {doc_id}.")

    @commands.command(name="llmknowdeletetag")
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx, tag: str):
        """Delete all entries with a tag"""
        await self.ensure_qdrant()
        pts,_=self.q_client.scroll('fus_wiki', with_payload=False, limit=1000, scroll_filter={"must":[{"key":"tag","match":{"value":tag.lower()}}]})
        ids=[p.id for p in pts]
        if ids: self.q_client.delete('fus_wiki',ids)
        await ctx.send(f"Deleted {len(ids)} entries tagged '{tag}'.")

    @commands.command(name="llmknowdeletelast")
    @commands.has_permissions(administrator=True)
    async def llmknowdeletelast(self, ctx):
        """Delete the last manually added entry"""
        if not self._last_manual_id: return await ctx.send("No recent manual entry.")
        await self.ensure_qdrant()
        self.q_client.delete('fus_wiki',[self._last_manual_id])
        await ctx.send(f"Deleted last entry {self._last_manual_id}.")
        self._last_manual_id=None

    @commands.command(name="addautochannel")
    @commands.has_permissions(administrator=True)
    async def addautochannel(self, ctx, channel: discord.TextChannel):
        """Enable auto-reply in a channel"""
        chans=await self.config.auto_channels()
        if channel.id in chans: return await ctx.send("Already enabled.")
        chans.append(channel.id)
        await self.config.auto_channels.set(chans)
        await ctx.send(f"Auto-reply enabled in {channel.mention}.")

    @commands.command(name="removeautochannel")
    @commands.has_permissions(administrator=True)
    async def removeautochannel(self, ctx, channel: discord.TextChannel):
        """Disable auto-reply in a channel"""
        chans=await self.config.auto_channels()
        if channel.id not in chans: return await ctx.send("Not enabled.")
        chans.remove(channel.id)
        await self.config.auto_channels.set(chans)
        await ctx.send(f"Auto-reply disabled in {channel.mention}.")

    @commands.command(name="listautochannels")
    async def listautochannels(self, ctx):
        """List all auto-reply channels"""
        chans=await self.config.auto_channels()
        if not chans: return await ctx.send("No auto-reply channels set.")
        await ctx.send("Auto-reply active in: " + ", ".join(f"<#{c}>" for c in chans))

    @commands.command(name="setthreshold")
    @commands.has_permissions(administrator=True)
    async def setthreshold(self, ctx, value: float):
        """Set token-overlap threshold (0.0–1.0)"""
        if not 0<=value<=1: return await ctx.send("Threshold must be 0.0–1.0")
        await self.config.threshold.set(value)
        await ctx.send(f"Threshold set to {value:.0%}.")

    @commands.command(name="setvectorthreshold")
    @commands.has_permissions(administrator=True)
    async def setvectorthreshold(self, ctx, value: float):
        """Set vector similarity threshold (0.0–1.0)"""
        if not 0<=value<=1: return await ctx.send("Vector threshold must be 0.0–1.0")
        await self.config.vector_threshold.set(value)
        await ctx.send(f"Vector threshold set to {value:.0%}.")

async def setup(bot: commands.Bot):
    await bot.add_cog(LLMManager(bot))

  
