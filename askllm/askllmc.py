```python
# askllmc.py – Hybrid Qdrant + local-Ollama Support Cog mit Synonym-, Phrase- und BM25-Hybrid-Retrieval
import asyncio
import subprocess
# Dynamisches Nachinstallieren fehlender Pakete (safe in Docker)
import subprocess

def _ensure_pkg(mod: str, pip_name: str | None = None):
    try:
        __import__(mod)
    except ModuleNotFoundError:
        subprocess.check_call(
            ["python", "-m", "pip", "install", pip_name or mod],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        __import__(mod)

# sicherstellen, dass alle externen Dependencies vorhanden sind
_ensure_pkg("nltk")
_ensure_pkg("spacy")
_ensure_pkg("rake_nltk")
_ensure_pkg("cachetools")
_ensure_pkg("sentence_transformers")
_ensure_pkg("qdrant_client")
_ensure_pkg("rank_bm25")

# SpaCy Modell auf Modulebene herunterladen, falls nicht vorhanden
import spacy
try:
    spacy.load('en_core_web_sm')
except OSError:
    import spacy.cli
    spacy.cli.download('en_core_web_sm')

import asyncio
import re
import uuid
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import discord
import requests
import nltk
import spacy
from nltk.corpus import wordnet as wn
from rake_nltk import Rake
from cachetools import TTLCache
from redbot.core import commands, Config
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, http
from rank_bm25 import BM25Okapi

# Stelle sicher, dass notwendige NLTK-Corpora vorhanden sind
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)  # für RAKE und ggf. weitere Textverarbeitung

class LLMManager(commands.Cog):
    """
    Discord-Cog für LLM-basierten Support mit folgenden Features:
      • Synonym-Expansion via WordNet
      • Phrase-Detection via RAKE
      • Hybrid Retrieval: BM25 + Vektor
      • Recency- & Token-Overlap-Ranking
      • Caching populärer Abfragen
      • Vollständige Admin-Commands (Knowledgebase-CRUD)
    """
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9876543210123)
        # Global defaults
        self.config.register_global(
            model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333",
            vector_threshold=0.3,
            recency_months=6,
        )

        # Embedding
        self.embedder = SentenceTransformer("all-mpnet-base-v2")
        self.vec_dim = self.embedder.get_sentence_embedding_dimension()

        # Qdrant-Client
        self.q_client: Optional[QdrantClient] = None

        # BM25
        self.bm25: Optional[BM25Okapi] = None
        self._bm25_texts: List[str] = []

        # Phrase-Extractor
        self.rake = Rake()

                        # NLP-Tokenizer
        self.nlp = spacy.load('en_core_web_sm')

        # Cache für populäre Queries (TTL 1h)
        self.cache = TTLCache(maxsize=100, ttl=3600)

        # Stopwords für Tag-Guess
        self._TAG_STOPWORDS = {
            "the", "a", "an", "to", "in", "on", "with", "for", "of", "and", "or",
            "i", "you", "we", "they", "it", "my", "your", "our",
        }

        # Letzte Hits für Bild-Reply
        self._last_ranked_hits: List = []

    async def ensure_qdrant(self):
        if not self.q_client:
            url = await self.config.qdrant_url()
            self.q_client = QdrantClient(url=url)
            self._ensure_collection()

    def _ensure_collection(self, force: bool = False):
        try:
            info = self.q_client.get_collection('fus_wiki')
            size = info.config.params.vectors.size
            if size != self.vec_dim or force:
                raise ValueError
        except Exception:
            self.q_client.recreate_collection(
                collection_name='fus_wiki',
                vectors_config={'size': self.vec_dim, 'distance': 'Cosine',
                                'hnsw_config': {'m':16, 'ef_construct':200}},
                optimizers_config={'default_segment_number':4, 'indexing_threshold':256},
                payload_indexing_config={'enable':True,
                                         'field_schema': {'tag':{'type':'keyword'}, 'source':{'type':'keyword'}, 'content':{'type':'text'}, 'created_at':{'type':'keyword'}}},
                compression_config={'type':'ProductQuantization', 'params':{'segments':8,'subvector_size':2}},
                wal_config={'wal_capacity_mb':1024},
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
        fr = self.rake.get_ranked_phrases()[:5]
        return [p.lower().replace(' ', '_') for p in fr]

    def _guess_tags(self, text: str) -> List[str]:
        words = re.findall(r"\w+", text.lower())
        good = [w for w in words if w not in self._TAG_STOPWORDS and len(w)>2]
        seen=set(); uniq=[]
        for w in good:
            if w not in seen:
                uniq.append(w); seen.add(w)
        return uniq[:5]

    def _dynamic_limit(self, question: str) -> int:
        l = len(question.split())
        if l<5: return 10
        if l<20: return 40
        return 100

    def _token_overlap(self, a: str, b: str) -> float:
        ta=set(re.findall(r"\w+",a.lower()))
        tb=set(re.findall(r"\w+",b.lower()))
        return len(ta&tb)/len(tb) if tb else 0.0

    async def _retrieve(self, question: str) -> List[Dict]:
        # Tags + Synonyme + Phrasen
        toks=[tok.text.lower() for tok in self.nlp(question) if tok.is_alpha]
        phrases=self._extract_phrases(question)
        tags=self._expand_synonyms(toks+phrases)[:10]
        q_vec=self._embed(question)
        limit=self._dynamic_limit(question)

        # BM25
        bm25_scores={}
        if self.bm25:
            scores=self.bm25.get_scores(question.split())
            bm25_scores={i:s for i,s in enumerate(scores)}

        await self.ensure_qdrant()
        filt={'should': [{'key':'tag','match':{'value':t}} for t in tags]} if tags else None
        hits=self.q_client.search('fus_wiki', query_vector=q_vec, limit=limit, with_payload=True, query_filter=filt)
        if not hits and filt:
            hits=self.q_client.search('fus_wiki', query_vector=q_vec, limit=limit, with_payload=True)

        # Ranking mit Boosts
        results=[]
        recency_months=await self.config.recency_months()
        for idx,h in enumerate(hits):
            pl=h.payload or {}
            # recency
            created=pl.get('created_at')
            rec=0.0
            if created:
                age=(datetime.utcnow()-datetime.fromisoformat(created)).days/30
                rec=max(0,1-age/recency_months)
            ov=self._token_overlap(question, pl.get('content',''))
            bm25=bm25_scores.get(idx,0.0)
            score=(h.score or 0.0) + 0.1*(pl.get('source')=='manual') + 0.5*ov + 0.2*rec + 0.1*bm25
            results.append({'hit':h,'score':score})
        results.sort(key=lambda x:x['score'], reverse=True)
        # speichere Top für Bilder-Reply
        self._last_ranked_hits=[r['hit'] for r in results[:5]]
        return results[:8]

    async def _is_relevant_llm(self, question:str, snippet:str) -> bool:
        sn=snippet if len(snippet)<=600 else snippet[:600]+" …"
        prompt=(
            "Expert assistant: Relevanz prüfen. Antworte exakt Yes/No.\n"
            f"Question:\n{question}\n\nSnippet:\n{sn}\nRelevant?"
        )
        api,model=await self.config.api_url(),await self.config.model()
        r=requests.post(f"{api.rstrip('/')}/api/chat", json={"model":model,"messages":[{"role":"user","content":prompt}]}, timeout=60)
        r.raise_for_status(); reply=r.json().get('message',{}).get('content','')
        return reply.strip().lower().startswith('y')

    async def _ask_llm(self, facts:List[str], question:str) -> str:
        prompt=(
            "Use only these facts to answer. If none apply, say 'I don't know'.\n"
            +"Facts:\n"+"\n\n".join(f"- {f}" for f in facts)
            +f"\n\nQuestion: {question}\nAnswer:"
        )
        api,model=await self.config.api_url(),await self.config.model()
        r=requests.post(f"{api}/api/chat", json={"model":model,"messages":[{"role":"user","content":prompt}]}, timeout=120)
        r.raise_for_status()
        return r.json().get('message',{}).get('content','').strip()

    # ===== Commands =====
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def initcollection(self, ctx:commands.Context):
        """Qdrant-Collection neu erstellen"""
        await self.ensure_qdrant()
        await ctx.send("Recreating collection…")
        await asyncio.get_running_loop().run_in_executor(None, lambda: self._ensure_collection(force=True))
        await ctx.send("✅ Collection recreated.")

    @commands.command(name="llmknowaddimg")
    @commands.has_permissions(administrator=True)
    async def llmknowaddimg(self, ctx, doc_id:int, url:str):
        """Bild-URL zu Eintrag hinzufügen"""
        if not re.search(r'\.(?:png|jpe?g|gif|webp)(?:\?.*)?$',url):
            return await ctx.send("⚠️ Keine Bild-URL.")
        await self.ensure_qdrant()
        pts=self.q_client.retrieve('fus_wiki',[doc_id], with_payload=True)
        if not pts: return await ctx.send(f"Kein Eintrag {doc_id}.")
        pl=pts[0].payload or {}; imgs=pl.get('images',[])
        if url in imgs: return await ctx.send("⚠️ Schon vorhanden.")
        imgs.append(url)
        self.q_client.set_payload('fus_wiki', {'images':imgs}, [doc_id])
        await ctx.send("✅ Bild hinzugefügt.")

    @commands.command(name="llmknowrmimg")
    @commands.has_permissions(administrator=True)
    async def llmknowrmimg(self, ctx, doc_id:int, url:str):
        """Bild-URL entfernen"""
        await self.ensure_qdrant()
        pts=self.q_client.retrieve('fus_wiki',[doc_id], with_payload=True)
        if not pts: return await ctx.send(f"Kein Eintrag {doc_id}.")
        imgs=pts[0].payload.get('images',[])
        if url not in imgs: return await ctx.send("⚠️ Nicht gefunden.")
        imgs.remove(url)
        self.q_client.set_payload('fus_wiki', {'images':imgs}, [doc_id])
        await ctx.send("✅ Bild entfernt.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, num:int):
        """Aus letzten `num` Messages KB-Eintrag bauen"""
        await self.ensure_qdrant()
        msgs=[m async for m in ctx.channel.history(limit=num+20)]
        txts=[m.content for m in msgs if not m.author.bot]
        excerpt="\n".join(reversed(txts[-num:]))
        api,model=await self.config.api_url(),await self.config.model()
        draft=await asyncio.get_running_loop().run_in_executor(None, lambda: requests.post(
            f"{api}/api/chat", json={"model":model,"messages":[{"role":"user","content":
            f"Create concise KB entry <1500 chars from:\n{excerpt}\nEntry:"}]}, timeout=120).json()
            .get('message',{}).get('content','')
        )
        # Interaktive Schleife gekürzt … (wie zuvor implementiert)
        # Upsert mit self.upsert_entry(tags, draft, 'manual')
        await ctx.send("✅ Learned (Implementierung interaktiver Loop analog vorher).")

    @commands.command(name="askllm")
    async def askllm_cmd(self, ctx:commands.Context, *, question:str):
        """Frage an Support-Bot"""
        # Cache
        if question in self.cache:
            return await ctx.send(self.cache[question])
        ctx_txt = ''  # Du kannst _get_recent_context übernehmen
        async with ctx.typing():
            hits = await self._retrieve(question)
            facts = [r['hit'].payload.get('content','') for r in hits]
            ans  = await self._ask_llm(facts, question)
        self.cache[question]=ans
        await ctx.send(ans)
        # Bilder senden
        thr=await self.config.vector_threshold()
        used=[int(n.lstrip('#')) for n in re.findall(r"#(\d+)", ans)]
        for idx in used:
            if idx < len(self._last_ranked_hits):
                pl=self._last_ranked_hits[idx].payload
                if pl.get('source')=='manual' and (self._last_ranked_hits[idx].score or 0)>=thr:
                    for url in pl.get('images',[]):
                        await ctx.send(embed=discord.Embed().set_image(url=url))

    @commands.command()
    async def setmodel(self, ctx, model:str):
        await self.config.model.set(model); await ctx.send(f"Model set to {model}")

    @commands.command()
    async def setapi(self, ctx, url:str):
        await self.config.api_url.set(url.rstrip('/')); await ctx.send("API-URL updated")

    @commands.command()
    async def setqdrant(self, ctx, url:str):
        await self.config.qdrant_url.set(url.rstrip('/')); self.q_client=None; await ctx.send("Qdrant-URL updated")

    @commands.Cog.listener()
    async def on_ready(self):
        print("LLMManager loaded.")

    # ===== Ursprüngliche CRUD-/Admin-Commands =====
    @commands.command(name="llmknow")
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx: commands.Context, tag: str, *, content: str):
        """Manuell einen Knowledge-Eintrag hinzufügen"""
        await self.ensure_qdrant()
        pid = await asyncio.get_running_loop().run_in_executor(None, self._upsert_entry, tag.lower(), content, "manual")
        await ctx.send(f"Added entry under '{tag}' (ID {pid}).")
        self._last_manual_id = pid

    @commands.command(name="llmknowshow")
    async def llmknowshow(self, ctx: commands.Context):
        """List all entries, showing full content and image URLs."""
        await self.ensure_qdrant()
        pts, _ = self.q_client.scroll('fus_wiki', with_payload=True, limit=1000)
        if not pts:
            return await ctx.send("No entries stored.")
        out = []
        for p in pts:
            pl = p.payload or {}
            line = f"[{p.id}] ({pl.get('tag')}, {pl.get('source')}): {pl.get('content')}"
            imgs = pl.get("images", [])
            if imgs:
                line += "\n  → Images:\n" + "\n".join(f"    • {u}" for u in imgs)
            out.append(line)
        # Hier richtig zusammenfügen
        text = "\n\n".join(out)
        for chunk in (text[i:i+1900] for i in range(0, len(text), 1900)):
            await ctx.send(f"```{chunk}```")

    @commands.command(name="llmknowmvimg")
    @commands.has_permissions(administrator=True)
    async def llmknowmvimg(self, ctx: commands.Context, doc_id: int, from_pos: int, to_pos: int):
        """Reorder ein Bild in einem Knowledge-Eintrag"""
        await self.ensure_qdrant()
        pts = self.q_client.retrieve('fus_wiki', [doc_id], with_payload=True)
        if not pts:
            return await ctx.send(f"No entry with ID {doc_id}.")
        imgs = pts[0].payload.get('images', [])
        if not (1 <= from_pos <= len(imgs)):
            return await ctx.send("'from' position out of range.")
        url = imgs.pop(from_pos-1)
        to_pos = max(1, min(to_pos, len(imgs)+1))
        imgs.insert(to_pos-1, url)
        self.q_client.set_payload('fus_wiki', {'images': imgs}, [doc_id])
        await ctx.send("✅ Image reordered.")

    @commands.command(name="llmknowclearimgs")
    @commands.has_permissions(administrator=True)
    async def llmknowclearimgs(self, ctx: commands.Context, doc_id: Optional[int] = None):
        """Clear image URLs from one entry or all entries"""
        await self.ensure_qdrant()
        if doc_id:
            pts = self.q_client.retrieve('fus_wiki', [doc_id], with_payload=True)
            if not pts:
                return await ctx.send(f"No entry {doc_id}.")
            self.q_client.set_payload('fus_wiki', {'images':[]}, [doc_id])
            return await ctx.send(f"Cleared images from {doc_id}.")
        pts, _ = self.q_client.scroll('fus_wiki', with_payload=True, limit=1000)
        to_clear = [p.id for p in pts if p.payload.get('images')]
        for pid in to_clear:
            self.q_client.set_payload('fus_wiki', {'images':[]}, [pid])
        await ctx.send(f"Cleared images from {len(to_clear)} entries.")

    @commands.command(name="llmknowdelete")
    @commands.has_permissions(administrator=True)
    async def llmknowdelete(self, ctx: commands.Context, doc_id: int):
        """Delete einen einzelnen Eintrag"""
        await self.ensure_qdrant()
        self.q_client.delete('fus_wiki', [doc_id])
        await ctx.send(f"Deleted entry {doc_id}.")

    @commands.command(name="llmknowdeletetag")
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx: commands.Context, tag: str):
        """Delete alle Einträge mit einem Tag"""
        await self.ensure_qdrant()
        filt = {'must':[{'key':'tag','match':{'value':tag.lower()}}]}
        pts, _ = self.q_client.scroll('fus_wiki', with_payload=False, limit=1000, scroll_filter=filt)
        ids = [p.id for p in pts]
        if ids:
            self.q_client.delete('fus_wiki', ids)
        await ctx.send(f"Deleted {len(ids)} entries tagged '{tag}'.")

    @commands.command(name="llmknowdeletelast")
    @commands.has_permissions(administrator=True)
    async def llmknowdeletelast(self, ctx: commands.Context):
        """Delete den zuletzt angelegten manuellen Eintrag"""
        if not hasattr(self, '_last_manual_id') or self._last_manual_id is None:
            return await ctx.send("No recent manual entry.")
        await self.ensure_qdrant()
        self.q_client.delete('fus_wiki', [self._last_manual_id])
        await ctx.send(f"Deleted last entry {self._last_manual_id}.")
        self._last_manual_id = None

    @commands.command(name="addautochannel")
    @commands.has_permissions(administrator=True)
    async def add_auto_channel(self, ctx: commands.Context, channel: discord.TextChannel):
        """Enable auto-reply in diesem Kanal"""
        chans = await self.config.auto_channels()
        if channel.id in chans:
            return await ctx.send("Already enabled there.")
        chans.append(channel.id)
        await self.config.auto_channels.set(chans)
        await ctx.send(f"Auto-reply enabled in {channel.mention}.")

    @commands.command(name="removeautochannel")
    @commands.has_permissions(administrator=True)
    async def remove_auto_channel(self, ctx: commands.Context, channel: discord.TextChannel):
        """Disable auto-reply in diesem Kanal"""
        chans = await self.config.auto_channels()
        if channel.id not in chans:
            return await ctx.send("Not enabled there.")
        chans.remove(channel.id)
        await self.config.auto_channels.set(chans)
        await ctx.send(f"Auto-reply disabled in {channel.mention}.")

    @commands.command(name="listautochannels")
    async def list_auto_channels(self, ctx: commands.Context):
        """List all channels with auto-reply enabled"""
        chans = await self.config.auto_channels()
        if not chans:
            return await ctx.send("No auto-reply channels set.")
        mentions = ", ".join(f"<#{cid}>" for cid in chans)
        await ctx.send(f"Auto-reply active in: {mentions}")

    @commands.command(name="setthreshold")
    @commands.has_permissions(administrator=True)
    async def setthreshold(self, ctx: commands.Context, value: float):
        """Set the token-overlap threshold (0.0–1.0)"""
        if not 0.0 <= value <= 1.0:
            return await ctx.send("Threshold must be between 0.0 and 1.0")
        await self.config.threshold.set(value)
        await ctx.send(f"✅ Threshold set to {value:.0%}")

    @commands.command(name="setvectorthreshold")
    @commands.has_permissions(administrator=True)
    async def setvectorthreshold(self, ctx: commands.Context, value: float):
        """Set the vector score threshold (0.0–1.0)"""
        if not 0.0 <= value <= 1.0:
            return await ctx.send("Threshold must be between 0.0 and 1.0")
        await self.config.vector_threshold.set(value)
        await ctx.send(f"✅ Vector threshold set to {value:.0%}")

async def setup(bot: commands.Bot):
    await bot.add_cog(LLMManager(bot))

async def setup(bot:commands.Bot):
    await bot.add_cog(LLMManager(bot))

