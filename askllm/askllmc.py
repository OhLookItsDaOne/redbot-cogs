import discord
import subprocess
import re
import requests
import time
import asyncio
import glob
import os
import shutil
import uuid

from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

class LLMManager(commands.Cog):
    """Cog to interact with the LLM using a Qdrant-based knowledge storage.

    Commands:
      - llmknow: add manual entry
      - llmknowshow: display entries
      - llmknowdelete: delete by ID
      - llmknowdeletetag: delete by tag
      - llmknowedit: edit entry
      - importwiki: import/update GitHub wiki
      - askllm: query LLM with Qdrant context
      - setmodel, setapi, setqdrant, initcollection
    """
    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9999999999)
        self.config.register_global(
            model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333"
        )
        self.collection_name = "fus_wiki"
        self.q_client = None
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    async def ensure_qdrant_client(self):
        if not self.q_client:
            url = await self.config.get_raw("qdrant_url", default="http://192.168.10.5:6333")
            self.q_client = QdrantClient(url=url)

    def upsert_knowledge(self, tag, content, source="manual"):
        try:
            self.q_client.get_collection(collection_name=self.collection_name)
        except:
            self.q_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={"size": 384, "distance": "Cosine"}
            )
        vec = self.embedding_model.encode(content).tolist()
        payload = {"tag": tag, "content": content, "source": source}
        doc_id = uuid.uuid4().int & ((1 << 64) - 1)
        self.q_client.upsert(
            collection_name=self.collection_name,
            points=[{"id": doc_id, "vector": vec, "payload": payload}]
        )

    @commands.command()
    async def llmknow(self, ctx, tag: str, *, info: str):
        await self.ensure_qdrant_client()
        self.upsert_knowledge(tag.lower(), info, source="manual")
        await ctx.send(f"Added manual info under '{tag.lower()}'.")

    def _get_all_knowledge_sync(self):
        try:
            pts = self.q_client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=1000
            )
            return list(pts) if pts else []
        except:
            return []

    async def get_all_knowledge(self):
        await self.ensure_qdrant_client()
        return await asyncio.get_running_loop().run_in_executor(None, self._get_all_knowledge_sync)

    @commands.command()
    async def llmknowshow(self, ctx):
        pts = await self.get_all_knowledge()
        if not pts:
            return await ctx.send("No knowledge entries stored.")
        seen, lines = set(), []
        for pt in pts:
            raw_id = getattr(pt, 'id', None) or (pt.get('id') if isinstance(pt, dict) else None)
            payload = getattr(pt, 'payload', None) or (pt.get('payload') if isinstance(pt, dict) else {})
            if raw_id is None: continue
            key = str(raw_id)
            if key in seen: continue
            seen.add(key)
            if not isinstance(payload, dict): payload = payload.dict() if hasattr(payload, 'dict') else dict(payload)
            tag = payload.get('tag','NoTag')
            src = payload.get('source','unknown')
            text = payload.get('content','')
            single = ' '.join(text.splitlines())
            lines.append(f"[{key}] ({tag},src={src}): {single}")
        header, footer, max_len = "```\n", "```", 2000
        chunks, cur = [], header
        for line in lines:
            if len(cur) + len(line) + 1 > max_len - len(footer):
                chunks.append(cur + footer)
                cur = header + line + "\n"
            else:
                cur += line + "\n"
        if cur != header: chunks.append(cur + footer)
        for c in chunks: await ctx.send(c)

    @commands.command()
    async def llmknowdelete(self, ctx, doc_id: int):
        await self.ensure_qdrant_client()
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.q_client.delete(
                self.collection_name, [doc_id], None
            )
        )
        await ctx.send(f"Deleted entry with ID {doc_id}.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx, tag: str):
        await self.ensure_qdrant_client()
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.q_client.delete(
                self.collection_name, [],
                filter={"must":[{"key":"tag","match":{"value":tag.lower()}}]}
            )
        )
        await ctx.send(f"Deleted all entries with tag '{tag.lower()}'.")

    @commands.command()
    async def llmknowedit(self, ctx, doc_id: int, new_tag: str, *, new_content: str):
        await self.ensure_qdrant_client()
        def upd():
            vec = self.embedding_model.encode(new_content).tolist()
            self.q_client.upsert(
                collection_name=self.collection_name,
                points=[{"id":doc_id,"vector":vec,"payload":{"tag":new_tag,"content":new_content}}]
            )
        await asyncio.get_running_loop().run_in_executor(None, upd)
        await ctx.send(f"Updated entry {doc_id}.")

    @commands.command()
    async def importwiki(self, ctx, wiki_url: str=None):
        await self.ensure_qdrant_client()
        wiki_url = wiki_url or "https://github.com/Kvitekvist/FUS.wiki.git"
        base = str(cog_data_path(self)); os.makedirs(base, exist_ok=True)
        clone = os.path.join(base, "wiki")
        # delete old wiki entries
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.q_client.delete(
                self.collection_name, None,
                filter={"must":[{"key":"source","match":{"value":"wiki"}}]}
            )
        )
        # clone or pull
        if os.path.isdir(os.path.join(clone, ".git")):
            subprocess.run(["git","-C",clone,"pull"], check=False)
            await ctx.send("Wiki updated.")
        else:
            shutil.rmtree(clone, ignore_errors=True)
            subprocess.run(["git","clone",wiki_url,clone], check=True)
            await ctx.send("Wiki cloned.")
        # process md
        import markdown
        from bs4 import BeautifulSoup
        def strip(h): return re.sub(r"<.*?>","",h)
        def tags(html):
            soup=BeautifulSoup(html,"html.parser")
            hs=soup.find_all(re.compile("^h[1-3]$"))
            return ", ".join(dict.fromkeys(h.get_text().strip() for h in hs)) or None
        files=glob.glob(os.path.join(clone,"*.md"))
        await ctx.send(f"Importing {len(files)} pages...")
        for f in files:
            txt=open(f,encoding="utf-8").read()
            html=markdown.markdown(txt)
            tg=tags(html) or os.path.splitext(os.path.basename(f))[0]
            pl=strip(html)
            vid=uuid.uuid4().int & ((1<<64)-1)
            vec=self.embedding_model.encode(pl).tolist()
            payload={"tag":tg,"content":pl,"source":"wiki"}
            self.q_client.upsert(
                collection_name=self.collection_name,
                points=[{"id":vid,"vector":vec,"payload":payload}]
            )
        await ctx.send("Wiki import done.")

    async def query_llm(self, prompt, channel):
        model=await self.config.model()
        api=await self.config.api_url()
        data={"model":model,"messages":[{"role":"user","content":prompt}],"stream":False,"options":{"num_ctx":14000,"temperature":0}}
        async with channel.typing():
            resp=requests.post(f"{api}/api/chat",json=data,headers={"Content-Type":"application/json"})
            resp.raise_for_status()
            return resp.json().get("message",{}).get("content","No valid response received.")

    async def search_knowledge(self, query, top_k=5):
        await self.ensure_qdrant_client()
        vec=self.embedding_model.encode(query).tolist()
        return self.q_client.search(collection_name=self.collection_name,query_vector=vec,limit=top_k)

    async def process_question(self, question, channel, author=None):
        res=await self.search_knowledge(question,top_k=5)
        if not res: return await channel.send("No relevant information found.")
        ctxtxt="\n\n".join(f"[{r.id}] ({r.payload.get('tag','NoTag')}): {r.payload.get('content','')}" for r in res)
        ctxclean=re.sub(r"\s+"," ",re.sub(r"<.*?>","",ctxtxt))[:4000]
        p=f"Using the following context:\n{ctxclean}\n\nPlease answer concisely. Include links as Markdown.\n\nQuestion: {question}"
        ans=await self.query_llm(p,channel)
        await channel.send(ans)

    @commands.command()
    async def askllm(self, ctx, *, question: str):
        await self.process_question(question, ctx.channel)

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmodel(self, ctx, model: str): await self.config.model.set(model); await ctx.send(f"Model set to {model}")
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setapi(self, ctx, url: str): await self.config.api_url.set(url.rstrip('/')); await ctx.send(f"API URL set to {url}")
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setqdrant(self, ctx, url: str): await self.config.qdrant_url.set(url.rstrip('/')); self.q_client=QdrantClient(url=url.rstrip('/')); await ctx.send(f"Qdrant URL set to {url}")
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def initcollection(self, ctx):
        await self.ensure_qdrant_client()
        try:
            self.q_client.recreate_collection(collection_name=self.collection_name,vectors_config={"size":384,"distance":"Cosine"})
            await ctx.send("Collection initialized.")
        except Exception as e:
            await ctx.send(f"Init error: {e}")

    @commands.Cog.listener()
    async def on_ready(self): print("LLMManager loaded.")

def setup(bot):
    bot.add_cog(LLMManager(bot))
