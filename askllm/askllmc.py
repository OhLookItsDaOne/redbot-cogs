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

    This cog uses Qdrant as its exclusive knowledge base.
    Knowledge entries can be added manually (source "manual") with !llmknow,
    or imported from a GitHub Wiki using !importwiki (source "wiki").
    Commands:
      - !llmknow, !llmknowshow, !llmknowdelete, !llmknowdeletetag, !llmknowedit
      - !importwiki to import/update wiki pages
      - !askllm to query LLM based on Qdrant context
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
        except Exception:
            self.q_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={"size": 384, "distance": "Cosine"}
            )
        vector = self.embedding_model.encode(content).tolist()
        payload = {"tag": tag, "content": content, "source": source}
        # 64-bit unique ID
        doc_id = uuid.uuid4().int & ((1 << 64) - 1)
        self.q_client.upsert(
            collection_name=self.collection_name,
            points=[{"id": doc_id, "vector": vector, "payload": payload}]
        )

    @commands.command()
    async def llmknow(self, ctx, tag: str, *, info: str):
        await self.ensure_qdrant_client()
        self.upsert_knowledge(tag.lower(), info, source="manual")
        await ctx.send(f"Added manual info under '{tag.lower()}'.")

    def _get_all_knowledge_sync(self):
        # Direkt alle Knowledge-Punkte als Liste zurückgeben
        return list(
            self.q_client.scroll(
                collection_name=self.collection_name,
                with_payload=True,
                limit=1000
            )
        )

    async def get_all_knowledge(self):
        await self.ensure_qdrant_client()
        return await asyncio.get_running_loop().run_in_executor(None, self._get_all_knowledge_sync)

    def _delete_knowledge_by_id_sync(self, doc_id):
        self.q_client.delete(collection_name=self.collection_name, points_selector=[doc_id])

    @commands.command()
    async def llmknowdelete(self, ctx, doc_id: int):
        await self.ensure_qdrant_client()
        await asyncio.get_running_loop().run_in_executor(None, self._delete_knowledge_by_id_sync, doc_id)
        await ctx.send(f"Deleted entry with ID {doc_id}.")

    def _delete_knowledge_by_tag_sync(self, tag):
        filt = {"must": [{"key": "tag", "match": {"value": tag}}]}
        self.q_client.delete(collection_name=self.collection_name, points_selector=[], filter=filt)

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx, tag: str):
        await self.ensure_qdrant_client()
        await asyncio.get_running_loop().run_in_executor(None, self._delete_knowledge_by_tag_sync, tag.lower())
        await ctx.send(f"Deleted all entries with tag '{tag.lower()}'.")

    def _edit_knowledge_sync(self, doc_id, new_tag, new_content):
        vector = self.embedding_model.encode(new_content).tolist()
        payload = {"tag": new_tag, "content": new_content}
        self.q_client.upsert(
            collection_name=self.collection_name,
            points=[{"id": doc_id, "vector": vector, "payload": payload}]
        )

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowedit(self, ctx, doc_id: int, new_tag: str, *, new_content: str):
        await self.ensure_qdrant_client()
        await asyncio.get_running_loop().run_in_executor(None, self._edit_knowledge_sync, doc_id, new_tag.lower(), new_content)
        await ctx.send(f"Updated entry {doc_id}.")

    @commands.command()
    async def llmknowshow(self, ctx):
        await self.ensure_qdrant_client()
        points = await self.get_all_knowledge()
        if not points:
            return await ctx.send("No knowledge entries stored.")
        seen, lines = set(), []
        for pt in points:
            if hasattr(pt, "id") and hasattr(pt, "payload"):
                raw_id, payload = pt.id, pt.payload
            elif isinstance(pt, dict):
                raw_id, payload = pt.get("id"), pt.get("payload", {})
            else:
                try:
                    raw_id, payload = pt[0], pt[1]
                except:
                    continue
            id_key = str(raw_id)
            if id_key in seen:
                continue
            seen.add(id_key)
            if not isinstance(payload, dict):
                try:
                    payload = payload.dict()
                except:
                    payload = dict(payload)
            tag = payload.get("tag", "NoTag")
            src = payload.get("source", "unknown")
            content = payload.get("content", "")
            single_line = " ".join(content.splitlines())
            lines.append(f"[{id_key}] ({tag}, src={src}): {single_line}")
        max_len, header, footer = 2000, "```\n", "```"
        chunks, cur = [], header
        for l in lines:
            if len(cur) + len(l) + 1 > max_len - len(footer):
                chunks.append(cur + footer)
                cur = header + l + "\n"
            else:
                cur += l + "\n"
        if cur != header:
            chunks.append(cur + footer)
        for c in chunks:
            await ctx.send(c)

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def importwiki(self, ctx, wiki_url: str = None):
        await self.ensure_qdrant_client()
        wiki_url = wiki_url or "https://github.com/Kvitekvist/FUS.wiki.git"
        base = str(cog_data_path(self))
        os.makedirs(base, exist_ok=True)
        clone_path = os.path.join(base, "wiki")
        # Lösche alte Wiki-Einträge über Filter
        def _del_sync():
            filt = {"must": [{"key": "source", "match": {"value": "wiki"}}]}
            self.q_client.delete(collection_name=self.collection_name, points_selector=[], filter=filt)
        await asyncio.get_running_loop().run_in_executor(None, _del_sync)
        # Clone oder Pull
        if os.path.isdir(os.path.join(clone_path, ".git")):
            subprocess.run(["git", "-C", clone_path, "pull"], check=False)
            await ctx.send("Wiki repo updated.")
        else:
            if os.path.exists(clone_path): shutil.rmtree(clone_path)
            subprocess.run(["git", "clone", wiki_url, clone_path], check=True)
            await ctx.send("Wiki repo cloned.")
        # Verarbeite Markdown
        import markdown
        from bs4 import BeautifulSoup
        def strip_html(h): return re.sub(r"<.*?>", "", h)
        def gen_tags(html):
            soup = BeautifulSoup(html, "html.parser")
            hs = soup.find_all(re.compile("^h[1-3]$"))
            return ", ".join(dict.fromkeys(h.get_text().strip() for h in hs if h.get_text().strip())) or None
        md_files = glob.glob(os.path.join(clone_path, "*.md"))
        await ctx.send(f"Found {len(md_files)} wiki pages. Importing…")
        for fp in md_files:
            text = open(fp, encoding="utf-8").read()
            html = markdown.markdown(text)
            tags = gen_tags(html) or os.path.splitext(os.path.basename(fp))[0]
            plain = strip_html(html)
            vid = uuid.uuid4().int & ((1 << 64) - 1)
            vec = self.embedding_model.encode(plain).tolist()
            payload = {"tag": tags, "content": plain, "source": "wiki"}
            self.q_client.upsert(collection_name=self.collection_name, points=[{"id": vid, "vector": vec, "payload": payload}])
        await ctx.send("Wiki import done.")

    async def query_llm(self, prompt, channel):
        model = await self.config.model()
        api_url = await self.config.api_url()
        headers = {"Content-Type": "application/json"}
        data = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "options": {"num_ctx": 14000, "temperature": 0}}
        async with channel.typing():
            resp = requests.post(f"{api_url}/api/chat", json=data, headers=headers)
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "No valid response received.")

    async def search_knowledge(self, query, top_k=5):
        await self.ensure_qdrant_client()
        vec = self.embedding_model.encode(query).tolist()
        return self.q_client.search(collection_name=self.collection_name, query_vector=vec, limit=top_k)

    async def process_question(self, question, channel, author=None):
        results = await self.search_knowledge(question, top_k=5)
        if not results:
            return await channel.send("No relevant information found.")
        context = "\n\n".join(f"[{r.id}] ({r.payload.get('tag','NoTag')}): {r.payload.get('content','')}" for r in results)
        def strip(raw): return re.sub(r'<.*?>', '', raw)
        def validate(text):
            for url in re.findall(r'https?://[^\s]+', text):
                if 'example.com' in url or requests.head(url, timeout=5).status_code >= 400:
                    text = text.replace(url, '')
            return re.sub(r'\s+', ' ', text)
        ctxt = validate(strip(context))[:4000]
        prompt = (f"Using the following context extracted from the documentation:\n{ctxt}\n\n"
                  "Please answer concisely and accurately. Include relevant links as Markdown.\n\n"
                  f"Question: {question}")
        ans = await self.query_llm(prompt, channel)
        await channel.send(ans)

    @commands.command()
    async def askllm(self, ctx, *, question: str):
        await self.process_question(question, ctx.channel, author=ctx.author)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild: return
        if self.bot.user.mentioned_in(message):
            q = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
            if q: await self.process_question(q, message.channel, author=message.author)

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmodel(self, ctx, model: str): await self.config.model.set(model); await ctx.send(f"LLM model set to '{model}'.")
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setapi(self, ctx, url: str):  await self.config.api_url.set(url.rstrip("/")); await ctx.send(f"Ollama URL set to '{url}'.")
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setqdrant(self, ctx, url: str): await self.config.qdrant_url.set(url.rstrip("/")); self.q_client = QdrantClient(url=url.rstrip("/")); await ctx.send(f"Qdrant URL set to '{url}'.")
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def initcollection(self, ctx):
        await self.ensure_qdrant_client()
        try:
            self.q_client.recreate_collection(collection_name=self.collection_name, vectors_config={"size":384,"distance":"Cosine"})
            await ctx.send(f"Collection '{self.collection_name}' initialized.")
        except Exception as e:
            await ctx.send(f"Error initializing: {e}")

    @commands.Cog.listener()
    async def on_ready(self): print("LLMManager cog loaded.")

def setup(bot):
    bot.add_cog(LLMManager(bot))
