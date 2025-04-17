import discord
import subprocess
import sys
import json
import re
import requests
import time
import asyncio
import glob
import os
import uuid
import shutil

from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path
from redbot.core.utils.chat_formatting import box
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

class LLMManager(commands.Cog):
    """Cog to interact with the LLM using a Qdrant-based knowledge storage.

    This cog uses Qdrant as its exclusive knowledge base.
    Knowledge entries can be added manually (source "manual") with !llmknow,
    or imported from a GitHub Wiki using !importwiki (which marks these with source "wiki").
    The commands !llmknowshow, !llmknowdelete, !llmknowdeletetag and !llmknowedit allow management of the stored knowledge.
    The command !askllm (and automatic processing when mentioned) perform a semantic search in Qdrant,
    aggregate the context and then query the LLM for a final answer.
    """
    
    def __init__(self, bot):
        self.bot = bot
        # Konfiguration: Modell, API-URL und Qdrant-URL
        self.config = Config.get_conf(self, identifier=9999999999)
        self.config.register_global(
            model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333"
        )
        self.collection_name = "fus_wiki"  # Name der Qdrant-Collection
        self.q_client = None  # Lazy initialization
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Erzeugt 384-dim Vektoren

    async def ensure_qdrant_client(self):
        """Stellt sicher, dass der Qdrant-Client initialisiert ist."""
        if not self.q_client:
            qdrant_url = await self.config.get_raw("qdrant_url", default="http://192.168.10.5:6333")
            self.q_client = QdrantClient(url=qdrant_url)

    # --- Knowledge Management in Qdrant ---
    def upsert_knowledge(self, tag, content, source="manual"):
        """Berechnet das Embedding für den Inhalt und fügt den Wissenspunkt in Qdrant ein.
        Der Parameter 'source' unterscheidet zwischen manuellen und Wiki-Einträgen.
        Falls die Collection nicht existiert, wird sie erstellt."""
        try:
            self.q_client.get_collection(collection_name=self.collection_name)
        except Exception:
            self.q_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={"size": 384, "distance": "Cosine"}
            )
        vector = self.embedding_model.encode(content).tolist()
        payload = {"tag": tag, "content": content, "source": source}
        doc_id = uuid.uuid4().int  # garantiert einmalige ID
        self.q_client.upsert(
            collection_name=self.collection_name,
            points=[{"id": doc_id, "vector": vector, "payload": payload}]
        )

    @commands.command()
    async def llmknow(self, ctx, tag: str, *, info: str):
        """Adds new manual knowledge under the specified tag into Qdrant."""
        await self.ensure_qdrant_client()
        self.upsert_knowledge(tag.lower(), info, source="manual")
        await ctx.send(f"Added manual info under '{tag.lower()}'.")
    
    def _get_all_knowledge_sync(self):
        """Liefert alle Dokumente aus der Qdrant-Collection als Liste."""
        return list(self.q_client.scroll(collection_name=self.collection_name, with_payload=True))
    
    async def get_all_knowledge(self):
        await self.ensure_qdrant_client()
        return await asyncio.get_running_loop().run_in_executor(None, self._get_all_knowledge_sync)
    
    def _delete_knowledge_by_id_sync(self, doc_id):
        self.q_client.delete(
            collection_name=self.collection_name,
            points_selector=[doc_id]
        )

    @commands.command()
    async def llmknowdelete(self, ctx, doc_id: int):
        """Deletes a knowledge entry from Qdrant by its ID."""
        await self.ensure_qdrant_client()
        await asyncio.get_running_loop().run_in_executor(
            None, self._delete_knowledge_by_id_sync, doc_id
        )
        await ctx.send(f"Deleted entry with ID {doc_id}.")

    
    def _delete_knowledge_by_tag_sync(self, tag):
        filt = {"must": [{"key": "tag", "match": {"value": tag}}]}
        self.q_client.delete(collection_name=self.collection_name, filter=filt)
    
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx, tag: str):
        """Deletes all knowledge entries with the given tag."""
        await self.ensure_qdrant_client()
        await asyncio.get_running_loop().run_in_executor(None, self._delete_knowledge_by_tag_sync, tag.lower())
        await ctx.send(f"Deleted all entries with tag '{tag.lower()}'.")
    
    def _delete_wiki_entries_sync(self):
        """Deletes all Wiki entries from Qdrant (where source is 'wiki')."""
        filt = {"must": [{"key": "source", "match": {"value": "wiki"}}]}
        self.q_client.delete(collection_name=self.collection_name, filter=filt)
    
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def deletewiki(self, ctx):
        """Deletes only the Wiki-based knowledge entries."""
        await self.ensure_qdrant_client()
        await asyncio.get_running_loop().run_in_executor(None, self._delete_wiki_entries_sync)
        await ctx.send("Deleted all Wiki entries.")
    
    def _edit_knowledge_sync(self, doc_id, new_tag, new_content):
        new_vector = self.embedding_model.encode(new_content).tolist()
        payload = {"tag": new_tag, "content": new_content}
        self.q_client.upsert(
            collection_name=self.collection_name,
            points=[{"id": doc_id, "vector": new_vector, "payload": payload}]
        )
    
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowedit(self, ctx, doc_id: int, new_tag: str, *, new_content: str):
        """Edits a knowledge entry by re-upserting it with a new tag and content."""
        await self.ensure_qdrant_client()
        await asyncio.get_running_loop().run_in_executor(None, self._edit_knowledge_sync, doc_id, new_tag.lower(), new_content)
        await ctx.send(f"Updated entry {doc_id}.")
    
    @commands.command()
    async def llmknowshow(self, ctx):
        """Displays all knowledge entries from Qdrant in chunks of <= 2000 characters."""
        await self.ensure_qdrant_client()
        # Lade bis zu 1000 Einträge auf einmal
        points = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: list(
                self.q_client.scroll(
                    collection_name=self.collection_name,
                    with_payload=True,
                    limit=1000
                )
            )
        )
        if not points:
            return await ctx.send("No knowledge entries stored.")
    
        seen = set()
        lines = []
        for pt in points:
            # Extrahiere id und payload
            if hasattr(pt, "id") and hasattr(pt, "payload"):
                raw_id = pt.id
                payload = pt.payload
            elif isinstance(pt, dict):
                raw_id = pt.get("id")
                payload = pt.get("payload", {})
            else:
                try:
                    raw_id, payload = pt[0], pt[1]
                except:
                    continue
    
            # Cast ID & Payload zu hashbaren Typen
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
            source = payload.get("source", "unknown")
            content = payload.get("content", "")
            # Eine einzige Zeile, ohne echte Zeilenumbrüche
            single_line = " ".join(content.splitlines())
            lines.append(f"[{id_key}] ({tag}, src={source}): {single_line}")
    
        # Chunking in Blöcke à 2000 Zeichen
        max_length = 2000
        header, footer = "```\n", "```"
        chunks, cur = [], header
        for line in lines:
            if len(cur) + len(line) + 1 > max_length - len(footer):
                chunks.append(cur + footer)
                cur = header + line + "\n"
            else:
                cur += line + "\n"
        if cur != header:
            chunks.append(cur + footer)
    
        for chunk in chunks:
            await ctx.send(chunk)

    
    # --- LLM-Abfrage (z. B. via Ollama API) ---
    async def query_llm(self, prompt, channel):
        model = await self.config.model()
        api_url = await self.config.api_url()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_ctx": 14000, "temperature": 0}
        }
        headers = {"Content-Type": "application/json"}
        async with channel.typing():
            resp = requests.post(f"{api_url}/api/chat", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data.get("message", {}).get("content", "No valid response received.")
    
    async def search_knowledge(self, query, top_k=5):
        await self.ensure_qdrant_client()
        query_vector = self.embedding_model.encode(query).tolist()
        results = self.q_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return results
    
    async def process_question(self, question, channel, author=None):
        # Führe eine semantische Suche in Qdrant durch.
        qdrant_results = await self.search_knowledge(question, top_k=5)
        if not qdrant_results:
            await channel.send("No relevant information found.")
            return
        aggregated_context = "\n\n".join([
            f"[{result.id}] ({result.payload.get('tag', 'NoTag')}): {result.payload.get('content', '')}"
            for result in qdrant_results
        ])
        def strip_html(raw_html):
            cleanr = re.compile('<.*?>')
            return re.sub(cleanr, '', raw_html)
        def validate_links(text):
            url_regex = r'https?://[^\s]+'
            links = re.findall(url_regex, text)
            for link in links:
                if "example.com" in link:
                    text = text.replace(link, "")
                else:
                    try:
                        response = requests.head(link, timeout=5)
                        if response.status_code >= 400:
                            text = text.replace(link, "")
                    except Exception:
                        text = text.replace(link, "")
            text = re.sub(r'\s+', ' ', text)
            return text
        context_text = strip_html(aggregated_context)
        context_text = validate_links(context_text)
        if len(context_text) > 4000:
            context_text = context_text[:4000] + "\n...[truncated]"
        final_prompt = (
            f"Using the following context extracted from the documentation:\n{context_text}\n\n"
            "Please answer the following question concisely and accurately. Do not fabricate details. If the context contains conflicting information, explain briefly and state the most commonly recommended option. If no clear recommendation exists, state so. Include any relevant links as Markdown if present.\n\n"
            f"Question: {question}"
        )
        final_answer = await self.query_llm(final_prompt, channel)
        await channel.send(final_answer)
    
    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """Sends your question to the LLM using knowledge from Qdrant."""
        await self.process_question(question, ctx.channel, author=ctx.author)
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return
        if self.bot.user.mentioned_in(message):
            question = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
            if question:
                await self.process_question(question, message.channel, author=message.author)
    
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmodel(self, ctx, model: str):
        await self.config.model.set(model)
        await ctx.send(f"LLM model set to '{model}'.")
    
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setapi(self, ctx, url: str):
        await self.config.api_url.set(url.rstrip("/"))
        await ctx.send(f"Ollama API URL set to '{url}'.")
    
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setqdrant(self, ctx, url: str):
        await self.config.qdrant_url.set(url.rstrip("/"))
        self.q_client = QdrantClient(url=url.rstrip("/"))
        await ctx.send(f"Qdrant URL set to '{url}'.")
    
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def initcollection(self, ctx):
        """Initializes (or recreates) the Qdrant collection for storing knowledge."""
        await self.ensure_qdrant_client()
        try:
            self.q_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={"size": 384, "distance": "Cosine"}
            )
            await ctx.send(f"Collection '{self.collection_name}' initialized in Qdrant.")
        except Exception as e:
            await ctx.send(f"Error initializing collection: {e}")
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def importwiki(self, ctx, wiki_url: str = None):
        """
        Clones or updates a GitHub Wiki into Qdrant.
        Manual entries remain untouched.
        """
        await self.ensure_qdrant_client()
        wiki_url = wiki_url or "https://github.com/Kvitekvist/FUS.wiki.git"
        data_path = str(cog_data_path(self))
        clone_path = os.path.join(data_path, "wiki")
    
        # 1) lösche alte Wiki-Einträge
        def _del_sync():
            filt = {"must":[{"key":"source","match":{"value":"wiki"}}]}
            self.q_client.delete(collection_name=self.collection_name, filter=filt)
        await asyncio.get_running_loop().run_in_executor(None, _del_sync)
    
        # 2) clone oder pull
        if os.path.isdir(os.path.join(clone_path, ".git")):
            subprocess.run(["git","-C",clone_path,"pull"], check=False)
            await ctx.send("Wiki repo updated.")
        else:
            if os.path.exists(clone_path):
                shutil.rmtree(clone_path)
            subprocess.run(["git","clone",wiki_url,clone_path], check=True)
            await ctx.send("Wiki repo cloned.")
    
        # 3) verarbeite MD-Dateien
        import markdown
        from bs4 import BeautifulSoup
    
        def strip_html(h): return re.sub(r"<.*?>","",h)
        def gen_tags(html):
            soup = BeautifulSoup(html,"html.parser")
            hs = soup.find_all(re.compile("^h[1-3]$"))
            t = [h.get_text().strip() for h in hs if h.get_text().strip()]
            return ", ".join(dict.fromkeys(t)) or None
    
        md_files = glob.glob(os.path.join(clone_path,"*.md"))
        await ctx.send(f"Found {len(md_files)} wiki pages. Importing…")
        for fp in md_files:
            text = open(fp,encoding="utf-8").read()
            html = markdown.markdown(text)
            tags = gen_tags(html) or os.path.splitext(os.path.basename(fp))[0]
            plain = strip_html(html)
            # upsert mit unique ID
            import uuid
            vid = uuid.uuid4().int
            vec = self.embedding_model.encode(plain).tolist()
            payload = {"tag":tags,"content":plain,"source":"wiki"}
            self.q_client.upsert(
                collection_name=self.collection_name,
                points=[{"id":vid,"vector":vec,"payload":payload}]
            )
        await ctx.send("Wiki import done.")

    @commands.Cog.listener()
    async def on_ready(self):
        print("LLMManager cog loaded.")
    
def setup(bot):
    bot.add_cog(LLMManager(bot))
