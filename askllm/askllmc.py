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
import shutil

from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path
from redbot.core.utils.chat_formatting import box
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

class LLMManager(commands.Cog):
    """Cog to interact with the LLM and manage a Qdrant-based knowledge storage."""
    
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
        self.q_client = None  # Wird lazy initialisiert
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Erzeugt 384-dim Vektoren

    async def ensure_qdrant_client(self):
        """Stellt sicher, dass der Qdrant-Client initialisiert ist."""
        if not self.q_client:
            qdrant_url = await self.config.get_raw("qdrant_url", default="http://192.168.10.5:6333")
            self.q_client = QdrantClient(url=qdrant_url)

    # --- Knowledge Management in Qdrant ---
    def upsert_knowledge(self, tag, content):
        """Berechnet das Embedding für den Inhalt und fügt den Wissenspunkt in Qdrant ein.
           Falls die Collection nicht existiert, wird sie erstellt."""
        # Stelle sicher, dass die Collection existiert:
        try:
            self.q_client.get_collection(collection_name=self.collection_name)
        except Exception:
            self.q_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={"size": 384, "distance": "Cosine"}
            )
        vector = self.embedding_model.encode(content).tolist()
        payload = {"tag": tag, "content": content}
        # Verwende als ID die aktuelle Zeit in Millisekunden
        doc_id = int(time.time() * 1000)
        self.q_client.upsert(
            collection_name=self.collection_name,
            points=[{"id": doc_id, "vector": vector, "payload": payload}]
        )

    @commands.command()
    async def llmknow(self, ctx, tag: str, *, info: str):
        """Adds new information under the specified tag into Qdrant."""
        await self.ensure_qdrant_client()
        self.upsert_knowledge(tag.lower(), info)
        await ctx.send(f"Added info under '{tag.lower()}'.")
    
    def _get_all_knowledge_sync(self):
        """Liefert alle Dokumente aus der Qdrant-Collection als Liste."""
        return list(self.q_client.scroll(collection_name=self.collection_name, with_payload=True))
    
    async def get_all_knowledge(self):
        await self.ensure_qdrant_client()
        return await asyncio.get_running_loop().run_in_executor(None, self._get_all_knowledge_sync)
    
    def _delete_knowledge_by_id_sync(self, doc_id):
        self.q_client.delete(collection_name=self.collection_name, points=[doc_id])
    
    async def llmknowdelete(self, ctx, doc_id: int):
        """Deletes a knowledge entry from Qdrant by its ID."""
        await self.ensure_qdrant_client()
        await asyncio.get_running_loop().run_in_executor(None, self._delete_knowledge_by_id_sync, doc_id)
        await ctx.send(f"Deleted entry with ID {doc_id}.")
    
    def _delete_knowledge_by_tag_sync(self, tag):
        filt = {"must": [{"key": "tag", "match": {"value": tag}}]}
        self.q_client.delete(collection_name=self.collection_name, filter=filt)
    
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx, tag: str):
        """Deletes all knowledge entries with a given tag."""
        await self.ensure_qdrant_client()
        await asyncio.get_running_loop().run_in_executor(None, self._delete_knowledge_by_tag_sync, tag.lower())
        await ctx.send(f"Deleted all entries with tag '{tag.lower()}'.")
    
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
        points = await self.get_all_knowledge()
        if not points:
            await ctx.send("No knowledge entries stored.")
            return
        
        aggregated = "\n".join([
            f"[{point.id}] ({point.payload.get('tag','NoTag')}): {point.payload.get('content','')}"
            for point in points
        ])
        max_length = 2000
        header = "```\n"
        footer = "```"
        chunks = []
        current_chunk = header
        for line in aggregated.split("\n"):
            if len(current_chunk) + len(line) + 1 > max_length - len(footer):
                chunks.append(current_chunk + footer)
                current_chunk = header + line + "\n"
            else:
                current_chunk += line + "\n"
        if current_chunk != header:
            chunks.append(current_chunk + footer)
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
            "Please answer the following question concisely and accurately. Include any relevant links as Markdown if present.\n\n"
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
    
    # --- Import Wiki Command ---
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def importwiki(self, ctx):
        """
        Imports the GitHub Wiki into the Qdrant knowledge base.
        It clears the collection first, then clones (or updates) the repository,
        processes all markdown files, and upserts them into Qdrant.
        """
        await self.ensure_qdrant_client()
        # Recreate the collection to clear any existing data
        try:
            self.q_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={"size": 384, "distance": "Cosine"}
            )
            await ctx.send(f"Collection '{self.collection_name}' recreated in Qdrant (old entries cleared).")
        except Exception as e:
            await ctx.send(f"Error recreating collection: {e}")
            return
        
        # Git-Repository-Info
        repo_url = "https://github.com/Kvitekvist/FUS.wiki.git"
        # Lokaler Zielpfad (Passe diesen Pfad an deine Umgebung an)
        clone_path = r"C:\Users\Tim.Thießen\Documents\wiki"
        
        # Klonen oder Aktualisieren des Repositories
        if os.path.exists(clone_path):
            if os.path.isdir(os.path.join(clone_path, ".git")):
                try:
                    subprocess.run(["git", "-C", clone_path, "pull"], check=True)
                    await ctx.send("Repository updated.")
                except Exception as e:
                    await ctx.send(f"Error updating repository: {e}")
                    return
            else:
                shutil.rmtree(clone_path)
                try:
                    subprocess.run(["git", "clone", repo_url, clone_path], check=True)
                    await ctx.send("Repository cloned.")
                except Exception as e:
                    await ctx.send(f"Error cloning repository: {e}")
                    return
        else:
            try:
                subprocess.run(["git", "clone", repo_url, clone_path], check=True)
                await ctx.send("Repository cloned.")
            except Exception as e:
                await ctx.send(f"Error cloning repository: {e}")
                return
        
        # Hilfsfunktionen
        def strip_html(raw_html):
            cleanr = re.compile('<.*?>')
            return re.sub(cleanr, '', raw_html)
        
        def generate_smart_tags(html_content):
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            headers = soup.find_all(re.compile('^h[1-3]$'))
            tags = [header.get_text().strip() for header in headers if header.get_text().strip()]
            unique_tags = list(dict.fromkeys(tags))
            if unique_tags:
                return ", ".join(unique_tags)
            return "No Tags"
        
        def process_markdown_file(file_path):
            """Liest eine Markdown-Datei, wandelt sie in HTML um, generiert Smart Tags und bereinigt den Text."""
            with open(file_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            html_content = markdown.markdown(md_content)
            smart_tags = generate_smart_tags(html_content)
            plain_text = strip_html(html_content)
            return smart_tags if smart_tags != "No Tags" else os.path.splitext(os.path.basename(file_path))[0], plain_text
        
        # Finde alle Markdown-Dateien im Repository
        files = glob.glob(os.path.join(clone_path, "*.md"))
        await ctx.send(f"Found {len(files)} markdown files. Starting import...")
        
        for file_path in files:
            tag, content = process_markdown_file(file_path)
            try:
                vector = self.embedding_model.encode(content).tolist()
                payload = {"tag": tag, "content": content}
                doc_id = int(time.time() * 1000)
                self.q_client.upsert(
                    collection_name=self.collection_name,
                    points=[{"id": doc_id, "vector": vector, "payload": payload}]
                )
                await ctx.send(f"Inserted document ID {doc_id} with tag '{tag}' from file {file_path}.")
            except Exception as e:
                await ctx.send(f"Error inserting document from {file_path}: {e}")
            time.sleep(0.1)
        
        await ctx.send("Wiki import completed.")
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return
        if self.bot.user.mentioned_in(message):
            question = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
            if question:
                await self.process_question(question, message.channel, author=message.author)
    
    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """Sends your question to the LLM using knowledge from Qdrant."""
        await self.process_question(question, ctx.channel, author=ctx.author)
    
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
    
    # Weitere Konfigurationsbefehle...
    @commands.Cog.listener()
    async def on_ready(self):
        print("LLMManager cog loaded.")
    
def setup(bot):
    bot.add_cog(LLMManager(bot))
