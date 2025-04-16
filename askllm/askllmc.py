import discord
import subprocess
import sys
import json
import re
import requests
import time
import asyncio

from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path
from redbot.core.utils.chat_formatting import box
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

class LLMManager(commands.Cog):
    """Cog to interact with the LLM using a Qdrant-based knowledge storage."""
    
    def __init__(self, bot):
        self.bot = bot
        # Konfiguration: API-, Qdrant-URL und Modell
        self.config = Config.get_conf(self, identifier=9999999999)
        self.config.register_global(
            model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333"
        )
        self.collection_name = "fus_wiki"  # Name der Qdrant-Collection
        
        # Qdrant-Client und Embedding-Modell initialisieren
        qdrant_url = self.config.get_raw("qdrant_url", default="http://192.168.10.5:6333")
        self.q_client = QdrantClient(url=qdrant_url)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # erzeugt 384-dim Vektoren
    
    # --- Wissensbasis: Hinzufügen, Anzeigen, Löschen, Editieren ---
    def upsert_knowledge(self, tag, content):
        """Berechnet ein Embedding für den Inhalt und fügt es in Qdrant ein."""
        vector = self.embedding_model.encode(content).tolist()
        payload = {"tag": tag, "content": content}
        # Erzeuge eine eindeutige ID anhand der aktuellen Zeit (Millisekunden)
        doc_id = int(time.time() * 1000)
        self.q_client.upsert(
            collection_name=self.collection_name,
            points=[{"id": doc_id, "vector": vector, "payload": payload}]
        )
    
    @commands.command()
    async def llmknow(self, ctx, tag: str, *, info: str):
        """Adds new information under the specified tag into Qdrant."""
        self.upsert_knowledge(tag.lower(), info)
        await ctx.send(f"Added info under '{tag.lower()}'.")
    
    def _get_all_knowledge_sync(self):
        """Liefert alle Dokumente aus der Qdrant-Collection."""
        return list(self.q_client.scroll_iter(collection_name=self.collection_name, with_payload=True))
    
    async def get_all_knowledge(self):
        return await asyncio.get_running_loop().run_in_executor(None, self._get_all_knowledge_sync)
    
    def _delete_knowledge_by_id_sync(self, doc_id):
        self.q_client.delete(collection_name=self.collection_name, points=[doc_id])
    
    async def llmknowdelete(self, ctx, doc_id: int):
        """Deletes a knowledge entry from Qdrant by its ID."""
        await asyncio.get_running_loop().run_in_executor(None, self._delete_knowledge_by_id_sync, doc_id)
        await ctx.send(f"Deleted entry with ID {doc_id}.")
    
    def _delete_knowledge_by_tag_sync(self, tag):
        filt = {"must": [{"key": "tag", "match": {"value": tag}}]}
        self.q_client.delete(collection_name=self.collection_name, filter=filt)
    
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx, tag: str):
        """Deletes all knowledge entries with a given tag."""
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
        await asyncio.get_running_loop().run_in_executor(None, self._edit_knowledge_sync, doc_id, new_tag.lower(), new_content)
        await ctx.send(f"Updated entry {doc_id}.")
    
    @commands.command()
    async def llmknowshow(self, ctx):
        """Displays all knowledge entries from Qdrant in chunks of <= 2000 characters."""
        points = await self.get_all_knowledge()
        if not points:
            await ctx.send("No knowledge entries stored.")
            return
        
        # Erstelle einen aggregierten Text, der jeden Eintrag formatiert.
        aggregated = "\n".join([
            f"[{point.id}] ({point.payload.get('tag', 'NoTag')}): {point.payload.get('content', '')}"
            for point in points
        ])
        # Teile den Text in Chunks, damit keine Nachricht mehr als 2000 Zeichen hat.
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
    
    # --- LLM-Abfrage ---
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
    
    # --- Frageverarbeitung mit semantischer Suche über Qdrant ---
    async def search_knowledge(self, query, top_k=5):
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
        
        # Aggregiere den Kontext aus den top Ergebnissen.
        aggregated_context = "\n\n".join([
            f"[{result.id}] ({result.payload.get('tag', 'NoTag')}): {result.payload.get('content', '')}"
            for result in qdrant_results
        ])
        
        # Hilfsfunktionen zur Bereinigung:
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
            "Please answer the following question concisely and accurately. "
            "Include any relevant links as Markdown if present.\n\n"
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
        try:
            self.q_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={"size": 384, "distance": "Cosine"}
            )
            await ctx.send(f"Collection '{self.collection_name}' initialized in Qdrant.")
        except Exception as e:
            await ctx.send(f"Error initializing collection: {e}")
    
def setup(bot):
    bot.add_cog(LLMManager(bot))
