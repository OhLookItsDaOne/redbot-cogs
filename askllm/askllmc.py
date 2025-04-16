import discord
import subprocess
import sys
import json
import re
import requests
import time
import asyncio

# Dynamische Installation von mysql-connector-python, falls nicht vorhanden.
try:
    import mysql.connector
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mysql-connector-python"])
    import mysql.connector

from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path
from redbot.core.utils.chat_formatting import box
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

class LLMManager(commands.Cog):
    """Cog to interact with the LLM and manage a MariaDB-based knowledge storage."""

    def __init__(self, bot):
        self.bot = bot
        # Registrierung der globalen Konfiguration:
        self.config = Config.get_conf(self, identifier=9999999999)
        self.config.register_global(
            model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            db_config={"host": "", "user": "", "password": "", "database": ""}
        )

    # Hilfsmethode, um DB-Konfiguration abzurufen.
    async def get_db_config(self):
        return await self.config.db_config()

    # Führt synchrone DB-Aufgaben in einem Executor aus, um Blockierungen zu vermeiden.
    async def run_db_task(self, task, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, task, *args, **kwargs)

    # --- Datenbankoperationen (synchron implementiert, asynchron aufrufbar) ---

    def _add_tag_content_sync(self, tag, content, db_config):
        db = mysql.connector.connect(**db_config)
        cursor = db.cursor()
        cursor.execute("INSERT INTO tags (tag, content) VALUES (%s, %s);", (tag, content))
        db.commit()
        cursor.close()
        db.close()

    async def add_tag_content(self, tag, content):
        db_config = await self.get_db_config()
        await self.run_db_task(self._add_tag_content_sync, tag, content, db_config)

    def _edit_tag_content_sync(self, entry_id, new_content, db_config):
        db = mysql.connector.connect(**db_config)
        cursor = db.cursor()
        cursor.execute("UPDATE tags SET content = %s WHERE id = %s;", (new_content, entry_id))
        db.commit()
        cursor.close()
        db.close()

    async def edit_tag_content(self, entry_id, new_content):
        db_config = await self.get_db_config()
        await self.run_db_task(self._edit_tag_content_sync, entry_id, new_content, db_config)

    def _delete_tag_by_id_sync(self, entry_id, db_config):
        db = mysql.connector.connect(**db_config)
        cursor = db.cursor()
        cursor.execute("DELETE FROM tags WHERE id = %s;", (entry_id,))
        db.commit()
        cursor.close()
        db.close()

    async def delete_tag_by_id(self, entry_id):
        db_config = await self.get_db_config()
        await self.run_db_task(self._delete_tag_by_id_sync, entry_id, db_config)

    def _delete_tag_by_name_sync(self, tag, db_config):
        db = mysql.connector.connect(**db_config)
        cursor = db.cursor()
        cursor.execute("DELETE FROM tags WHERE tag = %s;", (tag,))
        db.commit()
        cursor.close()
        db.close()

    async def delete_tag_by_name(self, tag):
        db_config = await self.get_db_config()
        await self.run_db_task(self._delete_tag_by_name_sync, tag, db_config)

    def _get_all_content_sync(self, db_config):
        db = mysql.connector.connect(**db_config)
        cursor = db.cursor()
        cursor.execute("SELECT id, tag, content FROM tags ORDER BY id;")
        results = cursor.fetchall()
        cursor.close()
        db.close()
        return results

    async def get_all_content(self):
        db_config = await self.get_db_config()
        return await self.run_db_task(self._get_all_content_sync, db_config)

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

    # Lässt die LLM eine Liste von DB-Einträgen bewerten, um den besten relevanten Eintrag zu finden.
    async def pick_best_entry_with_llm(self, question: str, entries: list, channel) -> int:
        numbered_entries = []
        for i, (eid, tag, content) in enumerate(entries):
            snippet = (
                f"Entry {i}:\n"
                f"ID: {eid}, Tag: {tag}\n"
                f"{content}\n"
                "---"
            )
            numbered_entries.append(snippet)

        prompt = f"""You are a helpful ranking assistant.

User's question:
{question}

We have {len(numbered_entries)} database entries. Rate each from 0 to 10, how relevant it is to the question.
Return exact JSON in this format (no code fences, no extra text):

{{
  "scores": [score_for_entry0, score_for_entry1, ...]
}}

0 = not relevant at all
10 = extremely relevant

Entries:
{chr(10).join(numbered_entries)}
"""
        llm_response = await self.query_llm(prompt, channel)
        best_index = -1
        best_score = -1
        try:
            json_str = re.sub(r"```(json)?", "", llm_response)
            data = json.loads(json_str)
            scores = data.get("scores", [])
            for i, s in enumerate(scores):
                if s > best_score:
                    best_score = s
                    best_index = i
        except Exception as e:
            await channel.send(f"Error: could not parse LLM JSON rating: {e}")
            return -1

        if best_score < 2:
            return -1
        return best_index

    # Hier wird die Datenbank abgefragt und die Ergebnisse an die LLM übergeben (nur bei einer LLM-Anfrage)
    async def process_question(self, question, channel, author=None):
        all_entries = await self.get_all_content()
        if not all_entries:
            await channel.send("No information stored in the database.")
            return

        # Normalisiere sowohl die Frage als auch jeden Eintrag (Tag und Content), um Vergleiche zu ermöglichen.
        question_norm = re.sub(r"[^\w\s]", "", question.lower())
        words = question_norm.split()

        filtered = []
        for (eid, tag, content) in all_entries:
            # Normalisiere Tag und Content
            tag_norm = re.sub(r"[^\w\s]", "", tag.lower())
            content_norm = re.sub(r"[^\w\s]", "", content.lower())
            combined = f"{tag_norm} {content_norm}"
            score = sum(1 for w in set(words) if w in combined)
            if score > 0:
                filtered.append((eid, tag, content))

        if not filtered:
            await channel.send("No relevant info found.")
            return

        # Begrenze auf die Top 5 Einträge für die LLM-Auswertung
        filtered = filtered[:5]
        best_index = await self.pick_best_entry_with_llm(question, filtered, channel)
        if best_index < 0:
            await channel.send("No relevant entry found or question too unclear. Please refine your question.")
            return

        eid, best_tag, best_content = filtered[best_index]
        await channel.send(f"[{best_tag}] (ID: {eid})\n{best_content}")

    # --- Commands und Listener ---

    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """Sends your question to the LLM using data from the database."""
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
    async def setdb(self, ctx, host: str, user: str, password: str, database: str):
        await self.config.db_config.set({"host": host, "user": user, "password": password, "database": database})
        await ctx.send("Database configuration updated.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def initdb(self, ctx):
        """Initializes the database schema (creates the tags table if it doesn't exist)."""
        db_config = await self.get_db_config()
        try:
            db = mysql.connector.connect(**db_config)
            cursor = db.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    tag VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL
                );
            """)
            db.commit()
            cursor.close()
            db.close()
            await ctx.send("Database schema initialized.")
        except mysql.connector.Error as err:
            await ctx.send(f"Database initialization error: {err}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx, tag: str, *, info: str):
        """Adds new information under a specified tag."""
        await self.add_tag_content(tag.lower(), info)
        await ctx.send(f"Added info under '{tag.lower()}'.")
    @commands.command()
    async def llmknowshow(self, ctx):
        """Displays the current knowledge stored in the DB, splitting output into chunks so that no message exceeds 4000 characters."""
        results = await self.get_all_content()
        if not results:
            await ctx.send("No tags stored.")
            return

        max_length = 4000  # Maximale Zeichenlänge pro Nachricht (inklusive Formatierung)
        chunks = []
        current_chunk = "```\n"
        for _id, tag, text in results:
            line = f"[{_id}] ({tag}) {text}\n"
            # Wenn das Hinzufügen der nächsten Zeile das Limit überschreitet, schließe den aktuellen Chunk ab
            if len(current_chunk) + len(line) > max_length - 3:  # -3 für die abschließenden Backticks
                current_chunk += "```"
                chunks.append(current_chunk)
                current_chunk = "```\n" + line
            else:
                current_chunk += line

        if current_chunk.strip():
            current_chunk += "```"
            chunks.append(current_chunk)

        for chunk in chunks:
            await ctx.send(chunk)


    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdelete(self, ctx, entry_id: int):
        await self.delete_tag_by_id(entry_id)
        await ctx.send(f"Deleted entry with ID {entry_id}.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx, tag: str):
        await self.delete_tag_by_name(tag.lower())
        await ctx.send(f"Deleted all entries with tag '{tag.lower()}'.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowedit(self, ctx, entry_id: int, *, new_content: str):
        await self.edit_tag_content(entry_id, new_content)
        await ctx.send(f"Updated entry {entry_id}.")

def setup(bot):
    bot.add_cog(LLMManager(bot))
