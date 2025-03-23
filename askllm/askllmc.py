import discord
import subprocess
import sys
import json  # <-- We'll parse the LLM's JSON here.

# Module automatisch installieren
subprocess.check_call([sys.executable, "-m", "pip", "install", "mysql-connector-python"])

import mysql.connector
import re
import requests
from redbot.core import commands, Config

class LLMManager(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9999999999)
        self.config.register_global(
            model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            db_config={"host": "", "user": "", "password": "", "database": ""}
        )

    async def get_db_config(self):
        return await self.config.db_config()

    async def add_tag_content(self, tag, content):
        db = mysql.connector.connect(**await self.get_db_config())
        cursor = db.cursor()
        cursor.execute("INSERT INTO tags (tag, content) VALUES (%s, %s);", (tag, content))
        db.commit()
        cursor.close()
        db.close()

    async def edit_tag_content(self, entry_id, new_content):
        db = mysql.connector.connect(**await self.get_db_config())
        cursor = db.cursor()
        cursor.execute("UPDATE tags SET content = %s WHERE id = %s;", (new_content, entry_id))
        db.commit()
        cursor.close()
        db.close()

    async def delete_tag_by_id(self, entry_id):
        db = mysql.connector.connect(**await self.get_db_config())
        cursor = db.cursor()
        cursor.execute("DELETE FROM tags WHERE id = %s;", (entry_id,))
        db.commit()
        cursor.close()
        db.close()

    async def delete_tag_by_name(self, tag):
        db = mysql.connector.connect(**await self.get_db_config())
        cursor = db.cursor()
        cursor.execute("DELETE FROM tags WHERE tag = %s;", (tag,))
        db.commit()
        cursor.close()
        db.close()

    async def get_all_content(self):
        db = mysql.connector.connect(**await self.get_db_config())
        cursor = db.cursor()
        cursor.execute("SELECT id, tag, content FROM tags ORDER BY id;")
        results = cursor.fetchall()
        cursor.close()
        db.close()
        return results

    async def query_llm(self, prompt, channel):
        """Calls the main LLM with a straightforward prompt.
           `temperature=0` to minimize creative/hallucinated responses."""
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

    async def pick_best_entry_with_llm(self, question: str, entries: list, channel) -> int:
        """
        Let the LLM score each entry for relevance.
        Return the index of the best entry, or -1 if no entry is relevant.

        We'll ask the LLM for strict JSON like:
        {
          "scores": [ int, ... ]
        }
        where each int is 0..10.
        """
        # Build prompt listing the entries
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
            # remove possible code fences
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

        # if best_score < 2 => not relevant
        if best_score < 2:
            return -1

        return best_index

    async def process_question(self, question, channel, author=None):
        """We do naive DB filtering. Then let the LLM rank them. Finally we output exactly that entry's text."""
        # 1) Naive DB filter
        all_entries = await self.get_all_content()
        words = re.sub(r"[^\\w\\s]", "", question.lower()).split()

        # basic filter
        filtered = []
        for (eid, tag, content) in all_entries:
            # count how many words appear at least once
            score = sum(1 for w in set(words) if w in content.lower())
            if score > 0:
                filtered.append((eid, tag, content))

        if not filtered:
            await channel.send("No relevant info found.")
            return

        # 2) Let LLM pick best among these top 5
        filtered = filtered[:5]
        best_index = await self.pick_best_entry_with_llm(question, filtered, channel)
        if best_index < 0:
            await channel.send("No relevant entry found or user unclear. Please refine question.")
            return

        # 3) Show user the EXACT DB text (no summation => no hallucination)
        eid, best_tag, best_content = filtered[best_index]
        await channel.send(f"[{best_tag}] (ID: {eid})\n{best_content}")

    #--- Commands

    @commands.command()
    async def askllm(self, ctx, *, question: str):
        await self.process_question(question, ctx.channel, author=ctx.author)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return
        if self.bot.user.mentioned_in(message):
            question = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
            if question:
                await self.process_question(question, message.channel, author=message.author)

    #--- Admin commands (unchanged)

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
    async def llmknow(self, ctx, tag: str, *, info: str):
        await self.add_tag_content(tag.lower(), info)
        await ctx.send(f"Added info under '{tag.lower()}'.")

    @commands.command()
    async def llmknowshow(self, ctx):
        results = await self.get_all_content()
        if not results:
            await ctx.send("No tags stored.")
            return

        chunks = []
        current_chunk = "```\n"
        for _id, tag, text in results:
            line = f"[{_id}] ({tag}) {text}\n"
            if len(current_chunk) + len(line) > 1990:
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
        await ctx.send(f"Deleted all entries with tag '{tag}'.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowedit(self, ctx, entry_id: int, *, new_content: str):
        await self.edit_tag_content(entry_id, new_content)
        await ctx.send(f"Updated entry {entry_id}.")
