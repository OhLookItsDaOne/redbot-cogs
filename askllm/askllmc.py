import discord
import subprocess
import sys

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
        model = await self.config.model()
        api_url = await self.config.api_url()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_ctx": 32000}
        }
        headers = {"Content-Type": "application/json"}
        async with channel.typing():
            resp = requests.post(f"{api_url}/api/chat", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data.get("message", {}).get("content", "No valid response received.")

    async def process_question(self, question, channel, author=None):
        entries = await self.get_all_content()
        words = re.sub(r"[^\w\s]", "", question.lower()).split()
        relevance_scores = []
        for _id, tag, content in entries:
            score = sum(content.lower().count(w) for w in words)
            if score > 0:
                relevance_scores.append((score, tag, content))
        if not relevance_scores:
            await channel.send("No relevant info found.")
            return
        relevance_scores.sort(reverse=True)
        top_entries = relevance_scores[:20]
        knowledge = "\n".join(f"[{tag}] {content}" for _, tag, content in top_entries)

        chat_history = ""
        if author:
            async for msg in channel.history(limit=20):
                if msg.author == author and not msg.content.startswith("!") and not msg.content.startswith("<@"):
                    chat_history = msg.content.strip() + "\n" + chat_history
                    if chat_history.count("\n") >= 4:
                        break

        prompt = (
            f"You are a helpful assistant with access to a knowledge base.\n"
            f"Use the following knowledge base to answer the user's question.\n"
            f"Correct vague language or typos, and be specific and helpful.\n"
            f"Avoid repeating what's already been said in previous user messages.\n\n"
            f"Recent user context (last messages):\n{chat_history}\n\n"
            f"Knowledge Base:\n{knowledge}\n\n"
            f"User question:\n{question}"
        )
        answer = await self.query_llm(prompt, channel)
        await channel.send(answer)

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
            return await ctx.send("No tags stored.")
        content = "\n".join(f"[{_id}] ({tag}) {text}" for _id, tag, text in results)
        await ctx.send(f"```{content[:1990]}{'...' if len(content) > 1990 else ''}```")

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
