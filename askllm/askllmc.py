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

    # MariaDB Helpers
    async def get_db_config(self):
        return await self.config.db_config()

    async def add_tag_content(self, tag, content):
        db_config = await self.get_db_config()
        db = mysql.connector.connect(**db_config)
        cursor = db.cursor()
        cursor.execute("INSERT INTO tags (tag, content) VALUES (%s, %s);", (tag, content))
        db.commit()
        cursor.close()
        db.close()

    async def edit_tag_content(self, entry_id, new_content):
        db_config = await self.get_db_config()
        db = mysql.connector.connect(**db_config)
        cursor = db.cursor()
        cursor.execute("UPDATE tags SET content = %s WHERE id = %s;", (new_content, entry_id))
        db.commit()
        cursor.close()
        db.close()

    async def delete_tag_by_id(self, entry_id):
        db_config = await self.get_db_config()
        db = mysql.connector.connect(**db_config)
        cursor = db.cursor()
        cursor.execute("DELETE FROM tags WHERE id = %s;", (entry_id,))
        db.commit()
        cursor.close()
        db.close()

    async def delete_tag_by_name(self, tag):
        db_config = await self.get_db_config()
        db = mysql.connector.connect(**db_config)
        cursor = db.cursor()
        cursor.execute("DELETE FROM tags WHERE tag = %s;", (tag,))
        db.commit()
        cursor.close()
        db.close()

    async def get_all_content(self):
        db_config = await self.get_db_config()
        db = mysql.connector.connect(**db_config)
        cursor = db.cursor()
        cursor.execute("SELECT id, tag, content FROM tags ORDER BY id;")
        results = cursor.fetchall()
        cursor.close()
        db.close()
        return results

    # Query the LLM
    async def query_llm(self, prompt, channel):
        final_prompt = prompt + "\n\nPlease keep final answer under 2000 characters."
        model = await self.config.model()
        api_url = await self.config.api_url()

        payload = {"model": model, "messages": [{"role": "user", "content": final_prompt}], "stream": False}
        headers = {"Content-Type": "application/json"}

        async with channel.typing():
            resp = requests.post(f"{api_url}/api/chat", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return data.get("message", {}).get("content", "No valid response received.")

    # Commands
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmodel(self, ctx, model: str):
        await self.config.model.set(model)
        await ctx.send(f"LLM model set to '{model}'.")

    @commands.command()
    async def modellist(self, ctx):
        api_url = await self.config.api_url()
        try:
            r = requests.get(f"{api_url}/api/tags")
            r.raise_for_status()
            data = r.json()
            models = [m["name"] for m in data.get("models", [])]
            await ctx.send(f"Available models: {', '.join(models)}")
        except Exception as e:
            await ctx.send(f"Error: {e}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setapi(self, ctx, url: str):
        url = url.rstrip("/")
        await self.config.api_url.set(url)
        await ctx.send(f"Ollama API URL set to '{url}'.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setdb(self, ctx, host: str, user: str, password: str, database: str):
        await self.config.db_config.set({"host": host, "user": user, "password": password, "database": database})
        await ctx.send("Database configuration updated.")

    @commands.command()
    async def askllm(self, ctx, *, question: str):
        await self.process_question(question, ctx.channel)

    async def process_question(self, question, channel):
        entries = await self.get_all_content()
        words = re.sub(r"[^\w\s]", "", question.lower()).split()

        filtered = []
        for _id, tag, content in entries:
            if any(w in content.lower() for w in words):
                filtered.append((tag, content))

        if not filtered:
            await channel.send("No relevant info found.")
            return

        context = "\n".join(f"[{tag}] {content}" for tag, content in filtered)
        prompt = f"Using this knowledge base:\n{context}\nPlease answer the question as clearly and accurately as possible, correcting for typos or vague language if needed:\n{question}"
        answer = await self.query_llm(prompt, channel)
        await channel.send(answer)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return
        if self.bot.user.mentioned_in(message):
            question = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
            if question:
                await self.process_question(question, message.channel)

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, tag: str, amount: int = 20):
        msgs = [msg async for msg in ctx.channel.history(limit=amount+5) if not msg.author.bot][:amount]
        conversation = "\n".join(f"{m.author.name}: {m.content}" for m in reversed(msgs))

        prompt = f"Summarize or produce new helpful info for '{tag}':\n{conversation}"
        suggestion = await self.query_llm(prompt, ctx.channel)

        await ctx.send(suggestion)
        await ctx.send("Type 'yes' to confirm, 'edit [instructions]' to refine, or 'stop' to cancel.")

        def check(m):
            return m.author == ctx.author and m.channel == ctx.channel

        while True:
            reply = await self.bot.wait_for("message", check=check, timeout=180)
            text = reply.content.strip()
            lower = text.lower()

            if lower == "yes":
                await self.add_tag_content(tag.lower(), suggestion)
                await ctx.send(f"Stored new info under '{tag.lower()}'.")
                break

            elif lower.startswith("edit"):
                instructions = text[4:].strip()
                refine_prompt = f"{prompt}\n\nUser additional instructions: {instructions}"
                suggestion = await self.query_llm(refine_prompt, ctx.channel)
                await ctx.send(suggestion)
                await ctx.send("Type 'yes' to confirm, 'edit [instructions]' to refine, or 'stop' to cancel.")

            elif lower == "stop":
                await ctx.send("Learning canceled.")
                break

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

        lines = [f"[{_id}] ({tag}) {text}" for _id, tag, text in results]
        content = "\n".join(lines)
        if len(content) > 1900:
            content = content[:1900] + "..."
        await ctx.send(f"```{content}```")

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
