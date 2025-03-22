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

    async def search_content(self, query, limit=10):
        db_config = await self.get_db_config()
        db = mysql.connector.connect(**db_config)
        cursor = db.cursor()
        cursor.execute(
            "SELECT tag, content FROM tags WHERE MATCH(content) AGAINST(%s IN NATURAL LANGUAGE MODE) LIMIT %s;",
            (query, limit)
        )
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
        results = await self.search_content(question)
        if not results:
            await ctx.send("No relevant info found.")
            return

        context = "\n".join(f"[{tag}] {content}" for tag, content in results)
        prompt = f"Using this context:\n{context}\nAnswer the question: {question}"
        answer = await self.query_llm(prompt, ctx.channel)
        await ctx.send(answer)

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, tag: str, amount: int = 20):
        msgs = [msg async for msg in ctx.channel.history(limit=amount+5) if not msg.author.bot][:amount]
        conversation = "\n".join(f"{m.author.name}: {m.content}" for m in reversed(msgs))

        prompt = f"Summarize or produce new helpful info for '{tag}':\n{conversation}"
        suggestion = await self.query_llm(prompt, ctx.channel)

        await ctx.send(suggestion)
        await ctx.send("Type 'yes' to confirm, 'stop' to cancel.")

        def check(m):
            return m.author == ctx.author and m.channel == ctx.channel

        reply = await self.bot.wait_for("message", check=check, timeout=180)
        if reply.content.lower() == "yes":
            await self.add_tag_content(tag.lower(), suggestion)
            await ctx.send(f"Stored new info under '{tag.lower()}'.")
        else:
            await ctx.send("Learning canceled.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx, tag: str, *, info: str):
        await self.add_tag_content(tag.lower(), info)
        await ctx.send(f"Added info under '{tag.lower()}'.")
