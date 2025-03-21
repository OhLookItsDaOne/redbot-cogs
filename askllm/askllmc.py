import discord
from redbot.core import commands, Config
import requests
import json
import time
from redbot.core.data_manager import cog_data_path

class LLMManager(commands.Cog):
    """Cog to interact with Ollama LLM and manage knowledge storage."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9876543210)
        self.config.register_global(model="llama3.2", api_url="http://localhost:11434")
        self.knowledge_file = cog_data_path(self) / "llm_knowledge.json"
        self.ensure_knowledge_file()

    def ensure_knowledge_file(self):
        if not self.knowledge_file.exists():
            with self.knowledge_file.open('w') as file:
                json.dump({}, file)

    def load_knowledge(self):
        with self.knowledge_file.open('r') as file:
            return json.load(file)

    def save_knowledge(self, knowledge):
        with self.knowledge_file.open('w') as file:
            json.dump(knowledge, file, indent=4)

    async def _get_api_url(self):
        return await self.config.api_url()

    async def get_llm_response(self, question: str):
        knowledge = self.load_knowledge()
        model = await self.config.model()
        api_url = await self._get_api_url()

        prompt = (
            "Use the provided knowledge to answer accurately. Do not guess.\n\n"
            f"Knowledge:\n{json.dumps(knowledge)}\n\n"
            f"Question: {question}"
        )

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        headers = {"Content-Type": "application/json"}

        response = requests.post(f"{api_url}/api/chat", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "No valid response received.")

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author.bot or not message.guild:
            return

        if self.bot.user.mentioned_in(message):
            question = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
            if question:
                try:
                    async with message.channel.typing():
                        answer = await self.get_llm_response(question)
                    await message.channel.send(answer)
                except Exception as e:
                    await message.channel.send(f"Error: {e}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmodel(self, ctx, model: str):
        """Sets the default LLM model to be used."""
        await self.config.model.set(model)
        await ctx.send(f"Default LLM model set to `{model}`.")

    @commands.command()
    async def modellist(self, ctx):
        """Lists available models in Ollama."""
        api_url = await self._get_api_url()
        response = requests.get(f"{api_url}/api/tags")
        response.raise_for_status()
        models = response.json()
        model_names = [m["name"] for m in models.get("models", [])]
        await ctx.send(f"Available models: {', '.join(model_names)}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setapi(self, ctx, url: str):
        """Sets the API URL for Ollama."""
        url = url.rstrip("/")
        await self.config.api_url.set(url)
        await ctx.send(f"Ollama API URL set to `{url}`")

    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """Sends a message to the LLM and returns its response using stored knowledge."""
        async with ctx.typing():
            answer = await self.get_llm_response(question)
        await ctx.send(answer)

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, amount: int = 20):
        """Learns from recent messages and updates knowledge base upon confirmation."""
        messages = [message async for message in ctx.channel.history(limit=amount+1) if message.author != self.bot.user]
        messages.reverse()
        content = "\n".join(f"{msg.author.name}: {msg.content}" for msg in messages)

        prompt = f"Extract useful and relevant information from the conversation for storing as knowledge:\n\n{content}"        
        async with ctx.typing():
            suggested_info = await self.get_llm_response(prompt)

        await ctx.send(f"Suggested info to add:\n```\n{suggested_info}\n```\nType `yes` to confirm, `no` to retry, or `stop` to cancel.")

        def check(m):
            return m.author == ctx.author and m.channel == ctx.channel and m.content.lower() in {"yes", "no", "stop"}

        while True:
            try:
                response = await self.bot.wait_for("message", check=check, timeout=120)
                if response.content.lower() == "yes":
                    knowledge = self.load_knowledge()
                    knowledge.setdefault("General", []).append(suggested_info)
                    self.save_knowledge(knowledge)
                    await ctx.send("Information successfully saved.")
                    break
                elif response.content.lower() == "no":
                    suggested_info = await self.get_llm_response(prompt)
                    await ctx.send(f"Updated suggestion:\n```\n{suggested_info}\n```\nType `yes` to confirm, `no` to retry, or `stop` to cancel.")
                else:
                    await ctx.send("Learning process cancelled.")
                    break
            except Exception as e:
                await ctx.send(f"Error or timeout: {e}")
                break
