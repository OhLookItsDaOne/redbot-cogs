import discord
from redbot.core import commands, Config
import requests
import json
import time

class LLMManager(commands.Cog):
    """Cog to interact with Ollama LLM and manage knowledge storage."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9876543210)
        self.config.register_global(model="llama3.2", api_url="http://localhost:11434")
        self.config.register_global(knowledge={})

    async def _get_api_url(self):
        return await self.config.api_url()

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
        try:
            response = requests.get(f"{api_url}/api/tags")
            response.raise_for_status()
            models = response.json()
            model_names = [m["name"] for m in models.get("models", [])]
            await ctx.send(f"Available models: {', '.join(model_names)}")
        except Exception as e:
            await ctx.send(f"Error fetching models: {e}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setapi(self, ctx, url: str):
        """Sets the API URL for Ollama."""
        url = url.rstrip("/")
        await self.config.api_url.set(url)
        await ctx.send(f"Ollama API URL set to `{url}`")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def addinfo(self, ctx, key: str, *, value: str):
        """Adds information to the knowledge database."""
        async with self.config.knowledge() as knowledge:
            knowledge[key] = value
        await ctx.send(f"Information stored under `{key}`.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def loadmodel(self, ctx, model: str):
        """Pulls a model and ensures it's ready to use."""
        api_url = await self._get_api_url()
        try:
            await ctx.send(f"Pulling model `{model}`...")
            requests.post(f"{api_url}/api/pull", json={"name": model}).raise_for_status()

            for _ in range(30):
                response = requests.get(f"{api_url}/api/tags")
                response.raise_for_status()
                models = [m["name"] for m in response.json().get("models", [])]
                if model in models:
                    await ctx.send(f"Model `{model}` is now available.")
                    return
                time.sleep(2)

            await ctx.send(f"Model `{model}` is not yet available. Try again later.")
        except Exception as e:
            await ctx.send(f"Error loading model `{model}`: {e}")

    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """Sends a message to the LLM and returns its response."""
        knowledge = await self.config.knowledge()
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

        try:
            response = requests.post(f"{api_url}/api/chat", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            answer = data.get("message", {}).get("content", "No valid response received.")
            await ctx.send(answer)

        except Exception as e:
            await ctx.send(f"Error communicating with Ollama: {e}")
