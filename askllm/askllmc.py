import discord
from redbot.core import commands, Config
import requests
import json

class LLMManager(commands.Cog):
    """Cog to interact with Ollama LLM and manage knowledge storage."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9876543210)
        self.config.register_global(model="default-llm", context_length=32000, api_url="http://localhost:11434")
        self.config.register_global(knowledge={})
    
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmodel(self, ctx, model: str):
        """Sets the LLM model to be used."""
        await self.config.model.set(model)
        await ctx.send(f"LLM model set to `{model}`.")

    @commands.command()
    async def modellist(self, ctx):
        """Lists available models in Ollama."""
        api_url = await self.config.api_url()
        try:
            response = requests.get(f"{api_url}/api/tags")
            response.raise_for_status()
            models = response.json()
            if not models or "models" not in models:
                raise ValueError("No models found.")
            
            model_names = [m["name"] for m in models["models"]]
            await ctx.send("Available models: " + ", ".join(model_names))
        except Exception as e:
            await ctx.send(f"Error fetching models. Ensure Ollama API URL is correct. Current URL: {api_url}\nError: {str(e)}")
    
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setcontext(self, ctx, length: int):
        """Sets the context length for the LLM."""
        await self.config.context_length.set(length)
        await ctx.send(f"LLM context length set to `{length}`.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setapi(self, ctx, url: str):
        """Sets the API URL for Ollama."""
        cleaned_url = url.rstrip("/")  # Remove trailing slash if present
        await self.config.api_url.set(cleaned_url)
        await ctx.send(f"Ollama API URL set to `{cleaned_url}`. Example format: `http://localhost:11434`")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def addinfo(self, ctx, key: str, *, value: str):
        """Adds information to the knowledge database."""
        async with self.config.knowledge() as knowledge:
            knowledge[key] = value
        await ctx.send(f"Stored information under `{key}`.")

    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """Asks the LLM a question using stored knowledge as reference."""
        knowledge = await self.config.knowledge()
        model = await self.config.model()
        context_length = await self.config.context_length()
        api_url = await self.config.api_url()
        
        knowledge_str = json.dumps(knowledge, indent=2)
        prompt = ("Use the following stored information as reference to answer the question accurately. "
                  "Do not generate speculative responses. If the information is not available, state that you do not know.\n\n"
                  f"Stored Knowledge:\n{knowledge_str}\n\n"
                  f"Question: {question}")
        
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(f"{api_url}/api/generate", json=payload)
            response.raise_for_status()
            answer = response.json().get("response", "Error: No response from model.")
            await ctx.send(answer)
        except Exception as e:
            await ctx.send(f"Error communicating with Ollama. Ensure API URL is correct: {api_url}\nError: {str(e)}")
