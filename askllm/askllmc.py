import sys
import subprocess

# Dynamische Installation der benötigten Pakete
required_packages = ['qdrant-client', 'sentence-transformers']

def install_missing_packages(packages):
    try:
        import pkg_resources  # Zum Überprüfen installierter Pakete
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools"])
        import pkg_resources

    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = [pkg for pkg in packages if pkg.lower() not in installed]
    
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    else:
        print("All required packages are already installed.")

install_missing_packages(required_packages)

# Nun die Imports, nachdem wir sichergestellt haben, dass die Pakete vorhanden sind:
import discord
from redbot.core import commands, Config
import requests
import json
import time
import os
from redbot.core.data_manager import cog_data_path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

class LLMManager(commands.Cog):
    """Cog to interact with Ollama LLM and manage knowledge storage."""
    
    def __init__(self, bot):
        self.bot = bot
        # Hier wird das Standardmodell (gemma3:4b) gesetzt und die URLs konfiguriert.
        self.config = Config.get_conf(self, identifier=9876543210)
        self.config.register_global(
            model="gemma3:4b",  # Standardmodell auf gemma3:4b
            api_url="http://192.168.10.5:11434",  # Ollama API URL
            chroma_url="http://192.168.10.5:6333"   # Qdrant URL, passe ggf. an deine Umgebung an
        )
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

    async def qdrant_query(self, query_text, top_k=3):
        chroma_url = await self.config.chroma_url()
        # Hier laden wir das Modell; für Performance könntest du das einmal cachen.
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode(query_text).tolist()
        client = QdrantClient(url=chroma_url)
        results = client.search(
            collection_name="fuswiki",
            query_vector=embedding,
            limit=top_k
        )
        return "\n\n".join([hit.payload["text"] for hit in results])

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
    async def setchroma(self, ctx, url: str):
        """Sets Qdrant API URL."""
        await self.config.chroma_url.set(url.rstrip("/"))
        await ctx.send(f"Qdrant API URL set to `{url}`")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setapi(self, ctx, url: str):
        """Sets the API URL for Ollama."""
        url = url.rstrip("/")
        await self.config.api_url.set(url)
        await ctx.send(f"Ollama API URL set to `{url}`")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx, tag: str, *, info: str):
        """Adds information under a tag to the LLM's knowledge base."""
        knowledge = self.load_knowledge()
        if tag not in knowledge:
            knowledge[tag] = []
        knowledge[tag].append(info)
        self.save_knowledge(knowledge)
        await ctx.send(f"Information stored under tag `{tag}` in LLM knowledge base.")

    @commands.command()
    async def llmknowshow(self, ctx):
        """Displays the current knowledge stored in the LLM's knowledge base."""
        knowledge = self.load_knowledge()
        formatted_knowledge = json.dumps(knowledge, indent=4)
        await ctx.send(f"LLM Knowledge Base:\n```json\n{formatted_knowledge}\n```")

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
        """Sends question to LLM using Qdrant embeddings."""
        model = await self.config.model()
        api_url = await self._get_api_url()
        context_info = await self.qdrant_query(question, top_k=3)
    
        prompt = (
            "Use the provided context to answer accurately. Do not guess.\n\n"
            f"Context:\n{context_info}\n\n"
            f"Question: {question}"
        )
    
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
    
        try:
            response = requests.post(f"{api_url}/api/chat", json=payload)
            response.raise_for_status()
            answer = response.json().get("message", {}).get("content", "No valid response received.")
            await ctx.send(answer)
        except Exception as e:
            await ctx.send(f"Error: {e}")

def setup(bot):
    bot.add_cog(LLMManager(bot))
