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
        """
        Sends a prompt (including the knowledge base) to the LLM.
        Returns the LLM's textual response.
        """
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
        """
        If the bot is mentioned in a message, respond with an LLM-generated answer.
        """
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

    # -----------------------------------------------------------------------------------
    #                    Basic commands for LLM / Model management
    # -----------------------------------------------------------------------------------

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmodel(self, ctx, model: str):
        """
        Sets the default LLM model to be used.
        """
        await self.config.model.set(model)
        await ctx.send(f"Default LLM model set to `{model}`.")

    @commands.command()
    async def modellist(self, ctx):
        """
        Lists available models in Ollama.
        """
        api_url = await self._get_api_url()
        response = requests.get(f"{api_url}/api/tags")
        response.raise_for_status()
        models = response.json()
        model_names = [m["name"] for m in models.get("models", [])]
        await ctx.send(f"Available models: {', '.join(model_names)}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setapi(self, ctx, url: str):
        """
        Sets the API URL for Ollama.
        """
        url = url.rstrip("/")
        await self.config.api_url.set(url)
        await ctx.send(f"Ollama API URL set to `{url}`")

    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """
        Sends a message to the LLM and returns its response using stored knowledge.
        """
        async with ctx.typing():
            answer = await self.get_llm_response(question)
        await ctx.send(answer)

    # -----------------------------------------------------------------------------------
    #                    Knowledge Base Commands (Tags, Indices)
    # -----------------------------------------------------------------------------------

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx, tag: str, *, info: str):
        """
        Adds information under a tag to the LLM's knowledge base.
        """
        knowledge = self.load_knowledge()
        knowledge.setdefault(tag, []).append(info)
        self.save_knowledge(knowledge)
        await ctx.send(f"Information stored under tag `{tag}`.")

    @commands.command()
    async def llmknowshow(self, ctx):
        """
        Displays the current knowledge stored in the LLM's knowledge base,
        sorted by tag with indices.
        """
        knowledge = self.load_knowledge()
        formatted_knowledge = "\n".join(
            f"{tag}:\n" + "\n".join(f"  [{i}] {info}" for i, info in enumerate(infos))
            for tag, infos in sorted(knowledge.items())
        )
        await ctx.send(f"LLM Knowledge Base:\n```\n{formatted_knowledge}\n```")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdelete(self, ctx, tag: str, index: int):
        """
        Deletes information under a tag by index.
        """
        knowledge = self.load_knowledge()
        if tag in knowledge and 0 <= index < len(knowledge[tag]):
            deleted = knowledge[tag].pop(index)
            if not knowledge[tag]:
                del knowledge[tag]
            self.save_knowledge(knowledge)
            await ctx.send(f"Deleted info `{deleted}` from tag `{tag}`.")
        else:
            await ctx.send("Tag or index invalid.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx, tag: str):
        """
        Deletes an entire tag and its associated information.
        """
        knowledge = self.load_knowledge()
        if tag in knowledge:
            del knowledge[tag]
            self.save_knowledge(knowledge)
            await ctx.send(f"Deleted entire tag `{tag}`.")
        else:
            await ctx.send("Tag not found.")

    # -----------------------------------------------------------------------------------
    #                    "Learn" Command (Admin Confirmation)
    # -----------------------------------------------------------------------------------

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, amount: int = 20):
        """
        Learns from recent messages and updates knowledge base upon confirmation.
        Pulls the last [amount] messages from the channel (excluding bot messages),
        asks the LLM to extract relevant info, then asks for admin confirmation.
        """
        # Gather recent messages
        messages = [msg async for msg in ctx.channel.history(limit=amount+1) if msg.author != self.bot.user]
        messages.reverse()
        content = "\n".join(f"{m.author.name}: {m.content}" for m in messages)

        # Prompt for LLM to suggest knowledge
        prompt = (
            "Extract useful and relevant information from the conversation "
            "for storing as knowledge:\n\n"
            f"{content}"
        )

        # Get the suggestion from LLM
        async with ctx.typing():
            suggested_info = await self.get_llm_response(prompt)

        # Present suggestion to admin
        await ctx.send(
            f"Suggested info to add:\n```\n{suggested_info}\n```\n"
            "Type `yes` to confirm, `no` to retry, or `stop` to cancel."
        )

        def check(m):
            return (
                m.author == ctx.author
                and m.channel == ctx.channel
                and m.content.lower() in {"yes", "no", "stop"}
            )

        while True:
            try:
                # Wait for admin decision
                response = await self.bot.wait_for("message", check=check, timeout=120)
                choice = response.content.lower()

                if choice == "yes":
                    # Save info to "General" tag (or any tag you prefer)
                    knowledge = self.load_knowledge()
                    knowledge.setdefault("General", []).append(suggested_info)
                    self.save_knowledge(knowledge)
                    await ctx.send("Information successfully saved.")
                    break

                elif choice == "no":
                    # Retry extracting new suggestion
                    async with ctx.typing():
                        suggested_info = await self.get_llm_response(prompt)
                    await ctx.send(
                        f"Updated suggestion:\n```\n{suggested_info}\n```\n"
                        "Type `yes` to confirm, `no` to retry, or `stop` to cancel."
                    )

                else:  # stop
                    await ctx.send("Learning process cancelled.")
                    break

            except Exception as e:
                await ctx.send(f"Error or timeout: {e}")
                break
