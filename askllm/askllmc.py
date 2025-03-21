import discord
from redbot.core import commands, Config
import requests
import json
import time
import re
from redbot.core.data_manager import cog_data_path

# Helper function for splitting messages into safe chunks
def chunkify(text, max_size=1900):
    """Splits a string into a list of chunks that fit under max_size characters."""
    lines = text.split("\n")
    current_chunk = ""
    chunks = []

    for line in lines:
        # If adding this line would exceed the max_size, we start a new chunk
        if len(current_chunk) + len(line) + 1 > max_size:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            if current_chunk:
                current_chunk += "\n" + line
            else:
                current_chunk = line

    # If there's leftover content in current_chunk, push it into the list
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

class LLMManager(commands.Cog):
    """Cog to interact with Ollama LLM and manage knowledge storage."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9876543210)
        self.config.register_global(model="llama3.2", api_url="http://localhost:11434")
        # Main knowledge file
        self.knowledge_file = cog_data_path(self) / "llm_knowledge.json"
        self.ensure_knowledge_file(self.knowledge_file)

    def ensure_knowledge_file(self, path):
        if not path.exists():
            with path.open("w") as file:
                json.dump({}, file)

    def load_knowledge(self):
        with self.knowledge_file.open("r") as file:
            return json.load(file)

    def save_knowledge(self, knowledge):
        with self.knowledge_file.open("w") as file:
            json.dump(knowledge, file, indent=4)

    async def _get_api_url(self):
        return await self.config.api_url()

    async def get_llm_response(self, question: str, knowledge_enabled: bool = True):
        """
        If knowledge_enabled=True, we embed the existing knowledge as context.
        Otherwise we just pass the user's question.
        """
        model = await self.config.model()
        api_url = await self._get_api_url()

        if knowledge_enabled:
            # Use existing knowledge from JSON
            knowledge = self.load_knowledge()
            context = (
                "Use the provided knowledge to answer accurately. Do not guess.\n\n"
                f"Knowledge:\n{json.dumps(knowledge)}\n\n"
            )
        else:
            # No existing knowledge
            context = (
                "Extract or summarize the conversation below as valid JSON if possible.\n"
                "Each top-level key in the JSON should represent a tag in the knowledge.\n\n"
            )

        prompt = context + f"Question: {question}"

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
    async def on_message(self, message: discord.Message):
        """
        If the bot is mentioned in a message, respond with an LLM-generated answer
        (including existing knowledge).
        """
        if message.author.bot or not message.guild:
            return

        if self.bot.user.mentioned_in(message):
            question = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
            if question:
                try:
                    async with message.channel.typing():
                        answer = await self.get_llm_response(question, knowledge_enabled=True)
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
            answer = await self.get_llm_response(question, knowledge_enabled=True)
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
        sorted by tag with indices. If it's too large, splits into multiple messages.
        """
        knowledge = self.load_knowledge()
        formatted_knowledge = "\n".join(
            f"{tag}:\n" + "\n".join(f"  [{i}] {info}" for i, info in enumerate(infos))
            for tag, infos in sorted(knowledge.items())
        )

        chunks = chunkify(formatted_knowledge, max_size=1900)
        for idx, chunk in enumerate(chunks, 1):
            await ctx.send(f"LLM Knowledge Base (Part {idx}/{len(chunks)}):\n```\n{chunk}\n```")

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
    #                    "Learn" Command (Separate from knowledge)
    # -----------------------------------------------------------------------------------

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, amount: int = 20):
        """
        Gathers last [amount] messages, ignoring bot msgs & this command,
        calls get_llm_response with knowledge_enabled=False,
        prompting the LLM to produce JSON or text,
        asks for admin confirmation. If yes => parse/store.
        """
        # gather messages except the command that triggered this
        def not_command_or_bot(m: discord.Message):
            if m.author.bot:
                return False
            if m.id == ctx.message.id:
                return False
            return True

        messages = []
        async for msg in ctx.channel.history(limit=amount+5):
            if not_command_or_bot(msg):
                messages.append(msg)
            if len(messages) >= amount:
                break

        messages.reverse()
        conversation = "\n".join(f"{m.author.name}: {m.content}" for m in messages)

        # we do a special LLM call with knowledge_enabled=False
        async with ctx.typing():
            suggestion = await self.get_llm_response(conversation, knowledge_enabled=False)

        await ctx.send(
            f"Suggested info to add:\n```\n{suggestion}\n```\n"
            "Type `yes` to confirm, `no [instructions]` to refine, or `stop` to cancel."
        )

        def check(msg: discord.Message):
            return msg.author == ctx.author and msg.channel == ctx.channel

        while True:
            try:
                response = await self.bot.wait_for("message", check=check, timeout=120)
                text = response.content.strip()
                lower = text.lower()

                if lower == "yes":
                    msg = self.store_learned_suggestion(suggestion)
                    await ctx.send(msg)
                    break

                elif lower == "stop":
                    await ctx.send("Learning process cancelled.")
                    break

                elif lower.startswith("no"):
                    instructions = text[2:].strip()
                    # refine the prompt
                    refined_prompt = (f"{conversation}\n\nUser clarifications:\n" + instructions)
                    async with ctx.typing():
                        suggestion = await self.get_llm_response(refined_prompt, knowledge_enabled=False)
                    await ctx.send(
                        f"Updated suggestion:\n```\n{suggestion}\n```\n"
                        "Type `yes` to confirm, `no [something else]` to refine again, or `stop` to cancel."
                    )
                else:
                    await ctx.send("Please type `yes`, `no [instructions]`, or `stop`.")
            except Exception as e:
                await ctx.send(f"Error or timeout: {e}")
                break

    def store_learned_suggestion(self, suggestion: str) -> str:
        """
        Attempt to parse a code block with triple-backticks first.
        If parse fails, parse entire string as JSON. If that fails, store in 'General'.
        """
        # attempt triple-backtick extraction
        pattern = r"(?s)```json\s*(\{.*?\})\s*```"
        match = re.search(pattern, suggestion)
        if match:
            raw_json = match.group(1)
            try:
                parsed = json.loads(raw_json)
                return self.apply_learned_json(parsed)
            except (json.JSONDecodeError, TypeError):
                pass

        # if no code block found or parse failed, try entire string
        try:
            parsed_entire = json.loads(suggestion)
            if isinstance(parsed_entire, dict):
                return self.apply_learned_json(parsed_entire)
        except (json.JSONDecodeError, TypeError):
            pass

        # fallback => store entire suggestion in "General"
        knowledge = self.load_knowledge()
        knowledge.setdefault("General", []).append(suggestion)
        self.save_knowledge(knowledge)
        return "No valid JSON found; suggestion stored in 'General'."

    def apply_learned_json(self, parsed_dict: dict) -> str:
        """
        If the JSON is a dictionary: each top-level key is a tag, value appended as item(s).
        """
        knowledge = self.load_knowledge()
        for tag, data in parsed_dict.items():
            if isinstance(data, list):
                for item in data:
                    knowledge.setdefault(tag, []).append(item)
            else:
                knowledge.setdefault(tag, []).append(data)
        self.save_knowledge(knowledge)
        return "Valid JSON found; top-level keys stored as tags!"
