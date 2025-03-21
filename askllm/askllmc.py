import discord
from redbot.core import commands, Config
import requests
import json
import time
import re
from redbot.core.data_manager import cog_data_path

####################################
# HELPER: Chunkify for large outputs
####################################
def chunkify(text, max_size=1900):
    """Splits a string into chunks that each fit under max_size characters."""
    lines = text.split("\n")
    current_chunk = ""
    chunks = []

    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_size:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            if current_chunk:
                current_chunk += "\n" + line
            else:
                current_chunk = line

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
        If knowledge_enabled=True, embed existing knowledge as context.
        Otherwise pass only the user's question.
        """
        model = await self.config.model()
        api_url = await self._get_api_url()

        if knowledge_enabled:
            # Mit existierendem Wissen
            knowledge = self.load_knowledge()
            context = (
                "Use the provided knowledge to answer accurately. Do not guess.\n\n"
                f"Knowledge:\n{json.dumps(knowledge)}\n\n"
            )
        else:
            # Ohne vorhandenes Wissen, nur Lösungen extrahieren
            context = (
                "You must extract only the SOLUTIONS (fixes, steps, or final answers) "
                "from the conversation below. No user mentions, no logs, no problem statements. "
                "Output them as valid JSON with multiple top-level keys if needed. Each key is a relevant tag, "
                "the value is the 'solution' or 'fix'. Keep it short and direct.\n\n"
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
        If the bot is mentioned, respond with an LLM-generated answer
        that includes old knowledge.
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

    # ----------------------------------------------------------------
    #     Basic LLM / Model mgmt
    # ----------------------------------------------------------------

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmodel(self, ctx, model: str):
        """Sets the default LLM model."""
        await self.config.model.set(model)
        await ctx.send(f"Default LLM model set to `{model}`.")

    @commands.command()
    async def modellist(self, ctx):
        """Lists available models."""
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
        """Sends a message to the LLM (with old knowledge)."""
        async with ctx.typing():
            answer = await self.get_llm_response(question, knowledge_enabled=True)
        await ctx.send(answer)

    # ----------------------------------------------------------------
    #   Knowledge Base Commands
    # ----------------------------------------------------------------

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx, tag: str, *, info: str):
        """Adds info under a tag."""
        knowledge = self.load_knowledge()
        knowledge.setdefault(tag, []).append(info)
        self.save_knowledge(knowledge)
        await ctx.send(f"Information stored under tag `{tag}`.")

    @commands.command()
    async def llmknowshow(self, ctx):
        """Shows the knowledge base, chunked by 1900 chars."""
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
        """Deletes info by index."""
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
        """Deletes an entire tag."""
        knowledge = self.load_knowledge()
        if tag in knowledge:
            del knowledge[tag]
            self.save_knowledge(knowledge)
            await ctx.send(f"Deleted entire tag `{tag}`.")
        else:
            await ctx.send("Tag not found.")

    # ----------------------------------------------------------------
    #          "Learn" Command: Only Solutions, ignoring old knowledge
    # ----------------------------------------------------------------

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, amount: int = 20):
        """
        Reads last [amount] msgs, ignoring bot msgs & this command.
        Asks LLM (no old knowledge) to produce only solutions as JSON,
        with multiple top-level keys if needed, each key = a solution's tag,
        each value = solution/fix. If invalid => stored in 'General'.
        """
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

        async with ctx.typing():
            suggestion = await self.get_llm_response(conversation, knowledge_enabled=False)

        await ctx.send(
            f"Suggested solutions to add:\n```\n{suggestion}\n```\n"
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
                    result = self.store_learned_suggestion(suggestion)
                    await ctx.send(result)
                    break
                elif lower == "stop":
                    await ctx.send("Learning process cancelled.")
                    break
                elif lower.startswith("no"):
                    instructions = text[2:].strip()
                    refined_prompt = (
                        "You must ONLY output solutions/fixes as valid JSON with multiple top-level keys if needed. "
                        "No user mentions, no logs, no problems, only final steps to fix or solve. \n\n"
                        f"{conversation}\n\nUser clarifications:\n{instructions}"
                    )

                    async with ctx.typing():
                        suggestion = await self.get_llm_response(refined_prompt, knowledge_enabled=False)

                    await ctx.send(
                        f"Updated solutions suggestion:\n```\n{suggestion}\n```\n"
                        "Type `yes` to confirm, `no [something else]` to refine again, or `stop` to cancel."
                    )
                else:
                    await ctx.send("Please type `yes`, `no [instructions]`, or `stop`.")
            except Exception as e:
                await ctx.send(f"Error or timeout: {e}")
                break

    def store_learned_suggestion(self, suggestion: str) -> str:
        """
        Try to parse triple-backtick code block. If parse fails, parse entire string.
        If valid JSON => each top-level key = tag, each value appended. Otherwise => 'General'.
        """
        pattern = r"(?s)```json\s*(\{.*?\})\s*```"
        match = re.search(pattern, suggestion)
        if match:
            raw_json = match.group(1)
            try:
                parsed = json.loads(raw_json)
                return self.apply_learned_json(parsed)
            except (json.JSONDecodeError, TypeError):
                pass

        # fallback: parse entire text
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
        return "Stored as tags (solutions) from JSON!"
