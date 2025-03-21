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
    """Splits a string into a list of chunks that each fit under max_size characters."""
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

    #
    #  Add an entry to a tag only if it doesn't already exist. Returns True if added, False if duplicate.
    #
    def add_entry_to_tag(self, knowledge: dict, tag: str, entry) -> bool:
        if tag not in knowledge:
            knowledge[tag] = []
        if entry in knowledge[tag]:  # simple direct-compare for duplicates
            return False
        knowledge[tag].append(entry)
        return True

    async def _get_api_url(self):
        return await self.config.api_url()

    async def get_llm_response(self, prompt: str):
        """
        We call this to send a custom prompt to the LLM
        without automatically including old knowledge in the context.
        """
        model = await self.config.model()
        api_url = await self._get_api_url()

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

    #
    #  on_message: if the bot is mentioned, reply with old knowledge
    #
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return
        if self.bot.user.mentioned_in(message):
            question = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
            if question:
                try:
                    async with message.channel.typing():
                        # build a prompt that includes all knowledge
                        # (like the old logic if you want)
                        # or if you already have a function for that, reuse it
                        knowledge = self.load_knowledge()
                        api_url = await self._get_api_url()
                        model = await self.config.model()

                        context = (
                            "Use the following knowledge to answer accurately. "
                            "Do not guess.\n\n"
                            f"Knowledge:\n{json.dumps(knowledge)}\n\n"
                            f"Question: {question}"
                        )
                        payload = {
                            "model": model,
                            "messages": [{"role": "user", "content": context}],
                            "stream": False
                        }
                        headers = {"Content-Type": "application/json"}
                        r = requests.post(f"{api_url}/api/chat", json=payload, headers=headers)
                        r.raise_for_status()
                        data = r.json()
                        answer = data.get("message", {}).get("content", "No valid response received.")
                    await message.channel.send(answer)
                except Exception as e:
                    await message.channel.send(f"Error: {e}")

    ##################################################
    # Basic LLM / Model mgmt
    ##################################################

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

    #
    #  askllm uses the entire knowledge
    #
    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """Sends a question to the LLM using entire knowledge in context."""
        knowledge = self.load_knowledge()
        model = await self.config.model()
        api_url = await self._get_api_url()

        prompt = (
            "Use the provided knowledge to answer accurately. Do not guess.\n\n"
            f"Knowledge:\n{json.dumps(knowledge)}\n\n"
            f"Question: {question}"
        )
        async with ctx.typing():
            r = requests.post(
                f"{api_url}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                },
                headers={"Content-Type": "application/json"}
            )
            r.raise_for_status()
            data = r.json()
            answer = data.get("message", {}).get("content", "No valid response received.")
        await ctx.send(answer)

    ##################################################
    #    Knowledge Base Commands
    ##################################################

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx, tag: str, *, info: str):
        """Adds info under a tag, skipping duplicates."""
        knowledge = self.load_knowledge()
        added = self.add_entry_to_tag(knowledge, tag, info)
        self.save_knowledge(knowledge)
        if added:
            await ctx.send(f"Information stored under tag `{tag}`.")
        else:
            await ctx.send(f"This info already exists under tag `{tag}` and was not re-added.")

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

    ###################################################
    #    New Learn Command: !learn <tag> <message_count>
    ###################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, tag: str, amount: int = 20):
        """
        Gathers last [amount] messages (excluding bots & this command).
        The LLM sees only that conversation and the existing knowledge for <tag>.
        LLM should output JSON with new solutions. Then admin decides yes/no/stop.
        """
        # 1) Gather messages
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

        # 2) Grab existing knowledge for this tag
        knowledge = self.load_knowledge()
        existing_entries = knowledge.get(tag, [])

        # 3) Build specialized prompt
        # LLM sees only the existing entries for that tag (if any),
        # and the conversation. Output new solutions as valid JSON
        base_prompt = (
            f"Existing solutions for tag '{tag}':\n"
            f"{json.dumps(existing_entries, indent=2)}\n\n"
            "Below is a conversation. Extract only new or refined solutions for that tag.\n"
            "Output them as valid JSON. Example structure:\n"
            "{\n"
            "  \"solutions\": [\n"
            "    \"some new fix...\"," 
            "    \"some other fix...\"\n"
            "  ]\n"
            "}\n\n"
            "No user mentions, no logs. Only final steps to fix.\n\n"
            f"Conversation:\n{conversation}"
        )

        async with ctx.typing():
            suggestion = await self.get_llm_response(base_prompt)

        # Present to admin
        await ctx.send(
            f"Suggested new solutions for tag '{tag}':\n```\n{suggestion}\n```\n"
            "Type `yes` to confirm, `no [instructions]` to refine, or `stop` to cancel."
        )

        def check(msg: discord.Message):
            return msg.author == ctx.author and msg.channel == ctx.channel

        # 4) Wait for admin interaction
        while True:
            try:
                response = await self.bot.wait_for("message", check=check, timeout=120)
                text = response.content.strip().lower()

                if text == "yes":
                    # parse JSON, store in that tag
                    ret = self.store_learn_solutions(tag, suggestion)
                    await ctx.send(ret)
                    break
                elif text == "stop":
                    await ctx.send("Learning process cancelled.")
                    break
                elif text.startswith("no"):
                    # refine
                    instructions = response.content[2:].strip()
                    new_prompt = base_prompt + f"\n\nAdditional instructions:\n{instructions}"
                    async with ctx.typing():
                        suggestion = await self.get_llm_response(new_prompt)
                    await ctx.send(
                        f"Updated suggestion for tag '{tag}':\n```\n{suggestion}\n```\n"
                        "Type `yes` to confirm, `no [something else]` to refine again, or `stop` to cancel."
                    )
                else:
                    await ctx.send("Please type `yes`, `no [instructions]`, or `stop`.")
            except Exception as e:
                await ctx.send(f"Error or timeout: {e}")
                break

    def store_learn_solutions(self, tag: str, suggestion: str) -> str:
        """
        The LLM is expected to output something like:
        ```json
        {
          "solutions": [
            "First fix",
            "Second fix"
          ]
        }
        ```
        We'll parse that and store each entry in 'tag', skipping duplicates.
        If parse fails, store entire suggestion in 'General'.
        """
        # attempt code block parse
        pattern = r"(?s)```json\s*(\{.*?\})\s*```"
        match = re.search(pattern, suggestion)
        raw = suggestion
        if match:
            raw = match.group(1)

        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Parsed data not a dict")

            solutions = data.get("solutions", [])
            if not isinstance(solutions, list):
                raise ValueError("No valid 'solutions' array")

            knowledge = self.load_knowledge()
            changed = 0
            if tag not in knowledge:
                knowledge[tag] = []

            for fix in solutions:
                if fix not in knowledge[tag]:
                    knowledge[tag].append(fix)
                    changed += 1

            self.save_knowledge(knowledge)
            return f"Added {changed} new solution(s) to tag '{tag}'."

        except Exception:
            # fallback => store entire suggestion in "General"
            knowledge = self.load_knowledge()
            added = self.add_entry_to_tag(knowledge, "General", suggestion)
            self.save_knowledge(knowledge)
            if added:
                return "LLM output wasn't valid JSON. Entire text stored in 'General'."
            else:
                return "LLM output wasn't valid JSON, but that text was already in 'General'."

