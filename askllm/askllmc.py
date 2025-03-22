import discord
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path
import requests
import json
import re
import time

####################################
# HELPER: Chunkify for large outputs
####################################
def chunkify(text, max_size=1900):
    """
    Splits a string into a list of chunks, each up to max_size chars.
    """
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
    """
    Cog to interact with Ollama-based LLM and manage knowledge storage.
    Includes:
    - askllm with partial knowledge filtering
    - chunked output to avoid 2000-char limit
    - learn <tag> <amount> for gathering solutions from recent messages
    - llmknow, llmknowshow, llmknowdelete, etc.
    """

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1234567890)
        self.config.register_global(model="gemma3:12b", api_url="http://localhost:11434")

        self.knowledge_file = cog_data_path(self) / "llm_knowledge.json"
        if not self.knowledge_file.exists():
            with self.knowledge_file.open("w") as f:
                json.dump({}, f)

    ################################################################
    # Basic knowledge file utilities
    ################################################################
    def load_knowledge(self):
        with self.knowledge_file.open("r") as f:
            return json.load(f)

    def save_knowledge(self, knowledge: dict):
        with self.knowledge_file.open("w") as f:
            json.dump(knowledge, f, indent=4)

    def add_entry_to_tag(self, knowledge: dict, tag: str, entry) -> bool:
        """
        Adds 'entry' to knowledge[tag], skipping duplicates.
        Returns True if successfully added, False if duplicate.
        """
        if tag not in knowledge:
            knowledge[tag] = []
        if entry in knowledge[tag]:
            return False
        knowledge[tag].append(entry)
        return True

    ################################################################
    # LLM query helpers
    ################################################################
    async def _get_api_url(self):
        return await self.config.api_url()

    async def _get_model(self):
        return await self.config.model()

    async def query_llm(self, prompt: str):
        """
        Sends a custom prompt to the LLM, returns the text response.
        """
        model = await self._get_model()
        api_url = await self._get_api_url()

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        headers = {"Content-Type": "application/json"}
        resp = requests.post(f"{api_url}/api/chat", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "No valid response received.")

    ################################################################
    # Partial knowledge filter
    ################################################################
    def filter_knowledge_by_query(self, question: str, knowledge: dict, max_entries=30) -> dict:
        """
        1) Tag-level match: if a word in question matches the tag name, we keep it.
        2) Entry-level match: check if any word from question is in the entry text.
        3) If total relevant entries > max_entries, slice it to avoid large context.
        4) Return the subset as { tag: [entries] }.
        """
        words = set(re.sub(r"[^\w\s]", "", question.lower()).split())
        result = {}
        total_count = 0

        for tag, entries in knowledge.items():
            tag_match = any(w in tag.lower() for w in words)
            matched = []
            for entry in entries:
                entry_str = re.sub(r"[^\w\s]", "", str(entry).lower())
                if any(w in entry_str for w in words):
                    matched.append(entry)
            if tag_match or matched:
                # If the tag matched but matched is empty, keep the entire tag.
                # Otherwise keep matched only.
                final_list = matched if matched else entries
                result[tag] = final_list
                total_count += len(final_list)

        # limit total_count to max_entries
        if total_count > max_entries:
            new_res = {}
            running = 0
            for t, ents in result.items():
                needed = max_entries - running
                if needed <= 0:
                    break
                if len(ents) > needed:
                    new_res[t] = ents[:needed]
                    running += needed
                else:
                    new_res[t] = ents
                    running += len(ents)
            return new_res
        return result

    ################################################################
    # on_message: mention the bot -> entire knowledge
    ################################################################
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return
        if self.bot.user.mentioned_in(message):
            question = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
            if question:
                try:
                    async with message.channel.typing():
                        knowledge = self.load_knowledge()
                        prompt = (
                            "Use the following knowledge to answer accurately. Do not guess.\n\n"
                            f"Knowledge:\n{json.dumps(knowledge)}\n\n"
                            f"Question: {question}"
                        )
                        answer = await self.query_llm(prompt)
                    # chunkify the final answer
                    chunks = chunkify(answer, max_size=1900)
                    for idx, c in enumerate(chunks, 1):
                        await message.channel.send(f"**(Part {idx}/{len(chunks)}):**\n```\n{c}\n```")
                except Exception as e:
                    await message.channel.send(f"Error: {e}")

    ################################################################
    # Basic LLM / Model mgmt commands
    ################################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmodel(self, ctx, model: str):
        """
        Sets the default LLM model.
        """
        await self.config.model.set(model)
        await ctx.send(f"LLM model set to {model}")

    @commands.command()
    async def modellist(self, ctx):
        """
        Lists available models from Ollama.
        """
        api_url = await self._get_api_url()
        r = requests.get(f"{api_url}/api/tags")
        r.raise_for_status()
        data = r.json()
        models = [m["name"] for m in data.get("models", [])]
        await ctx.send(f"Available models: {', '.join(models)}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setapi(self, ctx, url: str):
        """
        Sets the Ollama API URL.
        """
        url = url.rstrip("/")
        await self.config.api_url.set(url)
        await ctx.send(f"Ollama API URL set to {url}")

    ################################################################
    # askllm: partial filter -> chunkify final answer
    ################################################################
    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """
        1) Filter knowledge to only relevant entries (tag name or content).
        2) If nothing found, fallback to entire knowledge.
        3) Send to LLM, chunkify final answer to avoid 2k limit.
        """
        knowledge = self.load_knowledge()
        filtered = self.filter_knowledge_by_query(question, knowledge, max_entries=30)

        if filtered:
            note = "Filtered knowledge:\n"
            subset = filtered
        else:
            note = "No matched entries, using entire knowledge.\n"
            subset = knowledge

        prompt = (
            f"{note}"
            "Use the following knowledge to answer accurately. Do not guess.\n\n"
            f"Knowledge:\n{json.dumps(subset, indent=2)}\n\n"
            f"User Question: {question}"
        )
        try:
            async with ctx.typing():
                resp = await self.query_llm(prompt)
            chunks = chunkify(resp, 1900)
        except Exception as e:
            chunks = [f"Error: {e}"]

        for i, c in enumerate(chunks, start=1):
            await ctx.send(f"**Answer (Part {i}/{len(chunks)}):**\n```\n{c}\n```")

    ################################################################
    # Knowledge base commands
    ################################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx, tag: str, *, info: str):
        """
        Adds <info> under <tag>, skipping duplicates.
        """
        knowledge = self.load_knowledge()
        added = self.add_entry_to_tag(knowledge, tag, info)
        self.save_knowledge(knowledge)
        if added:
            await ctx.send(f"Stored info under tag {tag}")
        else:
            await ctx.send(f"Already exists under {tag}; skipped.")

    @commands.command()
    async def llmknowshow(self, ctx):
        """
        Shows the entire knowledge base in chunks to avoid 2k limit.
        """
        knowledge = self.load_knowledge()
        text = ""
        for tag, items in sorted(knowledge.items()):
            text += f"{tag}:\n"
            for i, val in enumerate(items):
                text += f"  [{i}] {val}\n"

        chunks = chunkify(text, 1900)
        for idx, chunk in enumerate(chunks, 1):
            await ctx.send(f"LLM Knowledge Base (Part {idx}/{len(chunks)}):\n```\n{chunk}\n```")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdelete(self, ctx, tag: str, index: int):
        """
        Deletes a specific entry by index under <tag>.
        """
        knowledge = self.load_knowledge()
        if tag not in knowledge:
            return await ctx.send("Tag not found.")
        if 0 <= index < len(knowledge[tag]):
            deleted = knowledge[tag].pop(index)
            if not knowledge[tag]:
                del knowledge[tag]
            self.save_knowledge(knowledge)
            await ctx.send(f"Deleted: {deleted}")
        else:
            await ctx.send("Invalid index.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx, tag: str):
        """
        Deletes the entire <tag>.
        """
        knowledge = self.load_knowledge()
        if tag not in knowledge:
            return await ctx.send("Tag not found.")
        del knowledge[tag]
        self.save_knowledge(knowledge)
        await ctx.send(f"Tag {tag} removed.")

    ################################################################
    # learn <tag> <amount>
    ################################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, tag: str, amount: int = 20):
        """
        Reads last [amount] messages (excluding bot msgs & cmd),
        shows existing solutions for <tag>, LLM attempts to produce new solutions in JSON.
        Admin says yes/no/stop; we store them if yes.
        """
        def not_command_or_bot(m: discord.Message):
            if m.author.bot:
                return False
            if m.id == ctx.message.id:
                return False
            return True

        msgs = []
        async for msg in ctx.channel.history(limit=amount+5):
            if not_command_or_bot(msg):
                msgs.append(msg)
            if len(msgs) >= amount:
                break

        msgs.reverse()
        conversation = "\n".join(f"{m.author.name}: {m.content}" for m in msgs)

        # existing
        knowledge = self.load_knowledge()
        existing = knowledge.get(tag, [])

        base_prompt = (
            f"Existing solutions for tag '{tag}':\n"
            f"{json.dumps(existing, indent=2)}\n\n"
            "Below is a conversation. Extract only new or refined solutions for that tag.\n"
            "Output them as valid JSON, for example:\n"
            "{ \"solutions\": [ \"some new fix\", \"some other fix\" ] }\n"
            "No user mentions, logs, or duplicates. Only final steps.\n\n"
            f"Conversation:\n{conversation}"
        )

        try:
            async with ctx.typing():
                suggestion = await self.query_llm(base_prompt)
        except Exception as e:
            return await ctx.send(f"Error: {e}")

        chunks = chunkify(suggestion, 1900)
        for i, c in enumerate(chunks, 1):
            await ctx.send(f"**Suggested new solutions (Part {i}/{len(chunks)}):**\n```\n{c}\n```")

        await ctx.send("Type `yes` to confirm, `no [instructions]` to refine, or `stop` to cancel.")

        def check(m: discord.Message):
            return m.author == ctx.author and m.channel == ctx.channel

        while True:
            try:
                reply = await self.bot.wait_for("message", check=check, timeout=120)
                text = reply.content.strip().lower()
                if text == "yes":
                    ret = self.store_learn_solutions(tag, suggestion)
                    await ctx.send(ret)
                    break
                elif text == "stop":
                    await ctx.send("Learning cancelled.")
                    break
                elif text.startswith("no"):
                    instructions = reply.content[2:].strip()
                    new_prompt = base_prompt + f"\n\nAdditional instructions:\n{instructions}"
                    try:
                        async with ctx.typing():
                            suggestion = await self.query_llm(new_prompt)
                        chunks2 = chunkify(suggestion, 1900)
                        for i2, c2 in enumerate(chunks2, 1):
                            await ctx.send(f"**Updated solutions (Part {i2}/{len(chunks2)}):**\n```\n{c2}\n```")
                        await ctx.send("Type `yes` to confirm, `no [instructions]` to refine again, or `stop` to cancel.")
                    except Exception as e:
                        await ctx.send(f"Error: {e}")
                else:
                    await ctx.send("Please type `yes`, `no [instructions]`, or `stop`.")
            except Exception as e:
                await ctx.send(f"Error or timeout: {e}")
                break

    def store_learn_solutions(self, tag: str, suggestion: str) -> str:
        """
        The LLM should produce something like:
        ```json
        {
          "solutions": [
            "some fix",
            "another fix"
          ]
        }
        ```
        We'll parse that block, store each fix in 'tag'.
        If invalid, store entire text in 'General'.
        """
        pattern = r"(?s)```json\s*(\{.*?\})\s*```"
        match = re.search(pattern, suggestion)
        raw = suggestion
        if match:
            raw = match.group(1)

        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("parsed data is not a dict")
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
            knowledge = self.load_knowledge()
            added = self.add_entry_to_tag(knowledge, "General", suggestion)
            self.save_knowledge(knowledge)
            if added:
                return "LLM output wasn't valid JSON. Entire text stored in 'General'."
            else:
                return "LLM output wasn't valid JSON, but that text was already in 'General'."
