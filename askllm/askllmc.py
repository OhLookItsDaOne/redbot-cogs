import discord
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path
import requests
import json
import re

####################################
# HELPER: Chunkify for large outputs
####################################
def chunkify(text, max_size=1900):
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
    """Cog with a separate knowledgesearch.json for partial data."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1234567890)
        self.config.register_global(model="gemma3:12b", api_url="http://localhost:11434")

        # Full knowledge base
        self.knowledge_file = cog_data_path(self) / "llm_knowledge.json"
        if not self.knowledge_file.exists():
            with self.knowledge_file.open("w") as f:
                json.dump({}, f)

        # Filtered knowledge file
        self.search_file = cog_data_path(self) / "knowledgesearch.json"
        if not self.search_file.exists():
            with self.search_file.open("w") as f:
                json.dump({}, f, indent=4)

    def load_knowledge(self):
        with self.knowledge_file.open("r") as f:
            return json.load(f)

    def save_knowledge(self, knowledge: dict):
        with self.knowledge_file.open("w") as f:
            json.dump(knowledge, f, indent=4)

    def load_searchfile(self):
        with self.search_file.open("r") as f:
            return json.load(f)

    def save_searchfile(self, subset: dict):
        with self.search_file.open("w") as f:
            json.dump(subset, f, indent=4)

    async def _get_model(self):
        return await self.config.model()

    async def _get_api_url(self):
        return await self.config.api_url()

    ############################################
    # Simple partial filter
    ############################################
    def filter_knowledge(self, question: str, knowledge: dict, max_entries=30) -> dict:
        """
        1) Tag-level match if a query word is in the tag.
        2) Entry-level match if a query word is in the entry text.
        3) Limit total relevant entries to max_entries.
        """
        words = set(re.sub(r"[^\w\s]", "", question.lower()).split())
        result = {}
        total_count = 0

        for tag, entries in knowledge.items():
            tagmatch = any(w in tag.lower() for w in words)
            matched_entries = []
            for entry in entries:
                entry_str = re.sub(r"[^\w\s]", "", str(entry).lower())
                if any(w in entry_str for w in words):
                    matched_entries.append(entry)
            if tagmatch or matched_entries:
                chosen = matched_entries if matched_entries else entries
                result[tag] = chosen
                total_count += len(chosen)

        # If more than max_entries, cut it down
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

    ############################################
    # Query LLM with a custom prompt
    ############################################
    async def query_llm(self, prompt: str):
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

    ############################################
    # on_message: entire knowledge
    ############################################
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
                            "Use the following knowledge to answer accurately.\n\n"
                            f"Knowledge:\n{json.dumps(knowledge)}\n\n"
                            f"Question: {question}"
                        )
                        response = await self.query_llm(prompt)
                    chunks = chunkify(response, 1900)
                    for idx, c in enumerate(chunks, 1):
                        await message.channel.send(f"**(Part {idx}/{len(chunks)}):**\n```\n{c}\n```")
                except Exception as e:
                    await message.channel.send(f"Error: {e}")

    ############################################
    # Basic / Model mgmt commands
    ############################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmodel(self, ctx, model: str):
        await self.config.model.set(model)
        await ctx.send(f"LLM model set to {model}")

    @commands.command()
    async def modellist(self, ctx):
        api_url = await self._get_api_url()
        r = requests.get(f"{api_url}/api/tags")
        r.raise_for_status()
        data = r.json()
        models = [m["name"] for m in data.get("models", [])]
        await ctx.send(f"Available models: {', '.join(models)}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setapi(self, ctx, url: str):
        url = url.rstrip("/")
        await self.config.api_url.set(url)
        await ctx.send(f"Ollama API URL set to {url}")

    ##################################################
    # askllm -> create a new knowledgesearch.json
    ##################################################
    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """
        Filter knowledge => save to 'knowledgesearch.json' => read that file => LLM => chunkify output
        """
        full_knowledge = self.load_knowledge()
        filtered = self.filter_knowledge(question, full_knowledge, max_entries=30)

        if filtered:
            note = "Filtered knowledge based on your query."
            self.save_searchfile(filtered)  # <--- STORE to knowledgesearch.json
        else:
            note = "No matched entries -> using entire knowledge."
            self.save_searchfile(full_knowledge)

        # Now read from knowledgesearch.json
        partial = self.load_searchfile()
        # Build LLM prompt
        prompt = (
            f"{note}\n\n"
            "Use the following knowledge to answer. Do not guess.\n\n"
            f"Knowledge:\n{json.dumps(partial, indent=2)}\n\n"
            f"User Question: {question}"
        )

        try:
            async with ctx.typing():
                answer = await self.query_llm(prompt)
            chunks = chunkify(answer, max_size=1900)
        except Exception as e:
            chunks = [f"Error: {e}"]

        for i, c in enumerate(chunks, 1):
            await ctx.send(f"**Answer (Part {i}/{len(chunks)}):**\n```\n{c}\n```")

    ##################################################
    # Knowledge base commands
    ##################################################
    def add_entry(self, knowledge: dict, tag: str, info: str):
        """
        Add info to a tag if not duplicate
        """
        if tag not in knowledge:
            knowledge[tag] = []
        if info in knowledge[tag]:
            return False
        knowledge[tag].append(info)
        return True

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx, tag: str, *, info: str):
        knowledge = self.load_knowledge()
        added = self.add_entry(knowledge, tag, info)
        if added:
            self.save_knowledge(knowledge)
            await ctx.send(f"Stored info under tag {tag}")
        else:
            await ctx.send(f"Already exists under {tag}; skipping.")

    @commands.command()
    async def llmknowshow(self, ctx):
        knowledge = self.load_knowledge()
        # Build text
        out = ""
        for tag, items in sorted(knowledge.items()):
            out += f"{tag}:\n"
            for i, val in enumerate(items):
                out += f"  [{i}] {val}\n"

        chunks = chunkify(out, 1900)
        for idx, chunk in enumerate(chunks, 1):
            await ctx.send(f"LLM Knowledge Base (Part {idx}/{len(chunks)}):\n```\n{chunk}\n```")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdelete(self, ctx, tag: str, index: int):
        knowledge = self.load_knowledge()
        if tag not in knowledge:
            return await ctx.send("Tag not found.")
        if 0 <= index < len(knowledge[tag]):
            removed = knowledge[tag].pop(index)
            if not knowledge[tag]:
                del knowledge[tag]
            self.save_knowledge(knowledge)
            await ctx.send(f"Deleted {removed}")
        else:
            await ctx.send("Invalid index")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx, tag: str):
        knowledge = self.load_knowledge()
        if tag not in knowledge:
            return await ctx.send("Tag not found.")
        del knowledge[tag]
        self.save_knowledge(knowledge)
        await ctx.send(f"Deleted entire tag {tag}")

    ############################################
    # learn <tag> <amount>
    ############################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, tag: str, amount: int=20):
        """
        Pulls last <amount> messages, shows existing solutions for <tag>,
        LLM tries to propose new solutions, which we store if user says yes.
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

        knowledge = self.load_knowledge()
        existing = knowledge.get(tag, [])

        base_prompt = (
            f"Existing solutions for tag '{tag}':\n"
            f"{json.dumps(existing, indent=2)}\n\n"
            "Below is a conversation. Extract only new or refined solutions for that tag.\n"
            "Output them as valid JSON, for example:\n"
            "{ \"solutions\": [\"some new fix\", \"some other fix\"] }\n"
            "No user mentions/logs.\n\n"
            f"Conversation:\n{conversation}"
        )

        try:
            async with ctx.typing():
                suggestion = await self.query_llm(base_prompt)
        except Exception as e:
            return await ctx.send(f"Error: {e}")

        # chunkify the suggestion itself
        sug_chunks = chunkify(suggestion, 1900)
        for i, ch in enumerate(sug_chunks, 1):
            await ctx.send(f"**Suggested (Part {i}/{len(sug_chunks)}):**\n```\n{ch}\n```")

        await ctx.send("Type `yes` to confirm, `no [instructions]` to refine, or `stop` to cancel.")

        def check(m: discord.Message):
            return m.author == ctx.author and m.channel == ctx.channel

        while True:
            try:
                resp = await self.bot.wait_for("message", check=check, timeout=120)
                txt = resp.content.strip().lower()
                if txt == "yes":
                    ret = self.store_learn_solutions(tag, suggestion)
                    await ctx.send(ret)
                    break
                elif txt == "stop":
                    await ctx.send("Learning cancelled.")
                    break
                elif txt.startswith("no"):
                    instructions = resp.content[2:].strip()
                    refine_prompt = base_prompt + "\nAdditional instructions:\n" + instructions
                    try:
                        async with ctx.typing():
                            suggestion = await self.query_llm(refine_prompt)
                        refine_chunks = chunkify(suggestion, 1900)
                        for ix, c2 in enumerate(refine_chunks, 1):
                            await ctx.send(f"**Updated (Part {ix}/{len(refine_chunks)}):**\n```\n{c2}\n```")
                        await ctx.send("Type `yes` to confirm, `no [instructions]` to refine again, or `stop` to cancel.")
                    except Exception as ee:
                        await ctx.send(f"Error: {ee}")
                else:
                    await ctx.send("Please type `yes`, `no [instructions]`, or `stop`.")
            except Exception as e:
                await ctx.send(f"Error or timeout: {e}")
                break

    def store_learn_solutions(self, tag: str, suggestion: str) -> str:
        pat = r"(?s)```json\s*(\{.*?\})\s*```"
        mt = re.search(pat, suggestion)
        raw = suggestion
        if mt:
            raw = mt.group(1)
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("parsed is not a dict")
            solutions = data.get("solutions", [])
            if not isinstance(solutions, list):
                raise ValueError("missing 'solutions' array")

            knowledge = self.load_knowledge()
            changed = 0
            if tag not in knowledge:
                knowledge[tag] = []
            for fix in solutions:
                if fix not in knowledge[tag]:
                    knowledge[tag].append(fix)
                    changed += 1
            self.save_knowledge(knowledge)
            return f"Added {changed} new solution(s) to tag '{tag}'"
        except Exception:
            knowledge = self.load_knowledge()
            # store entire text in "General"
            from_text = self.add_entry_to_tag(knowledge, "General", suggestion)
            self.save_knowledge(knowledge)
            if from_text:
                return "LLM output wasn't valid JSON. Entire text stored in 'General'."
            else:
                return "LLM output wasn't valid JSON, and was already in 'General'."
