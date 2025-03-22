import discord
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path
import requests
import json
import re

####################################
# HELPER: Chunkify
####################################
def chunkify(text, max_size=1900):
    """
    Splits a string into chunks of up to max_size characters.
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
    Cog that ALWAYS filters knowledge for mention or askllm,
    then writes only the filtered subset to knowledgesearch.json,
    which is what actually gets passed to the LLM.
    """

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1234567890)
        self.config.register_global(model="gemma3:12b", api_url="http://localhost:11434")

        # Full knowledge base
        self.knowledge_file = cog_data_path(self) / "llm_knowledge.json"
        if not self.knowledge_file.exists():
            with self.knowledge_file.open("w") as f:
                json.dump({}, f)

        # Filtered knowledge file (written for each query)
        self.search_file = cog_data_path(self) / "knowledgesearch.json"
        if not self.search_file.exists():
            with self.search_file.open("w") as f:
                json.dump({}, f, indent=2)

    ################################################################
    # Load/save knowledge
    ################################################################
    def load_knowledge(self) -> dict:
        with self.knowledge_file.open("r") as f:
            return json.load(f)

    def save_knowledge(self, knowledge: dict):
        with self.knowledge_file.open("w") as f:
            json.dump(knowledge, f, indent=4)

    def load_searchfile(self) -> dict:
        with self.search_file.open("r") as f:
            return json.load(f)

    def save_searchfile(self, subset: dict):
        with self.search_file.open("w") as f:
            json.dump(subset, f, indent=2)

    ################################################################
    # Word-based partial filter
    ################################################################
    def filter_knowledge(self, question: str, knowledge: dict, max_entries=30) -> dict:
        """
        1) For each tag, we check if question's words appear in tag name or in entry text.
        2) If total matched > max_entries, we cut it down so we don't blow up LLM context.
        3) Return the final subset.
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

        # If we exceed max_entries, cut down
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
    # LLM Query
    ################################################################
    async def _get_model(self):
        return await self.config.model()

    async def _get_api_url(self):
        return await self.config.api_url()

    async def query_llm(self, prompt: str) -> str:
        model = await self._get_model()
        api_url = await self._get_api_url()

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        headers = {"Content-Type": "application/json"}

        r = requests.post(f"{api_url}/api/chat", json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "No valid response received.")

    ################################################################
    # on_message -> word filter => knowledgesearch => LLM
    ################################################################
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return
        if self.bot.user.mentioned_in(message):
            query = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
            if not query:
                return

            # 1) load full knowledge
            knowledge = self.load_knowledge()
            # 2) filter
            subset = self.filter_knowledge(query, knowledge, max_entries=30)
            # 3) store in knowledgesearch
            self.save_searchfile(subset)

            # 4) read back the subset
            partial = self.load_searchfile()

            # 5) build prompt
            prompt = (
                "Use the following relevant knowledge to answer. Do not guess.\n\n"
                f"Knowledge:\n{json.dumps(partial, indent=2)}\n\n"
                f"User question: {query}"
            )

            try:
                async with message.channel.typing():
                    response = await self.query_llm(prompt)

                # If the response is short, send once
                if len(response) <= 2000:
                    await message.channel.send(f"```\n{response}\n```")
                else:
                    chunks = chunkify(response, 1900)
                    for idx, c in enumerate(chunks, start=1):
                        await message.channel.send(f"**(Part {idx}/{len(chunks)}):**\n```\n{c}\n```")
            except Exception as e:
                await message.channel.send(f"Error: {e}")

    ################################################################
    # setmodel, setapi, modellist
    ################################################################
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

    ################################################################
    # askllm -> also only partial knowledge => knowledgesearch
    ################################################################
    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """
        1) Load entire knowledge
        2) Filter by question
        3) Save subset in knowledgesearch.json
        4) Read that subset, prompt the LLM
        5) Only chunk if final answer > 2000 chars
        """
        knowledge = self.load_knowledge()
        subset = self.filter_knowledge(question, knowledge, max_entries=30)
        if subset:
            note = "Filtered knowledge based on your query."
            self.save_searchfile(subset)
        else:
            note = "No matched entries -> empty knowledge subset."
            self.save_searchfile({})  # store empty dict

        partial = self.load_searchfile()
        prompt = (
            f"{note}\n\n"
            "Use the following relevant knowledge to answer accurately. Do not guess.\n\n"
            f"Knowledge:\n{json.dumps(partial, indent=2)}\n\n"
            f"User question: {question}"
        )

        try:
            async with ctx.typing():
                answer = await self.query_llm(prompt)
        except Exception as e:
            return await ctx.send(f"Error: {e}")

        if len(answer) <= 2000:
            await ctx.send(f"```\n{answer}\n```")
        else:
            parts = chunkify(answer, 1900)
            for i, c in enumerate(parts, 1):
                await ctx.send(f"**Answer (Part {i}/{len(parts)}):**\n```\n{c}\n```")

    ################################################################
    # Knowledge base commands
    ################################################################
    def add_entry(self, knowledge: dict, tag: str, info: str):
        if tag not in knowledge:
            knowledge[tag] = []
        if info in knowledge[tag]:
            return False
        knowledge[tag].append(info)
        return True

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx, tag: str, *, info: str):
        k = self.load_knowledge()
        added = self.add_entry(k, tag, info)
        if added:
            self.save_knowledge(k)
            await ctx.send(f"Stored info under tag {tag}")
        else:
            await ctx.send(f"Already exists under {tag}, skipping.")

    @commands.command()
    async def llmknowshow(self, ctx):
        """
        Show entire knowledge, chunk if needed.
        """
        k = self.load_knowledge()
        out = ""
        for tag, items in sorted(k.items()):
            out += f"{tag}:\n"
            for i, val in enumerate(items):
                out += f"  [{i}] {val}\n"

        if len(out) <= 2000:
            await ctx.send(f"```\n{out}\n```")
        else:
            parts = chunkify(out, 1900)
            for idx, chunk in enumerate(parts, 1):
                await ctx.send(f"LLM Knowledge Base (Part {idx}/{len(parts)}):\n```\n{chunk}\n```")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdelete(self, ctx, tag: str, index: int):
        k = self.load_knowledge()
        if tag not in k:
            return await ctx.send("Tag not found.")
        if 0 <= index < len(k[tag]):
            removed = k[tag].pop(index)
            if not k[tag]:
                del k[tag]
            self.save_knowledge(k)
            await ctx.send(f"Deleted {removed}")
        else:
            await ctx.send("Invalid index")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx, tag: str):
        k = self.load_knowledge()
        if tag not in k:
            return await ctx.send("Tag not found.")
        del k[tag]
        self.save_knowledge(k)
        await ctx.send(f"Deleted entire tag {tag}")

    ################################################################
    # learn <tag> <amount> -> also no big knowledge
    ################################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, tag: str, amount: int = 20):
        """
        1) Read last <amount> messages (excluding bots + this cmd).
        2) LLM sees only existing solutions for <tag>, plus conversation, never entire knowledge.
        3) If user says yes, parse the new solutions.
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

        # Only pass existing solutions for that tag
        # => no other tags, no entire knowledge
        knowledge = self.load_knowledge()
        existing = knowledge.get(tag, [])

        base_prompt = (
            f"Current solutions for tag '{tag}':\n"
            f"{json.dumps(existing, indent=2)}\n\n"
            "Below is a conversation. Extract new or refined solutions for that tag only.\n"
            "Output valid JSON, e.g. {\"solutions\": [\"fix1\", \"fix2\"]}\n"
            "No user mentions/logs.\n\n"
            f"Conversation:\n{conversation}"
        )

        try:
            async with ctx.typing():
                suggestion = await self.query_llm(base_prompt)
        except Exception as e:
            return await ctx.send(f"Error: {e}")

        # chunk if needed
        if len(suggestion) <= 2000:
            await ctx.send(f"```\n{suggestion}\n```")
        else:
            parts = chunkify(suggestion, 1900)
            for i, c in enumerate(parts, 1):
                await ctx.send(f"**Suggested (Part {i}/{len(parts)}):**\n```\n{c}\n```")

        await ctx.send("Type `yes` to confirm, `no [instructions]` to refine, or `stop` to cancel.")

        def check(m: discord.Message):
            return (m.author == ctx.author) and (m.channel == ctx.channel)

        while True:
            try:
                msg = await self.bot.wait_for("message", check=check, timeout=120)
                lower = msg.content.strip().lower()
                if lower == "yes":
                    ret = self.store_learn_solutions(tag, suggestion)
                    await ctx.send(ret)
                    break
                elif lower == "stop":
                    await ctx.send("Learning cancelled.")
                    break
                elif lower.startswith("no"):
                    instructions = msg.content[2:].strip()
                    refine_prompt = base_prompt + "\nAdditional instructions:\n" + instructions
                    try:
                        async with ctx.typing():
                            suggestion = await self.query_llm(refine_prompt)
                        if len(suggestion) <= 2000:
                            await ctx.send(f"```\n{suggestion}\n```")
                        else:
                            more_parts = chunkify(suggestion, 1900)
                            for ix, cc in enumerate(more_parts, 1):
                                await ctx.send(f"**Updated (Part {ix}/{len(more_parts)}):**\n```\n{cc}\n```")
                        await ctx.send("Type `yes` to confirm, `no [instructions]` to refine again, or `stop` to cancel.")
                    except Exception as ee:
                        await ctx.send(f"Error: {ee}")
                else:
                    await ctx.send("Please type `yes`, `no [instructions]`, or `stop`.")
            except Exception as e:
                await ctx.send(f"Error or timeout: {e}")
                break

    def store_learn_solutions(self, tag: str, suggestion: str) -> str:
        """
        Parse LLM's suggestion for new solutions:
        Must be in {\"solutions\": [...]} JSON or fallback to 'General'.
        """
        pat = r"(?s)```json\s*(\{.*?\})\s*```"
        mt = re.search(pat, suggestion)
        raw = suggestion
        if mt:
            raw = mt.group(1)

        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("parsed data isn't a dict")
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
            return f"Added {changed} solution(s) to tag '{tag}'."
        except Exception:
            knowledge = self.load_knowledge()
            added = self.add_entry_to_tag(knowledge, "General", suggestion)
            self.save_knowledge(knowledge)
            if added:
                return "Not valid JSON. Entire text stored in 'General'."
            else:
                return "Not valid JSON, but text was already in 'General'."
