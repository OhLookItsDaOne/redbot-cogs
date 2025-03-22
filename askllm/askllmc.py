import re
import json
import requests
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path
import discord

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
    """Cog with advanced filtering for large knowledge bases."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=123456789)
        self.config.register_global(model="gemma3:12b", api_url="http://localhost:11434")

        # Knowledge base file
        self.knowledge_file = cog_data_path(self) / "llm_knowledge.json"
        if not self.knowledge_file.exists():
            with self.knowledge_file.open("w") as f:
                json.dump({}, f)

    def load_knowledge(self):
        with self.knowledge_file.open("r") as f:
            return json.load(f)

    def save_knowledge(self, knowledge):
        with self.knowledge_file.open("w") as f:
            json.dump(knowledge, f, indent=4)

    async def _get_api_url(self):
        return await self.config.api_url()

    async def _get_model(self):
        return await self.config.model()

    ############################
    # Filtering
    ############################
    def filter_knowledge_by_query(self, question: str, knowledge: dict, max_entries=30) -> dict:
        """
        1) Tag-level match:
           - If a word in question matches the tag name, we keep the entire tag for scanning.
        2) Entry-level match:
           - For each tag, we keep only the entries that partially match ANY word from question.
        3) If the total number of relevant entries > max_entries, we slice it to max_entries.
        4) Return the final subset as {tag: [entries]}.
        """
        # Very naive tokenizing
        words = set(re.sub(r"[^\w\s]", "", question.lower()).split())

        result = {}
        total_count = 0
        # first pass: which tags might match
        for tag, entries in knowledge.items():
            # check if any word is in tag name
            tagmatch = any(w in tag.lower() for w in words)
            filtered_entries = []
            for entry in entries:
                entry_str = re.sub(r"[^\w\s]", "", str(entry).lower())
                if any(w in entry_str for w in words):
                    filtered_entries.append(entry)
            # if the tag matched by name or we found matched entries
            if tagmatch or filtered_entries:
                result[tag] = filtered_entries if filtered_entries else entries
                total_count += len(result[tag])
        # Now we see if total_count > max_entries
        if total_count > max_entries:
            # We trim down so we don't exceed context
            # We'll do a simple approach: just slice each tag's list until we total max_entries
            new_result = {}
            running = 0
            for t, ents in result.items():
                needed = max_entries - running
                if needed <= 0:
                    break
                if len(ents) > needed:
                    new_result[t] = ents[:needed]
                    running += needed
                else:
                    new_result[t] = ents
                    running += len(ents)
            return new_result
        return result

    ############################
    # on_message mention -> uses entire knowledge (no filter)
    ############################
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
                    await message.channel.send(answer)
                except Exception as e:
                    await message.channel.send(f"Error: {e}")

    ############################
    # Basic LLM mgmt
    ############################
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

    async def query_llm(self, prompt: str):
        model = await self._get_model()
        api_url = await self._get_api_url()
        payload = {"model": model, "messages":[{"role":"user","content":prompt}], "stream":False}
        r = requests.post(f"{api_url}/api/chat", json=payload, headers={"Content-Type":"application/json"})
        r.raise_for_status()
        data = r.json()
        return data.get("message",{}).get("content","No valid response received.")

    ############################
    # askllm with filter
    ############################
    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """
        1) Filter knowledge by question.
        2) Build prompt.
        3) Send prompt to LLM.
        4) Chunk the response so it never exceeds Discord’s 2000-char limit.
        """
        full_knowledge = self.load_knowledge()
        relevant_knowledge = self.search_knowledge(question, full_knowledge)
    
        if relevant_knowledge:
            subset = relevant_knowledge
            note = "Filtered knowledge:\n"
        else:
            subset = full_knowledge
            note = "No matched entries, using entire knowledge.\n"
    
        prompt_context = (
            f"{note}"
            "Use the following knowledge to answer accurately. Do not guess.\n\n"
            f"Knowledge:\n{json.dumps(subset, indent=2)}\n\n"
            f"User Question: {question}"
        )
    
        model = await self.config.model()
        api_url = await self._get_api_url()
    
        async with ctx.typing():
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt_context}],
                    "stream": False
                }
                headers = {"Content-Type": "application/json"}
                r = requests.post(f"{api_url}/api/chat", json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
                answer = data.get("message", {}).get("content", "No valid response received.")
    
                # Now chunkify the final answer so it never exceeds 2000 chars
                chunks = chunkify(answer, max_size=1900)
            except Exception as e:
                chunks = [f"Error: {e}"]
    
        for i, c in enumerate(chunks, 1):
            await ctx.send(f"**Answer (Part {i}/{len(chunks)}):**\n```\n{c}\n```")

    ############################
    # Knowledge base commands
    ############################
    def add_entry(self, tag: str, info: str):
        knowledge = self.load_knowledge()
        if tag not in knowledge:
            knowledge[tag] = []
        if info not in knowledge[tag]:
            knowledge[tag].append(info)
            self.save_knowledge(knowledge)
            return True
        return False

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx, tag: str, *, info: str):
        added = self.add_entry(tag, info)
        if added:
            await ctx.send(f"Stored info under tag {tag}")
        else:
            await ctx.send(f"Already exists under {tag}; skipped.")

    @commands.command()
    async def llmknowshow(self, ctx):
        """
        Show entire knowledge in chunks
        """
        knowledge = self.load_knowledge()
        text = ""
        for tag, items in sorted(knowledge.items()):
            text += f"{tag}:\n"
            for i, val in enumerate(items):
                text += f"  [{i}] {val}\n"

        chunks = chunkify(text, 1900)
        for idx, c in enumerate(chunks, 1):
            await ctx.send(f"LLM Knowledge Base (Part {idx}/{len(chunks)}):\n```\n{c}\n```")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdelete(self, ctx, tag: str, index: int):
        knowledge = self.load_knowledge()
        if tag not in knowledge:
            return await ctx.send("Tag not found")
        if 0 <= index < len(knowledge[tag]):
            removed = knowledge[tag].pop(index)
            if not knowledge[tag]:
                del knowledge[tag]
            self.save_knowledge(knowledge)
            await ctx.send(f"Deleted: {removed}")
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
        await ctx.send(f"Tag {tag} removed.")

    ###############################################
    # learn <tag> <amount>
    ###############################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, tag: str, amount: int = 20):
        """
        Reads last [amount] messages (excluding bot and this cmd),
        reveals existing solutions for <tag>, 
        LLM tries to propose new solutions, storing them in `solutions` array
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
            "{ \"solutions\": [ \"fix one\", \"fix two\" ] }\n"
            "No user mentions/logs, only final steps. \n\n"
            f"Conversation:\n{conversation}"
        )

        async with ctx.typing():
            suggestion = await self.query_llm(base_prompt)

        await ctx.send(
            f"Suggested new solutions for tag '{tag}':\n```\n{suggestion}\n```\n"
            "Type `yes` to confirm, `no [instructions]` to refine, or `stop` to cancel."
        )

        def check(m: discord.Message):
            return m.author == ctx.author and m.channel == ctx.channel

        while True:
            try:
                reply = await self.bot.wait_for("message", check=check, timeout=120)
                low = reply.content.strip().lower()

                if low == "yes":
                    outcome = self.store_learn_solutions(tag, suggestion)
                    await ctx.send(outcome)
                    break
                elif low == "stop":
                    await ctx.send("Learning process cancelled.")
                    break
                elif low.startswith("no"):
                    instructions = reply.content[2:].strip()
                    refined = base_prompt + "\nAdditional instructions:\n" + instructions
                    async with ctx.typing():
                        suggestion = await self.query_llm(refined)
                    await ctx.send(
                        f"Updated solutions for tag '{tag}':\n```\n{suggestion}\n```\n"
                        "Type `yes` to confirm, `no [instructions]` to refine again, or `stop` to cancel."
                    )
                else:
                    await ctx.send("Please type `yes`, `no [instructions]`, or `stop`.")
            except Exception as e:
                await ctx.send(f"Error or timeout: {e}")
                break

    def store_learn_solutions(self, tag: str, suggestion: str) -> str:
        pat = r"(?s)```json\s*(\{.*?\})\s*```"
        mat = re.search(pat, suggestion)
        raw = suggestion
        if mat:
            raw = mat.group(1)

        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("parsed not a dict")
            solutions = data.get("solutions", [])
            if not isinstance(solutions, list):
                raise ValueError("no valid solutions array")

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
            added = self.add_entry_to_tag(knowledge, "General", suggestion)
            self.save_knowledge(knowledge)
            if added:
                return "LLM output wasn't valid JSON. Entire text stored in 'General'."
            else:
                return "LLM output wasn't valid JSON, but that text was already in 'General'."
