import discord
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path
import requests
import json
import re

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
    """Cog that uses synonyms + fallback to avoid minimal matches and also trims extra newlines."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1234567890)
        self.config.register_global(model="gemma3:12b", api_url="http://localhost:11434")

        self.knowledge_file = cog_data_path(self) / "llm_knowledge.json"
        if not self.knowledge_file.exists():
            with self.knowledge_file.open("w") as f:
                json.dump({}, f, indent=2)

        self.search_file = cog_data_path(self) / "knowledgesearch.json"
        if not self.search_file.exists():
            with self.search_file.open("w") as f:
                json.dump({}, f, indent=2)

        # Example synonyms dictionary: user keywords -> normalized tokens
        self.synonyms_map = {
            "crashed": "crash",
            "crashes": "crash",
            "crash": "crash",
            "ctd": "crash",
            "log": "log",
            "crashlog": "log",
            "broken": "issue",
            # add more synonyms if needed
        }

    def load_knowledge(self) -> dict:
        with self.knowledge_file.open("r") as f:
            return json.load(f)

    def save_knowledge(self, knowledge: dict):
        with self.knowledge_file.open("w") as f:
            json.dump(knowledge, f, indent=2)

    def load_searchfile(self) -> dict:
        with self.search_file.open("r") as f:
            return json.load(f)

    def save_searchfile(self, subset: dict):
        with self.search_file.open("w") as f:
            json.dump(subset, f, indent=2)

    async def _get_model(self):
        return await self.config.model()

    async def _get_api_url(self):
        return await self.config.api_url()

    def normalize_query_words(self, query: str) -> set:
        """
        1) remove punctuation
        2) split
        3) apply synonyms
        """
        words = re.sub(r"[^\w\s]", "", query.lower()).split()
        result = []
        for w in words:
            if w in self.synonyms_map:
                result.append(self.synonyms_map[w])
            else:
                result.append(w)
        return set(result)

    def filter_knowledge(self, query: str, knowledge: dict, max_entries=30) -> dict:
        """
        1) synonyms for words
        2) check each tag or entry for these words
        3) cut if > max_entries
        """
        words = self.normalize_query_words(query)
        out = {}
        total_count = 0

        for tag, entries in knowledge.items():
            tag_match = any(w in tag.lower() for w in words)
            matched = []
            for entry in entries:
                entry_str = re.sub(r"[^\w\s]", "", str(entry).lower())
                # Also apply synonyms
                if any(w in entry_str for w in words):
                    matched.append(entry)
            if tag_match or matched:
                chosen = matched if matched else entries
                out[tag] = chosen
                total_count += len(chosen)

        # cut if total_count > max_entries
        if total_count > max_entries:
            new_res = {}
            running = 0
            for tg, ents in out.items():
                needed = max_entries - running
                if needed <= 0:
                    break
                if len(ents) > needed:
                    new_res[tg] = ents[:needed]
                    running += needed
                else:
                    new_res[tg] = ents
                    running += len(ents)
            return new_res
        return out

    async def query_llm(self, prompt: str) -> str:
        """
        Send the prompt to LLM, remove excessive double newlines from response.
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
        text = data.get("message", {}).get("content", "No valid response received.")
        # remove big spacing
        text = text.replace("\n\n", "\n")
        return text

    ############################################
    # on_message => partial filter
    ############################################
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return
        if self.bot.user.mentioned_in(message):
            raw = message.content.replace(f"<@{self.bot.user.id}>","").strip()
            if not raw:
                return
            knowledge = self.load_knowledge()
            subset = self.filter_knowledge(raw, knowledge, max_entries=30)
            self.save_searchfile(subset)
            partial = self.load_searchfile()

            # If still empty, user might see disclaimers
            prompt = (
                "Use the following relevant knowledge to answer. Do not guess.\n"
                f"Knowledge:\n{json.dumps(partial, indent=2)}\n\n"
                f"User question: {raw}"
            )
            try:
                async with message.channel.typing():
                    llm_answer = await self.query_llm(prompt)
                if len(llm_answer) <= 2000:
                    await message.channel.send(f"```\n{llm_answer}\n```")
                else:
                    cparts = chunkify(llm_answer, 1900)
                    for idx, c in enumerate(cparts, 1):
                        await message.channel.send(f"**(Part {idx}/{len(cparts)}):**\n```\n{c}\n```")
            except Exception as e:
                await message.channel.send(f"Error: {e}")

    ############################################
    # Basic commands
    ############################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmodel(self, ctx, model: str):
        await self.config.model.set(model)
        await ctx.send(f"LLM model set to {model}")

    @commands.command()
    async def modellist(self, ctx):
        api_url = await self._get_api_url()
        try:
            r = requests.get(f"{api_url}/api/tags")
            r.raise_for_status()
            dat = r.json()
            models = [m["name"] for m in dat.get("models",[])]
            await ctx.send(f"Available models: {', '.join(models)}")
        except Exception as e:
            await ctx.send(f"Error: {e}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setapi(self, ctx, url: str):
        url = url.rstrip("/")
        await self.config.api_url.set(url)
        await ctx.send(f"Ollama API URL set to {url}")

    ############################################
    # askllm => partial filter => search file
    ############################################
    @commands.command()
    async def askllm(self, ctx, *, question: str):
        knowledge = self.load_knowledge()
        subset = self.filter_knowledge(question, knowledge, max_entries=30)
        self.save_searchfile(subset)

        partial = self.load_searchfile()
        prompt = (
            "Use the following relevant knowledge to answer accurately.\n"
            f"Knowledge:\n{json.dumps(partial, indent=2)}\n\n"
            f"User question: {question}"
        )

        try:
            async with ctx.typing():
                answer = await self.query_llm(prompt)
        except Exception as e:
            return await ctx.send(f"Error: {e}")

        answer = answer.replace("\n\n","\n")
        if len(answer) <= 2000:
            await ctx.send(f"```\n{answer}\n```")
        else:
            parts = chunkify(answer, 1900)
            for i, c in enumerate(parts,1):
                await ctx.send(f"**Answer (Part {i}/{len(parts)}):**\n```\n{c}\n```")

    ############################################
    # Knowledge base commands
    ############################################
    def add_entry(self, k: dict, tag: str, info: str):
        if tag not in k:
            k[tag] = []
        if info in k[tag]:
            return False
        k[tag].append(info)
        return True

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx, tag: str, *, info: str):
        kb = self.load_knowledge()
        added = self.add_entry(kb, tag, info)
        if added:
            self.save_knowledge(kb)
            await ctx.send(f"Stored info under tag {tag}")
        else:
            await ctx.send(f"Already in {tag}; skipping.")

    @commands.command()
    async def llmknowshow(self, ctx):
        kb = self.load_knowledge()
        out = ""
        for tag, items in sorted(kb.items()):
            out += f"{tag}:\n"
            for i, val in enumerate(items):
                out += f"  [{i}] {val}\n"

        if len(out) <= 2000:
            await ctx.send(f"```\n{out}\n```")
        else:
            cparts = chunkify(out,1900)
            for idx, cpart in enumerate(cparts,1):
                await ctx.send(f"LLM Knowledge Base (Part {idx}/{len(cparts)}):\n```\n{cpart}\n```")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdelete(self, ctx, tag: str, index: int):
        kb = self.load_knowledge()
        if tag not in kb:
            return await ctx.send("Tag not found.")
        if 0 <= index < len(kb[tag]):
            removed = kb[tag].pop(index)
            if not kb[tag]:
                del kb[tag]
            self.save_knowledge(kb)
            await ctx.send(f"Deleted: {removed}")
        else:
            await ctx.send("Invalid index")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx, tag: str):
        kb = self.load_knowledge()
        if tag not in kb:
            return await ctx.send("Tag not found.")
        del kb[tag]
        self.save_knowledge(kb)
        await ctx.send(f"Deleted entire tag {tag}")

    ############################################
    # learn <tag> <amount>
    ############################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, tag: str, amount: int=20):
        """
        Gather last <amount> messages (no entire knowledge),
        only existing solutions for <tag>,
        parse new solutions in JSON if user says 'yes'.
        """
        def not_command_or_bot(m: discord.Message):
            if m.author.bot: return False
            if m.id == ctx.message.id: return False
            return True

        msgs = []
        async for msg in ctx.channel.history(limit=amount+5):
            if not_command_or_bot(msg):
                msgs.append(msg)
            if len(msgs)>=amount:
                break

        msgs.reverse()
        conversation = "\n".join(f"{m.author.name}: {m.content}" for m in msgs)

        kb = self.load_knowledge()
        existing = kb.get(tag, [])

        prompt = (
            f"Existing solutions for tag '{tag}':\n"
            f"{json.dumps(existing, indent=2)}\n\n"
            "Below is a conversation. Extract new or refined solutions for that tag.\n"
            "Output valid JSON, e.g. {\"solutions\":[\"fix1\",\"fix2\"]}\n"
            "No user mentions.\n\n"
            f"Conversation:\n{conversation}"
        )

        try:
            async with ctx.typing():
                suggestion = await self.query_llm(prompt)
        except Exception as e:
            return await ctx.send(f"Error: {e}")

        # chunk if needed
        if len(suggestion)<=2000:
            await ctx.send(f"```\n{suggestion}\n```")
        else:
            sparts = chunkify(suggestion,1900)
            for i, spt in enumerate(sparts,1):
                await ctx.send(f"**(Part {i}/{len(sparts)}):**\n```\n{spt}\n```")

        await ctx.send("Type `yes` to confirm, `no [instr]` to refine, or `stop` to cancel.")

        def check(m: discord.Message):
            return (m.author==ctx.author) and (m.channel==ctx.channel)

        while True:
            try:
                rep = await self.bot.wait_for("message", check=check, timeout=120)
                lower = rep.content.strip().lower()
                if lower=="yes":
                    out = self.store_learn_solutions(tag, suggestion)
                    await ctx.send(out)
                    break
                elif lower=="stop":
                    await ctx.send("Learning cancelled.")
                    break
                elif lower.startswith("no"):
                    instructions = rep.content[2:].strip()
                    refine = prompt + "\nAdditional instructions:\n" + instructions
                    async with ctx.typing():
                        suggestion = await self.query_llm(refine)
                    if len(suggestion)<=2000:
                        await ctx.send(f"```\n{suggestion}\n```")
                    else:
                        reps2 = chunkify(suggestion,1900)
                        for i2, ct in enumerate(reps2,1):
                            await ctx.send(f"**(Part {i2}/{len(reps2)}):**\n```\n{ct}\n```")
                    await ctx.send("Type `yes` to confirm, `no [instr]` to refine again, or `stop` to cancel.")
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
                raise ValueError("parsed data not a dict")
            arr = data.get("solutions",[])
            if not isinstance(arr, list):
                raise ValueError("missing solutions array")

            kb = self.load_knowledge()
            changed=0
            if tag not in kb:
                kb[tag] = []
            for fix in arr:
                if fix not in kb[tag]:
                    kb[tag].append(fix)
                    changed+=1
            self.save_knowledge(kb)
            return f"Added {changed} new solutions to {tag}"
        except Exception:
            # store entire text in General
            kb = self.load_knowledge()
            if "General" not in kb:
                kb["General"]=[]
            if suggestion not in kb["General"]:
                kb["General"].append(suggestion)
                self.save_knowledge(kb)
                return "Not valid JSON. Stored entire text in 'General'."
            else:
                return "Not valid JSON. Already in 'General'."


