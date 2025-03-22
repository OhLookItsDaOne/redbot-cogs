import discord
import os
import json
import re
import requests
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path

##############################################
# Filter lines by naive keyword check
##############################################
def filter_lines_by_keywords(lines: list, keywords: list) -> list:
    filtered = []
    for line in lines:
        low = line.lower()
        if any(kw.lower() in low for kw in keywords):
            filtered.append(line)
    return filtered

class LLMManager(commands.Cog):
    """
    Revised Cog with fallback if LLM picks no files in Phase1 or returns invalid JSON.
    We then default to scanning all files for naive matches.
    Also supports the plain-text !learn approach, mention -> 2-phase approach, etc.
    """

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9999999999)
        self.config.register_global(model="gemma3:12b", api_url="http://localhost:11434")

        self.tags_folder = cog_data_path(self) / "tags"
        if not self.tags_folder.exists():
            self.tags_folder.mkdir(parents=True, exist_ok=True)

    ##############################################
    # Basic Model/API
    ##############################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmodel(self, ctx, model: str):
        """Sets the default LLM model."""
        await self.config.model.set(model)
        await ctx.send(f"LLM model set to '{model}'.")

    @commands.command()
    async def modellist(self, ctx):
        """Lists available models from the Ollama instance."""
        api_url = await self.config.api_url()
        try:
            r = requests.get(f"{api_url}/api/tags")
            r.raise_for_status()
            data = r.json()
            models = [m["name"] for m in data.get("models", [])]
            if models:
                await ctx.send(f"Available models: {', '.join(models)}")
            else:
                await ctx.send("No models found on Ollama.")
        except Exception as e:
            await ctx.send(f"Error: {e}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setapi(self, ctx, url: str):
        """Sets the Ollama API URL."""
        url = url.rstrip("/")
        await self.config.api_url.set(url)
        await ctx.send(f"Ollama API URL set to '{url}'.")

    ##############################################
    # Tag file loading/saving
    ##############################################
    def list_tag_files(self) -> list:
        """Return all .json files in tags folder."""
        all_files = []
        for fname in os.listdir(self.tags_folder):
            if fname.lower().endswith(".json"):
                all_files.append(fname)
        return all_files

    def load_tag_file(self, fname: str) -> list:
        """Loads a tag file as a list of lines, or returns [] if missing/corrupt."""
        path = self.tags_folder / fname
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except:
            pass
        return []

    def save_tag_file(self, fname: str, lines: list):
        """Saves lines to <fname> in JSON."""
        path = self.tags_folder / fname
        with path.open("w", encoding="utf-8") as f:
            json.dump(lines, f, indent=2)

    ##############################################
    # LLM call with typing, no chunkify
    ##############################################
    async def query_llm(self, prompt: str, channel: discord.abc.Messageable) -> str:
        """
        Tells LLM to keep it under 2000 chars. We rely on the LLM to obey.
        """
        final_prompt = prompt + "\n\nPlease keep final answer under 2000 characters."
        model = await self.config.model()
        api_url = await self.config.api_url()

        payload = {
            "model": model,
            "messages": [{"role":"user","content":final_prompt}],
            "stream": False
        }
        headers = {"Content-Type": "application/json"}

        async with channel.typing():
            resp = requests.post(f"{api_url}/api/chat", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        text = data.get("message", {}).get("content","No valid response received.")
        return text.replace("\n\n","\n")

    ##############################################
    # Phase2 fallback approach if no relevant files
    ##############################################
    def fallback_filter_all_files(self, question: str) -> dict:
        """
        If the LLM returns no relevant files or invalid JSON, we fallback:
        - We treat each word in the question as a naive keyword
        - We search all .json files
        - Return a subset_map: { fname: [matching lines] }
        """
        words = re.sub(r"[^\w\s]", "", question.lower()).split()
        all_files = self.list_tag_files()
        subset_map = {}
        for f in all_files:
            lines = self.load_tag_file(f)
            # if no words, or question is empty, we skip
            if words:
                filtered = [line for line in lines if any(w in line.lower() for w in words)]
            else:
                filtered = []
            if filtered:
                subset_map[f] = filtered
        return subset_map

    ##############################################
    # run_two_phase with fallback
    ##############################################
    async def run_two_phase(self, question: str, channel: discord.abc.Messageable):
        """
        Phase 1 => LLM sees file list, picks files+keywords => if empty/invalid, fallback
        Phase 2 => either we use LLM's chosen subset or fallback subset => final answer
        """
        all_files = self.list_tag_files()
        if not all_files:
            await channel.send("No tag files exist. Use !learn or !llmknow first.")
            return

        # Phase1
        files_str = "\n".join(all_files)
        phase1_prompt = (
            "We have multiple JSON files containing specialized lines.\n"
            f"User question:\n{question}\n\n"
            "Possible files:\n"
            f"{files_str}\n\n"
            "Return JSON: { \"files\": [...], \"keywords\": { 'file.json': ['kw1','kw2'], ... } }.\n"
            "If none relevant => 'files':[], 'keywords':{}."
        )
        phase1_ans = await self.query_llm(phase1_prompt, channel)

        # parse out JSON
        pat = r"```json\s*(\{.*?\})\s*```"
        mt = re.search(pat, phase1_ans)
        raw_json = mt.group(1) if mt else phase1_ans.strip()

        chosen_files, keywords_map = [], {}
        try:
            data = json.loads(raw_json)
            chosen_files = data.get("files", [])
            keywords_map = data.get("keywords", {})
            if not isinstance(chosen_files, list):
                chosen_files=[]
            if not isinstance(keywords_map, dict):
                keywords_map={}
        except:
            pass

        final_files = [f for f in chosen_files if f in all_files]
        # if invalid or empty => fallback
        if not final_files:
            # fallback approach => scan all for naive matches
            subset_map = self.fallback_filter_all_files(question)
            if not subset_map:
                # truly no matches
                await channel.send("No relevant info found, even after fallback. Sorry!")
                return
            # now we do Phase2 prompt w/ subset_map
            fallback_prompt = (
                f"No valid Phase1 JSON or no relevant files. We used a fallback naive filter:\n"
                f"{json.dumps(subset_map, indent=2)}\n\n"
                f"Now answer the user's question:\n{question}"
            )
            final_ans = await self.query_llm(fallback_prompt, channel)
            await channel.send(final_ans)
            return

        # else normal route => Phase 2
        subset_map = {}
        for f in final_files:
            lines = self.load_tag_file(f)
            file_keys = keywords_map.get(f, [])
            if file_keys:
                lines = filter_lines_by_keywords(lines, file_keys)
            subset_map[f] = lines

        # if subset_map is empty => fallback as well
        all_empty = all((not subset_map[k]) for k in subset_map)
        if all_empty:
            # fallback naive approach
            subset_map = self.fallback_filter_all_files(question)
            if not subset_map:
                await channel.send("No relevant info found, even after fallback. Sorry!")
                return
            fallback_prompt = (
                f"LMM gave no lines, so we fallback naive filter:\n"
                f"{json.dumps(subset_map, indent=2)}\n\n"
                f"Now answer:\n{question}"
            )
            final_ans = await self.query_llm(fallback_prompt, channel)
            await channel.send(final_ans)
            return

        # normal final prompt
        phase2_prompt = (
            "Here are lines from the chosen files after filtering:\n"
            f"{json.dumps(subset_map, indent=2)}\n\n"
            f"Now answer the user's question:\n{question}"
        )
        final_ans = await self.query_llm(phase2_prompt, channel)
        await channel.send(final_ans)

    ##############################################
    # on_message => mention => run_two_phase
    ##############################################
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return
        if self.bot.user.mentioned_in(message):
            question = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
            if question:
                await self.run_two_phase(question, message.channel)

    ##############################################
    # !askllm => run_two_phase
    ##############################################
    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """
        Two-phase approach. If invalid or empty from LLM => fallback naive filter
        """
        await self.run_two_phase(question, ctx.channel)

    ##############################################
    # !learn => store plain text if user says yes
    ##############################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, tag: str, amount: int = 20):
        """
        Reads last <amount> messages => LLM produces plain text => 
        user can refine with no [instr], or store with yes
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

        fname = f"{tag.lower()}.json"
        old_lines = self.load_tag_file(fname)

        base_prompt = (
            f"Existing lines for tag '{tag}':\n"
            f"{json.dumps(old_lines, indent=2)}\n\n"
            "Below is a conversation. Summarize or produce new helpful info for this tag.\n"
            "Return plain text only, no JSON.\n"
            f"Conversation:\n{conversation}"
        )
        try:
            suggestion = await self.query_llm(base_prompt, ctx.channel)
        except Exception as e:
            return await ctx.send(f"Error calling LLM: {e}")

        await ctx.send(suggestion)
        await ctx.send("Type 'yes' to confirm, 'no [instructions]' to refine, or 'stop' to cancel.")

        def check(m: discord.Message):
            return (m.author == ctx.author and m.channel == ctx.channel)

        final_text = suggestion
        while True:
            try:
                reply = await self.bot.wait_for("message", check=check, timeout=180)
                text = reply.content.strip()
                lower = text.lower()

                if lower == "yes":
                    lines = self.load_tag_file(fname)
                    if final_text not in lines:
                        lines.append(final_text)
                        self.save_tag_file(fname, lines)
                        await ctx.send(f"Stored new info under '{tag.lower()}.json'.")
                    else:
                        await ctx.send("That text is already stored, skipping.")
                    break

                elif lower == "stop":
                    await ctx.send("Learning canceled, nothing stored.")
                    break

                elif lower.startswith("no"):
                    instructions = text[2:].strip()
                    refine_prompt = (
                        base_prompt
                        + "\n\nUser additional instructions:\n"
                        + instructions
                    )
                    try:
                        new_suggestion = await self.query_llm(refine_prompt, ctx.channel)
                        final_text = new_suggestion
                        await ctx.send(new_suggestion)
                        await ctx.send("Type 'yes' to confirm, 'no [instructions]' or 'stop'.")
                    except Exception as ee:
                        await ctx.send(f"Error refining: {ee}")

                else:
                    await ctx.send("Please type 'yes', 'no [instructions]', or 'stop'.")

            except Exception as ex:
                await ctx.send(f"Error or timeout: {ex}")
                break

    ##############################################
    # llmknowshow => triple backticks
    ##############################################
    @commands.command()
    async def llmknowshow(self, ctx):
        """
        Merges all .json tags, shows them with triple-backtick code formatting.
        """
        files = self.list_tag_files()
        if not files:
            return await ctx.send("No tags exist.")
        out = ""
        for f in sorted(files):
            tag = f.rsplit(".",1)[0]
            lines = self.load_tag_file(f)
            out += f"{tag}:\n"
            for i, val in enumerate(lines):
                out += f"  [{i}] {val}\n"
            out += "\n"

        if len(out) <= 2000:
            await ctx.send(f"```{out}```")
        else:
            await ctx.send("Knowledge too large for one message. Sorry!")

    ##############################################
    # llmknow => add single line
    ##############################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx, tag: str, *, info: str):
        """
        Adds <info> to <tag>.json, skipping duplicates.
        """
        fname = f"{tag.lower()}.json"
        lines = self.load_tag_file(fname)
        if info in lines:
            return await ctx.send("That info is already present, skipping.")
        lines.append(info)
        self.save_tag_file(fname, lines)
        await ctx.send(f"Added info to '{fname}'.")

    ##############################################
    # llmknowdelete => remove one line
    ##############################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdelete(self, ctx, tag: str, index: int):
        """
        Removes a single line from <tag>.json by index.
        """
        fname = f"{tag.lower()}.json"
        lines = self.load_tag_file(fname)
        if not lines:
            return await ctx.send("No lines found or tag doesn't exist.")
        if 0<=index<len(lines):
            removed = lines.pop(index)
            self.save_tag_file(fname, lines)
            await ctx.send(f"Deleted: {removed}")
        else:
            await ctx.send("Invalid index.")

    ##############################################
    # llmknowdeletetag => remove entire .json
    ##############################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx, tag: str):
        """
        Deletes <tag>.json entirely.
        """
        fname = f"{tag.lower()}.json"
        path = self.tags_folder / fname
        if not path.exists():
            return await ctx.send("Tag file not found.")
        path.unlink()
        await ctx.send(f"Deleted entire tag '{fname}'.")
