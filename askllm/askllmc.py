import discord
import os
import json
import re
import requests
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path

##########################################################
# A simple filter for lines by given keywords
##########################################################
def filter_lines_by_keywords(lines: list, keywords: list) -> list:
    """Returns lines that contain ANY of the keywords (case-insensitive)."""
    filtered = []
    for line in lines:
        low = line.lower()
        if any(kw.lower() in low for kw in keywords):
            filtered.append(line)
    return filtered

class LLMManager(commands.Cog):
    """
    Cog that:
    - Uses separate .json files for each tag in ./tags/
    - Two-phase approach for ask:
      Phase1: LLM sees the file list + user question => picks { files: [...], keywords:{...} }
      Phase2: We load & filter those lines => final answer.
    - Responds to both !askllm and bot mention with same approach.
    - Shows 'typing' for each phase, doesn't chunkify final answer.
    - Includes learn, llmknowshow, llmknow, etc. with triple-backtick for show.
    """

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9999999999)
        self.config.register_global(model="gemma3:12b", api_url="http://localhost:11434")

        self.tags_folder = cog_data_path(self) / "tags"
        if not self.tags_folder.exists():
            self.tags_folder.mkdir(parents=True, exist_ok=True)

    ##########################################################
    # Basic Model/API commands
    ##########################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmodel(self, ctx, model: str):
        await self.config.model.set(model)
        await ctx.send(f"LLM model set to '{model}'.")

    @commands.command()
    async def modellist(self, ctx):
        api_url = await self.config.api_url()
        try:
            resp = requests.get(f"{api_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
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
        url = url.rstrip("/")
        await self.config.api_url.set(url)
        await ctx.send(f"Ollama API URL set to '{url}'.")

    ##########################################################
    # Tag file utilities
    ##########################################################
    def list_tag_files(self) -> list:
        """Returns all .json files in the tags_folder."""
        files = []
        for fname in os.listdir(self.tags_folder):
            if fname.lower().endswith(".json"):
                files.append(fname)
        return files

    def load_tag_file(self, fname: str) -> list:
        """Loads fname as a list of lines, or returns [] if missing/corrupt."""
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
        path = self.tags_folder / fname
        with path.open("w", encoding="utf-8") as f:
            json.dump(lines, f, indent=2)

    ##########################################################
    # The LLM call with "typing" indicator, no chunkify
    ##########################################################
    async def query_llm(self, prompt: str, channel: discord.abc.Messageable) -> str:
        """
        We instruct LLM to keep final answer under 2000 chars.
        We'll show 'typing' in the given channel while calling the LLM.
        """
        final_prompt = prompt + "\n\nPlease keep your final answer under 2000 characters."
        model = await self.config.model()
        api_url = await self.config.api_url()

        payload = {
            "model": model,
            "messages": [{"role":"user","content":final_prompt}],
            "stream": False
        }
        headers = {"Content-Type":"application/json"}

        async with channel.typing():
            resp = requests.post(f"{api_url}/api/chat", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        text = data.get("message", {}).get("content", "No valid response received.")
        return text.replace("\n\n", "\n")

    ##########################################################
    # The Shared Two-Phase Logic
    ##########################################################
    async def run_two_phase(self, question: str, channel: discord.abc.Messageable):
        """
        The two-phase logic:
          Phase1 => LLM sees file list + question => returns JSON
          Phase2 => We load+filter lines, pass them + question => final answer
        Then we send final to 'channel' with no chunkify.
        """
        all_files = self.list_tag_files()
        if not all_files:
            await channel.send("No tag files exist. Please use !learn or !llmknow first.")
            return

        # Phase1 prompt
        files_str = "\n".join(all_files)
        phase1_prompt = (
            "We have multiple JSON files with lines of info.\n"
            f"User question:\n{question}\n\n"
            "List of possible files:\n"
            f"{files_str}\n\n"
            "Return JSON: { \"files\": [...], \"keywords\": { 'file.json': [ 'kw1','kw2'], ... } }.\n"
            "If none relevant => 'files':[], 'keywords':{}."
        )
        # We do the LLM call
        phase1_answer = await self.query_llm(phase1_prompt, channel)

        # parse JSON
        pat = r"```json\s*(\{.*?\})\s*```"
        mt = re.search(pat, phase1_answer)
        raw_json = mt.group(1) if mt else phase1_answer.strip()

        chosen_files, keywords_map = [], {}
        try:
            data = json.loads(raw_json)
            chosen_files = data.get("files", [])
            keywords_map = data.get("keywords", {})
            if not isinstance(chosen_files, list):
                chosen_files = []
            if not isinstance(keywords_map, dict):
                keywords_map = {}
        except:
            pass

        # filter nonexistent
        final_files = [f for f in chosen_files if f in all_files]
        if not final_files:
            await channel.send("No relevant files or invalid Phase1 JSON. Stopping.")
            return

        # Phase 2 => load+filter lines
        subset_map = {}
        for f in final_files:
            lines = self.load_tag_file(f)
            file_keys = keywords_map.get(f, [])
            if file_keys:
                lines = filter_lines_by_keywords(lines, file_keys)
            subset_map[f] = lines

        phase2_prompt = (
            "Here are lines from the chosen files after filtering:\n"
            f"{json.dumps(subset_map, indent=2)}\n\n"
            f"Now answer the user's question:\n{question}"
        )
        final_answer = await self.query_llm(phase2_prompt, channel)

        # send final answer
        await channel.send(final_answer)

    ##########################################################
    # on_message => mention => run_two_phase
    ##########################################################
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return
        if self.bot.user.mentioned_in(message):
            question = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
            if question:
                await self.run_two_phase(question, message.channel)

    ##########################################################
    # !askllm => same approach
    ##########################################################
    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """Runs the same two-phase approach as mention => run_two_phase."""
        await self.run_two_phase(question, ctx.channel)

    ##########################################################
    # !learn <tag> <amount>
    ##########################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, tag: str, amount: int=20):
        """
        Reads last <amount> msgs, LLM => { "solutions":[...] }, user says yes/no/stop
        """
        def not_command_or_bot(m: discord.Message):
            return (not m.author.bot) and (m.id != ctx.message.id)

        msgs = []
        async for msg in ctx.channel.history(limit=amount+5):
            if not_command_or_bot(msg):
                msgs.append(msg)
            if len(msgs)>=amount:
                break
        msgs.reverse()
        conversation = "\n".join(f"{m.author.name}: {m.content}" for m in msgs)

        fname = f"{tag.lower()}.json"
        old_data = self.load_tag_file(fname)

        base_prompt = (
            f"Existing solutions for tag '{tag}':\n"
            f"{json.dumps(old_data, indent=2)}\n\n"
            "Below is a conversation. Extract new or refined solutions for that tag.\n"
            "Output JSON: {\"solutions\":[\"fix1\",\"fix2\"]}\n\n"
            f"Conversation:\n{conversation}"
        )
        try:
            suggestion = await self.query_llm(base_prompt, ctx.channel)
        except Exception as e:
            return await ctx.send(f"Error: {e}")

        await ctx.send(suggestion)
        await ctx.send("Type 'yes' to confirm, 'no [instructions]' to refine, or 'stop' to cancel.")

        def check(m: discord.Message):
            return (m.author == ctx.author and m.channel == ctx.channel)

        while True:
            try:
                reply = await self.bot.wait_for("message", check=check, timeout=180)
                text = reply.content.strip().lower()
                if text=="yes":
                    out = self.store_learn_solutions(fname, suggestion)
                    await ctx.send(out)
                    break
                elif text=="stop":
                    await ctx.send("Learning canceled.")
                    break
                elif text.startswith("no"):
                    instructions = reply.content[2:].strip()
                    refine_prompt = base_prompt + "\nAdditional instructions:\n" + instructions
                    try:
                        suggestion = await self.query_llm(refine_prompt, ctx.channel)
                        await ctx.send(suggestion)
                        await ctx.send("Type 'yes' to confirm, 'no [instructions]' or 'stop' to cancel.")
                    except Exception as ee:
                        await ctx.send(f"Error: {ee}")
                else:
                    await ctx.send("Please type 'yes', 'no [instructions]', or 'stop'.")

            except Exception as ex:
                await ctx.send(f"Error or timeout: {ex}")
                break

    def store_learn_solutions(self, fname: str, suggestion: str) -> str:
        pat = r"```json\s*(\{.*?\})\s*```"
        mt = re.search(pat, suggestion)
        raw = suggestion
        if mt:
            raw = mt.group(1)
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("parsed JSON not a dict")
            arr = data.get("solutions", [])
            if not isinstance(arr, list):
                raise ValueError("No valid 'solutions' array")

            old = self.load_tag_file(fname)
            changed=0
            for fix in arr:
                if fix not in old:
                    old.append(fix)
                    changed+=1
            self.save_tag_file(fname, old)
            return f"Added {changed} new solution(s) to '{fname}'."

        except Exception:
            # fallback => general
            gf = "general.json"
            gf_data = self.load_tag_file(gf)
            if suggestion not in gf_data:
                gf_data.append(suggestion)
                self.save_tag_file(gf, gf_data)
                return f"Invalid JSON; entire text stored in '{gf}'."
            else:
                return f"Invalid JSON; already in '{gf}'."

    ##########################################################
    # !llmknowshow => merges all tags with triple backticks
    ##########################################################
    @commands.command()
    async def llmknowshow(self, ctx):
        """Merge all .json tags, display them as code."""
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
        if len(out)<=2000:
            await ctx.send(f"```{out}```")
        else:
            await ctx.send("Knowledge too large for a single Discord message. Sorry.")

    ##########################################################
    # !llmknow <tag> <info>
    ##########################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx, tag: str, *, info: str):
        fname = f"{tag.lower()}.json"
        lines = self.load_tag_file(fname)
        if info in lines:
            return await ctx.send("That info already exists, skipping.")
        lines.append(info)
        self.save_tag_file(fname, lines)
        await ctx.send(f"Added info to '{fname}'.")

    ##########################################################
    # !llmknowdelete <tag> <index>
    ##########################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdelete(self, ctx, tag: str, index: int):
        fname = f"{tag.lower()}.json"
        lines = self.load_tag_file(fname)
        if not lines:
            return await ctx.send("No lines or tag file doesn't exist.")
        if 0<=index<len(lines):
            removed = lines.pop(index)
            self.save_tag_file(fname, lines)
            await ctx.send(f"Deleted: {removed}")
        else:
            await ctx.send("Invalid index.")

    ##########################################################
    # !llmknowdeletetag <tag>
    ##########################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx, tag: str):
        fname = f"{tag.lower()}.json"
        path = self.tags_folder / fname
        if not path.exists():
            return await ctx.send("Tag file not found.")
        path.unlink()
        await ctx.send(f"Deleted entire tag '{fname}'.")
