import discord
import os
import json
import re
import requests
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path

def filter_lines_by_keywords(lines: list, keywords: list) -> list:
    """
    Returns only those lines that contain ANY of the given keywords (case-insensitive).
    """
    filtered = []
    for line in lines:
        low = line.lower()
        if any(kw.lower() in low for kw in keywords):
            filtered.append(line)
    return filtered

class LLMManager(commands.Cog):
    """
    Final Cog:
    - on_message => mention => two-phase approach
    - !askllm => same two-phase approach
    - no chunkify in final answers
    - !learn => plain text storage (no JSON from LLM)
    - all other commands: setmodel, setapi, modellist, llmknowshow, llmknow, llmknowdelete, llmknowdeletetag
    """

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9999999999)
        self.config.register_global(model="gemma3:12b", api_url="http://localhost:11434")

        self.tags_folder = cog_data_path(self) / "tags"
        if not self.tags_folder.exists():
            self.tags_folder.mkdir(parents=True, exist_ok=True)

    ############################################################
    # Basic Model/API commands
    ############################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmodel(self, ctx, model: str):
        """
        Sets the default LLM model to 'model'.
        """
        await self.config.model.set(model)
        await ctx.send(f"LLM model set to '{model}'.")

    @commands.command()
    async def modellist(self, ctx):
        """
        Lists models from the Ollama instance.
        """
        api_url = await self.config.api_url()
        try:
            resp = requests.get(f"{api_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            if models:
                await ctx.send(f"Available models: {', '.join(models)}")
            else:
                await ctx.send("No models found on this Ollama instance.")
        except Exception as e:
            await ctx.send(f"Error: {e}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setapi(self, ctx, url: str):
        """
        Sets the Ollama API URL.
        """
        url = url.rstrip("/")
        await self.config.api_url.set(url)
        await ctx.send(f"Ollama API URL set to '{url}'.")

    ############################################################
    # Tag file utilities
    ############################################################
    def list_tag_files(self) -> list:
        """
        Returns all .json files in the 'tags' folder.
        """
        all_files = []
        for fname in os.listdir(self.tags_folder):
            if fname.lower().endswith(".json"):
                all_files.append(fname)
        return all_files

    def load_tag_file(self, fname: str) -> list:
        """
        Loads a .json file as a list of lines, or [] if not found/corrupt.
        """
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
        """
        Saves a list of lines to fname in JSON format.
        """
        path = self.tags_folder / fname
        with path.open("w", encoding="utf-8") as f:
            json.dump(lines, f, indent=2)

    ############################################################
    # LLM query with "typing" indicator, no chunkify
    ############################################################
    async def query_llm(self, prompt: str, channel: discord.abc.Messageable) -> str:
        """
        Calls Ollama with a final_prompt that instructs the LLM to remain under 2000 chars.
        We show 'typing' during the call. 
        """
        final_prompt = prompt + "\n\nPlease keep final answer under 2000 characters."
        model = await self.config.model()
        api_url = await self.config.api_url()

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": final_prompt}],
            "stream": False
        }
        headers = {"Content-Type": "application/json"}

        async with channel.typing():
            resp = requests.post(f"{api_url}/api/chat", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        text = data.get("message", {}).get("content", "No valid response received.")
        return text.replace("\n\n", "\n")

    ############################################################
    # Two-phase logic for mention or !askllm
    ############################################################
    async def run_two_phase(self, question: str, channel: discord.abc.Messageable):
        """
        Phase 1 => LLM sees file list, picks files & keywords => JSON
        Phase 2 => load & filter lines, final answer
        No chunkify, we rely on the LLM to be short.
        """
        all_files = self.list_tag_files()
        if not all_files:
            await channel.send("No tag files exist. Use !learn or !llmknow first.")
            return

        # PHASE 1
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

        # parse
        pat = r"```json\s*(\{.*?\})\s*```"
        mt = re.search(pat, phase1_ans)
        raw_json = mt.group(1) if mt else phase1_ans.strip()

        chosen_files = []
        keywords_map = {}
        try:
            data = json.loads(raw_json)
            chosen_files = data.get("files", [])
            keywords_map = data.get("keywords", {})
            if not isinstance(chosen_files, list):
                chosen_files = []
            if not isinstance(keywords_map, dict):
                keywords_map={}
        except:
            pass

        final_files = [f for f in chosen_files if f in all_files]
        if not final_files:
            await channel.send("No relevant files or invalid Phase1 JSON. Stopping.")
            return

        # PHASE 2
        subset_map = {}
        for f in final_files:
            lines = self.load_tag_file(f)
            kwds = keywords_map.get(f, [])
            if kwds:
                lines = filter_lines_by_keywords(lines, kwds)
            subset_map[f] = lines

        phase2_prompt = (
            "Here are lines from the chosen files after filtering:\n"
            f"{json.dumps(subset_map, indent=2)}\n\n"
            f"Now answer the user's question:\n{question}"
        )
        final_ans = await self.query_llm(phase2_prompt, channel)
        await channel.send(final_ans)

    ############################################################
    # on_message => mention => run_two_phase
    ############################################################
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return
        if self.bot.user.mentioned_in(message):
            question = message.content.replace(f"<@{self.bot.user.id}>", "").strip()
            if question:
                await self.run_two_phase(question, message.channel)

    ############################################################
    # !askllm => run_two_phase
    ############################################################
    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """
        Two-phase approach for user question => picks files & keywords => final answer
        """
        await self.run_two_phase(question, ctx.channel)

    ############################################################
    # !learn <tag> <amount> => plain text approach
    ############################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, tag: str, amount: int = 20):
        """
        1) Reads last <amount> msgs
        2) LLM returns plain text (not JSON!)
        3) If user says 'no [instructions]', we refine prompt
        4) If user says 'yes', we store final text under <tag>.json
        5) If user says 'stop', do nothing
        """
        def not_command_or_bot(m: discord.Message):
            if m.author.bot: return False
            if m.id == ctx.message.id: return False
            return True

        msgs=[]
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
            return await ctx.send(f"Error: {e}")

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

                if lower=="yes":
                    lines = self.load_tag_file(fname)
                    if final_text not in lines:
                        lines.append(final_text)
                        self.save_tag_file(fname, lines)
                        await ctx.send(f"Stored new info under '{tag.lower()}.json'.")
                    else:
                        await ctx.send("That text is already stored, skipping.")
                    break

                elif lower=="stop":
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

    ############################################################
    # !llmknowshow => merges all tags with triple-backtick
    ############################################################
    @commands.command()
    async def llmknowshow(self, ctx):
        """
        Merges all .json tags, displays them in triple backticks
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
            await ctx.send("Knowledge too large to fit in one message. Sorry!")

    ############################################################
    # !llmknow <tag> "info"
    ############################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx, tag: str, *, info: str):
        """
        Adds <info> to <tag>.json, skipping duplicates.
        """
        fname = f"{tag.lower()}.json"
        lines = self.load_tag_file(fname)
        if info in lines:
            return await ctx.send("That info already exists; skipping.")
        lines.append(info)
        self.save_tag_file(fname, lines)
        await ctx.send(f"Added info to '{fname}'.")

    ############################################################
    # !llmknowdelete <tag> <index>
    ############################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdelete(self, ctx, tag: str, index: int):
        """
        Deletes a single line from <tag>.json by index.
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

    ############################################################
    # !llmknowdeletetag <tag>
    ############################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx, tag: str):
        """
        Deletes the entire <tag>.json file
        """
        fname = f"{tag.lower()}.json"
        path = self.tags_folder / fname
        if not path.exists():
            return await ctx.send("Tag file not found.")
        path.unlink()
        await ctx.send(f"Deleted entire tag file '{fname}'.")
