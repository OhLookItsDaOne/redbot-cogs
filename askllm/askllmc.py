import discord
import os
import json
import re
import requests
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path

def filter_lines_by_keywords(lines: list, keywords: list) -> list:
    """Returns only lines that contain ANY of the given keywords (case-insensitive)."""
    filtered = []
    for line in lines:
        low = line.lower()
        if any(kw.lower() in low for kw in keywords):
            filtered.append(line)
    return filtered

class LLMManager(commands.Cog):
    """
    Revised two-phase code:
    - Shows 'typing' for LLM calls
    - 'no [instructions]' appends to existing conversation
    - !llmknowshow uses triple backticks
    """

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9999999999)
        self.config.register_global(model="gemma3:12b", api_url="http://localhost:11434")

        self.tags_folder = cog_data_path(self) / "tags"
        if not self.tags_folder.exists():
            self.tags_folder.mkdir(parents=True, exist_ok=True)

    #############################################
    # Basic Model/API
    #############################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmodel(self, ctx, model: str):
        await self.config.model.set(model)
        await ctx.send(f"LLM model set to '{model}'.")

    @commands.command()
    async def modellist(self, ctx):
        api_url = await self.config.api_url()
        try:
            r = requests.get(f"{api_url}/api/tags")
            r.raise_for_status()
            data = r.json()
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
        url = url.rstrip("/")
        await self.config.api_url.set(url)
        await ctx.send(f"Ollama API URL set to '{url}'.")

    #############################################
    # Tag file utilities
    #############################################
    def list_tag_files(self) -> list:
        """Returns all .json files from the tags/ folder."""
        files = []
        for fname in os.listdir(self.tags_folder):
            if fname.lower().endswith(".json"):
                files.append(fname)
        return files

    def load_tag_file(self, fname: str) -> list:
        """Loads a single .json (a list of lines) or returns [] if not found/corrupt."""
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

    #############################################
    # LLM query
    #############################################
    async def query_llm(self, prompt: str) -> str:
        """
        Instruct LLM to remain concise. If it disobeys, might exceed Discord’s limit.
        We show 'typing' while calling the LLM.
        """
        final_prompt = prompt + "\n\nPlease keep final answer under 2000 characters."

        model = await self.config.model()
        api_url = await self.config.api_url()

        payload = {
            "model": model,
            "messages": [{"role":"user","content":final_prompt}],
            "stream": False
        }
        headers = {"Content-Type":"application/json"}

        resp = requests.post(f"{api_url}/api/chat", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("message", {}).get("content","No valid response received.")
        return text.replace("\n\n","\n")

    #############################################
    # !llmknowshow => merges all tags, triple backticks
    #############################################
    @commands.command()
    async def llmknowshow(self, ctx):
        """Display all .json tags in code fences."""
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

        # put the entire thing into triple backticks
        if len(out) <= 2000:
            await ctx.send(f"```{out}```")
        else:
            await ctx.send("Knowledge is too large to fit in one Discord message. Sorry.")

    #############################################
    # !llmknow <tag> => add single line
    #############################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx, tag: str, *, info: str):
        """
        Adds <info> to <tag>.json, skipping duplicates.
        Creates the file if missing.
        """
        fname = f"{tag.lower()}.json"
        lines = self.load_tag_file(fname)
        if info in lines:
            return await ctx.send("That info already exists. Skipping.")
        lines.append(info)
        self.save_tag_file(fname, lines)
        await ctx.send(f"Added info to {fname}.")

    #############################################
    # !llmknowdelete <tag> <index>
    #############################################
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
        if 0 <= index < len(lines):
            removed = lines.pop(index)
            self.save_tag_file(fname, lines)
            await ctx.send(f"Deleted: {removed}")
        else:
            await ctx.send("Invalid index.")

    #############################################
    # !llmknowdeletetag <tag>
    #############################################
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

    #############################################
    # !learn <tag> <amount> => store solutions
    #############################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, tag: str, amount: int=20):
        """
        Reads last <amount> messages, calls LLM for { "solutions":[...] },
        user can say 'no [instructions]' to add new data to the final JSON,
        or 'yes' to confirm storing them in <tag>.json
        """
        def not_command_or_bot(m: discord.Message):
            if m.author.bot:
                return False
            if m.id == ctx.message.id:
                return False
            return True

        # gather last messages
        msgs = []
        async for msg in ctx.channel.history(limit=amount+5):
            if not_command_or_bot(msg):
                msgs.append(msg)
            if len(msgs)>=amount:
                break
        msgs.reverse()
        conversation = "\n".join(f"{m.author.name}: {m.content}" for m in msgs)

        # load existing lines
        fname = f"{tag.lower()}.json"
        old_lines = self.load_tag_file(fname)

        base_prompt = (
            f"Existing solutions for tag '{tag}':\n"
            f"{json.dumps(old_lines, indent=2)}\n\n"
            "Below is a conversation. Extract new or refined solutions for that tag.\n"
            "Output JSON: { \"solutions\":[\"fix1\",\"fix2\"] }\n"
            "No user mentions/logs.\n\n"
            f"Conversation:\n{conversation}"
        )

        async with ctx.typing():
            suggestion = await self.query_llm(base_prompt)

        # no code fencing, no chunkify
        await ctx.send(suggestion)

        await ctx.send("Type 'yes' to confirm, 'no [instructions]' to refine, or 'stop' to cancel.")

        def check(m: discord.Message):
            return (m.author == ctx.author) and (m.channel == ctx.channel)

        # We store the entire conversation in `base_prompt`.
        # If user says "no [instructions]", we append instructions + re-ask the LLM
        conversation_text = suggestion

        while True:
            try:
                reply = await self.bot.wait_for("message", check=check, timeout=180)
                text = reply.content.strip()
                low = text.lower()
                if low == "yes":
                    # parse & store
                    msg = self.store_learn_solutions(fname, suggestion)
                    await ctx.send(msg)
                    break

                elif low == "stop":
                    await ctx.send("Learning canceled.")
                    break

                elif low.startswith("no"):
                    instructions = text[2:].strip()
                    refine_prompt = base_prompt + "\nAdditional instructions:\n" + instructions

                    async with ctx.typing():
                        suggestion = await self.query_llm(refine_prompt)

                    await ctx.send(suggestion)
                    await ctx.send("Type 'yes' to confirm, 'no [instructions]' or 'stop' to cancel.")

                else:
                    await ctx.send("Please type 'yes', 'no [instructions]', or 'stop'.")

            except Exception as e:
                await ctx.send(f"Error or timeout: {e}")
                break

    def store_learn_solutions(self, fname: str, suggestion: str) -> str:
        """
        Looks for { "solutions": [ ... ] } in the LLM's suggestion,
        appends them to <fname>, skipping duplicates.
        If invalid JSON, store entire text in general.json
        """
        pattern = r"(?s)```json\s*(\{.*?\})\s*```"
        match = re.search(pattern, suggestion)
        raw = suggestion
        if match:
            raw = match.group(1)

        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Parsed JSON is not a dict.")
            arr = data.get("solutions", [])
            if not isinstance(arr, list):
                raise ValueError("No valid 'solutions' array found.")
            
            old = self.load_tag_file(fname)
            changed=0
            for fix in arr:
                if fix not in old:
                    old.append(fix)
                    changed+=1
            self.save_tag_file(fname, old)
            return f"Added {changed} new solution(s) to '{fname}'."

        except Exception:
            # fallback -> store entire text in general.json
            gf = "general.json"
            gf_lines = self.load_tag_file(gf)
            if suggestion not in gf_lines:
                gf_lines.append(suggestion)
                self.save_tag_file(gf, gf_lines)
                return f"Invalid JSON; entire text stored in '{gf}'."
            else:
                return f"Invalid JSON; already in '{gf}'."
