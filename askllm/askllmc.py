import discord
import os
import json
import re
import requests
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path

##################################
# Helper: chunkify
##################################
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

##################################
# Helper: naive keyword filter
##################################
def filter_lines_by_keywords(lines: list, keywords: list) -> list:
    """
    Returns only those lines that contain ANY of the keywords (case-insensitive).
    """
    filtered = []
    for line in lines:
        low = line.lower()
        if any(kw.lower() in low for kw in keywords):
            filtered.append(line)
    return filtered

class LLMManager(commands.Cog):
    """
    A Cog that stores each 'tag' in a separate JSON file in ./tags,
    then performs a two-phase approach for askllm:

    PHASE 1 -> "Which files/keywords are relevant?"
    PHASE 2 -> Filter those files for those keywords, pass lines to LLM.
    """

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9999999999)
        self.config.register_global(model="gemma3:12b", api_url="http://localhost:11434")

        self.tags_folder = cog_data_path(self) / "tags"
        if not self.tags_folder.exists():
            self.tags_folder.mkdir(parents=True, exist_ok=True)

    ##################################
    # Basic model/API commands
    ##################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmodel(self, ctx, model: str):
        await self.config.model.set(model)
        await ctx.send(f"LLM model set to {model}")

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
                await ctx.send("No models found.")
        except Exception as e:
            await ctx.send(f"Error: {e}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setapi(self, ctx, url: str):
        url = url.rstrip("/")
        await self.config.api_url.set(url)
        await ctx.send(f"Ollama API URL set to {url}")

    ##################################
    # Tag file utilities
    ##################################
    def list_tag_files(self) -> list:
        """All .json files in tags_folder, e.g. ['crash.json','enb.json']."""
        files = []
        for fname in os.listdir(self.tags_folder):
            if fname.lower().endswith(".json"):
                files.append(fname)
        return files

    def load_tag_file(self, fname: str) -> list:
        """Loads a single file (list of lines), or empty list if not found/invalid."""
        path = self.tags_folder / fname
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    return []
        except:
            return []

    def save_tag_file(self, fname: str, lines: list):
        path = self.tags_folder / fname
        with path.open("w", encoding="utf-8") as f:
            json.dump(lines, f, indent=2)

    ##################################
    # LLM call
    ##################################
    async def query_llm(self, prompt: str) -> str:
        model = await self.config.model()
        api_url = await self.config.api_url()

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        headers = {"Content-Type":"application/json"}
        r = requests.post(f"{api_url}/api/chat", json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        text = data.get("message",{}).get("content","No valid response received.")
        # remove double blank lines
        return text.replace("\n\n","\n")

    ##################################
    # The 2-phase logic for askllm
    ##################################
    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """
        PHASE 1: Send list of existing files + user question -> LLM suggests JSON: { "files": [...], "keywords": {...} }
        PHASE 2: For each file in "files", read lines, filter by the "keywords[file]", unify -> final prompt
        Return final answer chunked if needed
        """
        # 1) Gather the tag filenames
        all_files = self.list_tag_files()
        if not all_files:
            return await ctx.send("No tag files. Please create some with !llmknow or !learn first.")

        files_str = "\n".join(all_files)
        phase1_prompt = (
            "We have multiple tag files, each containing relevant lines of text.\n"
            "The user question is:\n"
            f"{question}\n\n"
            "Here is the list of possible files:\n"
            f"{files_str}\n\n"
            "Step: Propose which files might contain relevant info. Also propose what keywords to look for in each file. Return valid JSON with structure:\n"
            "```json\n"
            "{\n"
            "  \"files\": [ \"somefile.json\", ... ],\n"
            "  \"keywords\": {\n"
            "      \"somefile.json\": [\"keyword1\", \"keyword2\"],\n"
            "      ...\n"
            "  }\n"
            "}\n"
            "If no relevant files, use \"files\":[], \"keywords\":{}.\n"
            "Don't guess about nonexistent files, only pick from the given file list.\n"
            "End your response with the JSON block. Nothing else.\n"
            "```\n"
        )

        # 2) Query LLM for file+keyword suggestions
        try:
            async with ctx.typing():
                phase1_answer = await self.query_llm(phase1_prompt)
        except Exception as e:
            return await ctx.send(f"Error in phase1: {e}")

        # We'll parse out the JSON block
        pat = r"(?s)```json\s*(\{.*?\})\s*```"
        mt = re.search(pat, phase1_answer)
        raw_json = None
        if mt:
            raw_json = mt.group(1)
        else:
            # fallback: maybe it returned raw JSON without triple-backticks
            # or we can't find any. We'll use entire message
            # or treat it as empty
            raw_json = phase1_answer.strip()

        # parse
        chosen_files = []
        keywords_map = {}
        try:
            data = json.loads(raw_json)
            if not isinstance(data, dict):
                raise ValueError("parsed JSON not a dict")

            chosen_files = data.get("files", [])
            if not isinstance(chosen_files, list):
                chosen_files = []
            keywords_map = data.get("keywords", {})
            if not isinstance(keywords_map, dict):
                keywords_map = {}
        except Exception:
            # if parse fails, default to none
            pass

        # filter out any files that don't exist
        final_files = []
        for f in chosen_files:
            if f in all_files:
                final_files.append(f)
        if not final_files:
            # no relevant files
            return await ctx.send("Phase1 LLM said no relevant files or invalid parse. No data to check.")

        # 3) Phase 2: For each file in final_files, load lines, filter by keywords_map[file]
        # unify them into a small subset. Then final prompt with user question
        subset_map = {}
        for f in final_files:
            lines = self.load_tag_file(f)
            # if LLM gave a list of keywords for that file
            file_keywords = keywords_map.get(f, [])
            if not file_keywords:
                # if no keywords given, we might either skip or take all lines
                # let's take all lines if no keywords are specified
                subset_map[f] = lines
            else:
                # filter by those keywords
                filtered = filter_lines_by_keywords(lines, file_keywords)
                subset_map[f] = filtered

        # unify into final dictionary for prompt
        # e.g. { "crash.json": ["line1", "line2"], "enb.json": [...] }
        # then pass it to LLM with the question
        phase2_prompt = (
            "We found these lines from the chosen files, after filtering by the suggested keywords.\n\n"
            f"{json.dumps(subset_map, indent=2)}\n\n"
            "Now please answer the user's question:\n"
            f"{question}\n\n"
            "If you see no relevant info here for the question, say so. Otherwise, give a best possible answer."
        )
        try:
            async with ctx.typing():
                final_answer = await self.query_llm(phase2_prompt)
        except Exception as e:
            return await ctx.send(f"Error in phase2: {e}")

        # chunkify final
        if len(final_answer)<=2000:
            await ctx.send(f"```\n{final_answer}\n```")
        else:
            parts = chunkify(final_answer, 1900)
            for i, c in enumerate(parts,1):
                await ctx.send(f"**(Part {i}/{len(parts)}):**\n```\n{c}\n```")

    ############################################################
    # The user’s other commands: !learn, !llmknowshow, etc
    ############################################################
    @commands.command()
    async def llmknowshow(self, ctx):
        """
        Similar to the old approach: merges all .json tag files,
        shows them with each line indexed.
        """
        all_files = self.list_tag_files()
        if not all_files:
            return await ctx.send("No tag files found.")
        out = ""
        for f in sorted(all_files):
            tag = f.rsplit(".",1)[0]
            lines = self.load_tag_file(f)
            out += f"{tag}:\n"
            for i, l in enumerate(lines):
                out += f"  [{i}] {l}\n"
            out += "\n"
        if len(out)<=2000:
            await ctx.send(f"```\n{out}\n```")
        else:
            cparts = chunkify(out,1900)
            for idx, cp in enumerate(cparts,1):
                await ctx.send(f"**(Part {idx}/{len(cparts)}):**\n```\n{cp}\n```")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx, tag: str, *, info: str):
        """
        Adds 'info' to <tag>.json, skipping duplicates
        """
        fname = f"{tag.lower()}.json"
        lines = self.load_tag_file(fname)
        if info in lines:
            return await ctx.send("That info already exists, skipping.")
        lines.append(info)
        self.save_tag_file(fname, lines)
        await ctx.send(f"Added info to {fname}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdelete(self, ctx, tag: str, index: int):
        """
        Removes one line from <tag>.json
        """
        fname = f"{tag.lower()}.json"
        lines = self.load_tag_file(fname)
        if not lines:
            return await ctx.send("No lines or tag doesn't exist.")
        if 0<=index<len(lines):
            removed = lines.pop(index)
            self.save_tag_file(fname, lines)
            await ctx.send(f"Deleted: {removed}")
        else:
            await ctx.send("Invalid index")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx, tag: str):
        """
        Removes <tag>.json entirely
        """
        fname = f"{tag.lower()}.json"
        path = self.tags_folder / fname
        if not path.exists():
            return await ctx.send("Tag not found.")
        path.unlink()
        await ctx.send(f"Deleted entire tag file: {fname}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, tag: str, amount: int=20):
        """
        Reads last <amount> messages, sends them to LLM to produce JSON with "solutions",
        appends them to <tag>.json
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
        convo = "\n".join(f"{m.author.name}: {m.content}" for m in msgs)

        fname = f"{tag.lower()}.json"
        old_data = self.load_tag_file(fname)

        prompt = (
            f"Existing solutions for tag '{tag}':\n"
            f"{json.dumps(old_data, indent=2)}\n\n"
            "Below is a conversation. Extract new or refined solutions for that tag.\n"
            "Output valid JSON like {\"solutions\":[\"fix1\",\"fix2\"]}.\n\n"
            f"Conversation:\n{convo}"
        )
        try:
            async with ctx.typing():
                suggestion = await self.query_llm(prompt)
        except Exception as e:
            return await ctx.send(f"Error: {e}")

        if len(suggestion)<=2000:
            await ctx.send(f"```\n{suggestion}\n```")
        else:
            splitted = chunkify(suggestion,1900)
            for i, spt in enumerate(splitted,1):
                await ctx.send(f"**(Part {i}/{len(splitted)}):**\n```\n{spt}\n```")

        await ctx.send("Type 'yes' to confirm, 'no [instr]' to refine, or 'stop' to cancel.")

        def check(m: discord.Message):
            return (m.author==ctx.author) and (m.channel==ctx.channel)

        while True:
            try:
                user_msg = await self.bot.wait_for("message", check=check, timeout=120)
                low = user_msg.content.strip().lower()
                if low=="yes":
                    msg = self.store_learn_solutions(fname, suggestion)
                    await ctx.send(msg)
                    break
                elif low=="stop":
                    await ctx.send("Learning cancelled.")
                    break
                elif low.startswith("no"):
                    instructions = user_msg.content[2:].strip()
                    refine_prompt = prompt + "\nAdditional instructions:\n" + instructions
                    try:
                        async with ctx.typing():
                            suggestion = await self.query_llm(refine_prompt)
                        if len(suggestion)<=2000:
                            await ctx.send(f"```\n{suggestion}\n```")
                        else:
                            splitted2 = chunkify(suggestion,1900)
                            for ix, partt in enumerate(splitted2,1):
                                await ctx.send(f"**(Part {ix}/{len(splitted2)}):**\n```\n{partt}\n```")
                        await ctx.send("Type 'yes' to confirm, 'no [instr]' to refine, or 'stop'.")
                    except Exception as ee:
                        await ctx.send(f"Error: {ee}")
                else:
                    await ctx.send("Please type 'yes', 'no [instructions]', or 'stop'.")
            except Exception as e:
                await ctx.send(f"Error or timeout: {e}")
                break

    def store_learn_solutions(self, filename: str, suggestion: str) -> str:
        """
        Expects something like:
        ```json
        { "solutions": ["some fix","another fix"] }
        ```
        We'll parse & append to <filename>, skipping duplicates.
        """
        pat = r"(?s)```json\s*(\{.*?\})\s*```"
        mt = re.search(pat, suggestion)
        raw = suggestion
        if mt:
            raw = mt.group(1)
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Not a dict")
            arr = data.get("solutions", [])
            if not isinstance(arr, list):
                raise ValueError("No valid 'solutions' array")
            old = self.load_tag_file(filename)
            changed = 0
            for fix in arr:
                if fix not in old:
                    old.append(fix)
                    changed+=1
            self.save_tag_file(filename, old)
            return f"Added {changed} new solutions to {filename}"
        except Exception:
            # fallback store entire suggestion in 'general.json'
            gf = "general.json"
            gf_data = self.load_tag_file(gf)
            if suggestion not in gf_data:
                gf_data.append(suggestion)
                self.save_tag_file(gf, gf_data)
                return f"Not valid JSON. Entire text stored in {gf}"
            else:
                return f"Not valid JSON. Already in {gf}."
