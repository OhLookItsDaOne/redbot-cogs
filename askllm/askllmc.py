import discord
import os
import json
import re
import requests
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path

############################################
# Helper: keyword-based filtering
############################################
def filter_lines_by_keywords(lines: list, keywords: list) -> list:
    """Returns only lines that contain ANY of the keywords (case-insensitive)."""
    filtered = []
    for line in lines:
        low = line.lower()
        if any(kw.lower() in low for kw in keywords):
            filtered.append(line)
    return filtered

class LLMManager(commands.Cog):
    """
    Two-phase approach to askllm, with prints for debugging:
    1) Phase 1 -> LLM sees a list of existing JSON filenames, plus user's question
                  -> returns JSON specifying which files & keywords
    2) Code loads & filters lines from those files -> Phase 2 prompt for final answer.
    No chunkify for final answer, just a single message – we instruct LLM to keep it short.
    """

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9999999999)
        self.config.register_global(model="gemma3:12b", api_url="http://localhost:11434")

        self.tags_folder = cog_data_path(self) / "tags"
        if not self.tags_folder.exists():
            self.tags_folder.mkdir(parents=True, exist_ok=True)

    ########################################################
    # Basic Model/API commands
    ########################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmodel(self, ctx, model: str):
        await self.config.model.set(model)
        await ctx.send(f"LLM model set to {model}")

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
                await ctx.send("No models found.")
        except Exception as e:
            await ctx.send(f"Error: {e}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setapi(self, ctx, url: str):
        url = url.rstrip("/")
        await self.config.api_url.set(url)
        await ctx.send(f"Ollama API URL set to {url}")

    ########################################################
    # Tag File Management
    ########################################################
    def list_tag_files(self) -> list:
        """Returns a list of all .json files in 'tags/' folder."""
        all_files = []
        for fname in os.listdir(self.tags_folder):
            if fname.lower().endswith(".json"):
                all_files.append(fname)
        return all_files

    def load_tag_file(self, fname: str) -> list:
        """Loads a single file's JSON as a list of lines, or empty if missing/corrupt."""
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

    ########################################################
    # LLM query (with debug prints, no chunkify)
    ########################################################
    async def query_llm(self, prompt: str) -> str:
        """
        We instruct the LLM to keep the answer short (<2000 chars).
        Print the prompt to console for debugging.
        """
        final_prompt = (
            prompt
            + "\n\n"
            "IMPORTANT: Keep final answer concise (under 2000 characters)."
        )

        print("=== LLM Prompt ===")
        print(final_prompt)
        print("==================\n")

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
        text = data.get("message",{}).get("content","No valid response received.")
        return text.replace("\n\n","\n")

    ########################################################
    # The askllm command: 2-phase approach
    ########################################################
    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """
        Phase 1 -> LLM sees [filenames, question], returns { 'files': [...], 'keywords': {...} }
        Phase 2 -> code loads+filters lines from those files, merges them, final prompt -> LLM
        No chunkify: single reply. We rely on the LLM to keep it short.
        Print debug info to console.
        """
        all_files = self.list_tag_files()
        if not all_files:
            return await ctx.send("No tag files found. Please create some with !learn or !llmknow first.")

        # Phase 1 prompt
        files_str = "\n".join(all_files)
        phase1_prompt = (
            "We have multiple JSON files, each with specialized lines.\n"
            "The user question:\n"
            f"{question}\n\n"
            "File list:\n"
            f"{files_str}\n\n"
            "Propose which files are relevant, plus keywords to filter those files. Return JSON:\n"
            "```json\n"
            "{\n"
            "  \"files\": [ \"somefile.json\", ... ],\n"
            "  \"keywords\": {\n"
            "    \"somefile.json\": [\"keyword1\",\"keyword2\"],\n"
            "    ...\n"
            "  }\n"
            "}\n"
            "```\n"
            "If no relevant files, 'files':[], 'keywords':{}.\n"
        )

        # PHASE 1: LLM decides
        try:
            async with ctx.typing():
                phase1_answer = await self.query_llm(phase1_prompt)
        except Exception as e:
            return await ctx.send(f"Error in Phase1: {e}")

        print("=== PHASE1 LLM raw ===")
        print(phase1_answer)
        print("======================\n")

        pat = r"(?s)```json\s*(\{.*?\})\s*```"
        mt = re.search(pat, phase1_answer)
        raw_json = mt.group(1) if mt else phase1_answer.strip()

        chosen_files = []
        keywords_map = {}
        try:
            data = json.loads(raw_json)
            if not isinstance(data, dict):
                raise ValueError("parsed JSON not a dict.")
            chosen_files = data.get("files", [])
            keywords_map = data.get("keywords", {})
            if not isinstance(chosen_files, list):
                chosen_files = []
            if not isinstance(keywords_map, dict):
                keywords_map = {}
        except Exception:
            pass

        # filter out nonexistent
        final_files = [f for f in chosen_files if f in all_files]
        print(f"PHASE1: final_files={final_files}, keywords_map={keywords_map}")

        if not final_files:
            return await ctx.send("No relevant files or invalid JSON from LLM. Stopping.")

        # PHASE 2: load & filter
        subset_map = {}
        for f in final_files:
            lines = self.load_tag_file(f)
            file_keys = keywords_map.get(f, [])
            if file_keys:
                lines = filter_lines_by_keywords(lines, file_keys)
            subset_map[f] = lines

        phase2_prompt = (
            "We found these lines (from the chosen files) after applying your suggested keywords:\n\n"
            f"{json.dumps(subset_map, indent=2)}\n\n"
            "Now answer the user's question (concise, <2000 chars) referencing only these lines:\n"
            f"{question}"
        )

        print("=== PHASE2 Prompt ===")
        print(phase2_prompt)
        print("=====================\n")

        try:
            async with ctx.typing():
                final_answer = await self.query_llm(phase2_prompt)
        except Exception as e:
            return await ctx.send(f"Error in Phase2: {e}")

        # Single message only
        # If LLM fails to obey length, we get a 400 error or truncated by Discord
        await ctx.send(f"```\n{final_answer}\n```")

    ########################################################
    # The rest of your commands (no chunkify)
    ########################################################
    @commands.command()
    async def llmknowshow(self, ctx):
        """Merges all .json files, shows each line with an index under the 'tag name'."""
        files = self.list_tag_files()
        if not files:
            return await ctx.send("No tags exist.")
        text = ""
        for f in sorted(files):
            tag = f.rsplit(".",1)[0]
            lines = self.load_tag_file(f)
            text += f"{tag}:\n"
            for i, val in enumerate(lines):
                text += f"  [{i}] {val}\n"
            text += "\n"
        if len(text)<=2000:
            await ctx.send(f"```\n{text}\n```")
        else:
            await ctx.send("Knowledge is too large to show fully in one message. Sorry.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx, tag: str, *, info: str):
        """Adds <info> to <tag>.json, skipping duplicates."""
        fname = f"{tag.lower()}.json"
        lines = self.load_tag_file(fname)
        if info in lines:
            return await ctx.send("That info already exists.")
        lines.append(info)
        self.save_tag_file(fname, lines)
        await ctx.send(f"Added info to {fname}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdelete(self, ctx, tag: str, index: int):
        """Deletes a single line from <tag>.json by index."""
        fname = f"{tag.lower()}.json"
        lines = self.load_tag_file(fname)
        if not lines:
            return await ctx.send("Tag doesn't exist or no lines.")
        if 0<=index<len(lines):
            removed = lines.pop(index)
            self.save_tag_file(fname, lines)
            await ctx.send(f"Deleted: {removed}")
        else:
            await ctx.send("Invalid index.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdeletetag(self, ctx, tag: str):
        """Deletes the <tag>.json file entirely."""
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
        Reads last <amount> messages, LLM suggests { "solutions":[...] }, appended to <tag>.json
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
        convo = "\n".join(f"{m.author.name}: {m.content}" for m in msgs)

        fname = f"{tag.lower()}.json"
        old_lines = self.load_tag_file(fname)

        prompt = (
            f"Existing solutions for tag '{tag}':\n"
            f"{json.dumps(old_lines, indent=2)}\n\n"
            "Below is a conversation. Extract new or refined solutions for that tag.\n"
            "Output valid JSON like { \"solutions\": [\"fix1\",\"fix2\"] }.\n\n"
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
            await ctx.send("LLM suggestion is too long. It ignored our brevity request.")

        await ctx.send("Type 'yes' to confirm, 'no [instructions]' to refine, or 'stop' to cancel.")

        def check(m: discord.Message):
            return (m.author==ctx.author) and (m.channel==ctx.channel)

        while True:
            try:
                reply = await self.bot.wait_for("message", check=check, timeout=120)
                low = reply.content.strip().lower()
                if low=="yes":
                    msg = self.store_learn_solutions(fname, suggestion)
                    await ctx.send(msg)
                    break
                elif low=="stop":
                    await ctx.send("Learning cancelled.")
                    break
                elif low.startswith("no"):
                    instructions = reply.content[2:].strip()
                    new_prompt = prompt + "\nAdditional instructions:\n" + instructions
                    try:
                        async with ctx.typing():
                            suggestion = await self.query_llm(new_prompt)
                        if len(suggestion)<=2000:
                            await ctx.send(f"```\n{suggestion}\n```")
                        else:
                            await ctx.send("Refined suggestion too long. LLM isn't following instructions.")
                        await ctx.send("Type 'yes' to confirm, 'no [instr]' to refine, or 'stop'.")
                    except Exception as ee:
                        await ctx.send(f"Error: {ee}")
                else:
                    await ctx.send("Please type 'yes', 'no [instructions]', or 'stop'.")
            except Exception as e:
                await ctx.send(f"Error or timeout: {e}")
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
                raise ValueError("Not a dict")
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
            return f"Added {changed} new solutions to {fname}."
        except Exception:
            gf = "general.json"
            gf_data = self.load_tag_file(gf)
            if suggestion not in gf_data:
                gf_data.append(suggestion)
                self.save_tag_file(gf, gf_data)
                return f"Not valid JSON. Entire text stored in {gf}."
            else:
                return f"Not valid JSON, already in {gf}."
