import discord
import os
import json
import re
import requests
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path

def filter_lines_by_keywords(lines: list, keywords: list) -> list:
    """Returns only those lines that contain ANY of the given keywords (case-insensitive)."""
    filtered = []
    for line in lines:
        lower_line = line.lower()
        if any(kw.lower() in lower_line for kw in keywords):
            filtered.append(line)
    return filtered

class LLMManager(commands.Cog):
    """
    A simplified two-phase approach for !askllm, with separate JSON files per tag.
    No chunkify, no triple backticks in final answer. If LLM goes over 2000 chars, it might fail.
    Also ensures learn/tag commands create .json if missing.
    """

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1234567890)
        self.config.register_global(model="gemma3:12b", api_url="http://localhost:11434")

        self.tags_folder = cog_data_path(self) / "tags"
        if not self.tags_folder.exists():
            self.tags_folder.mkdir(parents=True, exist_ok=True)

    #############################################
    # Basic Model / API commands
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

    #############################################
    # Tag file utilities
    #############################################
    def list_tag_files(self) -> list:
        """Returns all .json files in tags_folder."""
        result = []
        for fname in os.listdir(self.tags_folder):
            if fname.lower().endswith(".json"):
                result.append(fname)
        return result

    def load_tag_file(self, fname: str) -> list:
        """Loads a single tag file as a list of lines, or empty if missing/corrupt."""
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
    # LLM query with "keep answer short"
    #############################################
    async def query_llm(self, prompt: str) -> str:
        """
        Instruct LLM to keep final answer short. 
        No chunkify. If it disobeys, we might see a Discord error.
        """
        final_prompt = prompt + "\n\nPlease keep your final answer concise (<2000 chars)."
        model = await self.config.model()
        api_url = await self.config.api_url()

        payload = {
            "model": model,
            "messages": [{"role":"user","content":final_prompt}],
            "stream": False
        }
        headers = {"Content-Type": "application/json"}
        r = requests.post(f"{api_url}/api/chat", json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        text = data.get("message",{}).get("content","No valid response received.")
        return text.replace("\n\n","\n")

    #############################################
    # The two-phase askllm approach
    #############################################
    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """
        Phase1 -> LLM sees file list + question -> returns JSON with { files:[], keywords:{} }
        Phase2 -> We load+filter those lines, pass them + question back to LLM, 
                  then send final answer with no triple backticks or chunkify
        """
        all_files = self.list_tag_files()
        if not all_files:
            return await ctx.send("No tag files found. Please use !learn or !llmknow to create some first.")

        # Phase 1
        files_str = "\n".join(all_files)
        phase1_prompt = (
            "We have multiple JSON files containing lines of specialized info.\n"
            "User question:\n"
            f"{question}\n\n"
            "List of possible files:\n"
            f"{files_str}\n\n"
            "Propose which files might be relevant and what keywords to look for in each file.\n"
            "Return valid JSON:\n"
            "{\n"
            "  \"files\": [ \"somefile.json\" ],\n"
            "  \"keywords\": {\n"
            "    \"somefile.json\": [\"keyword1\",\"keyword2\"]\n"
            "  }\n"
            "}\n"
            "If none relevant, 'files':[], 'keywords':{}."
        )
        try:
            phase1_answer = await self.query_llm(phase1_prompt)
        except Exception as e:
            return await ctx.send(f"Phase1 error: {e}")

        # Attempt to parse the JSON
        pat = r"(?s)```json\s*(\{.*?\})\s*```"
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

        # filter out nonexistent
        final_files = [f for f in chosen_files if f in all_files]
        if not final_files:
            return await ctx.send("No relevant files found or invalid Phase1 response. Stopping.")

        # Phase 2 -> load & filter
        subset_map = {}
        for f in final_files:
            lines = self.load_tag_file(f)
            file_keys = keywords_map.get(f, [])
            if file_keys:
                lines = filter_lines_by_keywords(lines, file_keys)
            subset_map[f] = lines

        phase2_prompt = (
            "Below is the filtered content from the chosen files:\n"
            f"{json.dumps(subset_map, indent=2)}\n\n"
            "Now answer the user's question based on this info only:\n"
            f"{question}"
        )
        try:
            final_answer = await self.query_llm(phase2_prompt)
        except Exception as e:
            return await ctx.send(f"Phase2 error: {e}")

        await ctx.send(final_answer)  # no triple backticks, no chunkify

    #############################################
    # !llmknowshow => merges & displays all tags
    #############################################
    @commands.command()
    async def llmknowshow(self, ctx):
        """Displays all lines from all tag files, with indexing."""
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

        # We won't chunkify. If it's 2k+ chars, we risk an error
        # If you prefer a fallback, you can do it here
        await ctx.send(out)

    #############################################
    # !llmknow <tag> "message"
    #############################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknow(self, ctx, tag: str, *, info: str):
        """Adds <info> to <tag>.json, skipping duplicates. Creates the file if missing."""
        fname = f"{tag.lower()}.json"
        lines = self.load_tag_file(fname)
        if info in lines:
            return await ctx.send("That info already exists; skipping.")
        lines.append(info)
        self.save_tag_file(fname, lines)
        await ctx.send(f"Added info to {fname}.")

    #############################################
    # !llmknowdelete <tag> <index>
    #############################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def llmknowdelete(self, ctx, tag: str, index: int):
        """Removes a single line from <tag>.json by index."""
        fname = f"{tag.lower()}.json"
        lines = self.load_tag_file(fname)
        if not lines:
            return await ctx.send("No lines or tag file doesn't exist.")
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
        """Removes <tag>.json entirely."""
        fname = f"{tag.lower()}.json"
        path = self.tags_folder / fname
        if not path.exists():
            return await ctx.send("Tag file not found.")
        path.unlink()
        await ctx.send(f"Deleted entire tag '{fname}'.")

    #############################################
    # !learn <tag> <message_count>
    #############################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, tag: str, amount: int=20):
        """
        Reads last <amount> messages, calls LLM for { "solutions": [ ... ] },
        appends them to <tag>.json
        """
        def not_command_or_bot(m: discord.Message):
            if m.author.bot:
                return False
            if m.id == ctx.message.id:
                return False
            return True

        # gather messages
        msgs = []
        async for msg in ctx.channel.history(limit=amount+5):
            if not_command_or_bot(msg):
                msgs.append(msg)
            if len(msgs) >= amount:
                break
        msgs.reverse()
        conversation = "\n".join(f"{m.author.name}: {m.content}" for m in msgs)

        fname = f"{tag.lower()}.json"
        old_lines = self.load_tag_file(fname)

        prompt = (
            f"Existing solutions for tag '{tag}':\n"
            f"{json.dumps(old_lines, indent=2)}\n\n"
            "Below is a conversation. Extract new or refined solutions for that tag.\n"
            "Output JSON: { \"solutions\":[\"fix1\",\"fix2\"] }\n\n"
            f"Conversation:\n{conversation}"
        )
        try:
            suggestion = await self.query_llm(prompt)
        except Exception as e:
            return await ctx.send(f"Error: {e}")

        # no chunkify, no triple backticks
        await ctx.send(suggestion)

        await ctx.send("Type 'yes' to confirm, 'no [instr]' to refine, or 'stop' to cancel.")

        def check(m: discord.Message):
            return (m.author==ctx.author) and (m.channel==ctx.channel)

        while True:
            try:
                reply = await self.bot.wait_for("message", check=check, timeout=120)
                txt = reply.content.strip().lower()
                if txt=="yes":
                    out = self.store_learn_solutions(fname, suggestion)
                    await ctx.send(out)
                    break
                elif txt=="stop":
                    await ctx.send("Learning cancelled.")
                    break
                elif txt.startswith("no"):
                    instructions = reply.content[2:].strip()
                    refine_prompt = prompt + "\nAdditional instructions:\n" + instructions
                    try:
                        suggestion = await self.query_llm(refine_prompt)
                        await ctx.send(suggestion)
                        await ctx.send("Type 'yes' to confirm, 'no [instr]' or 'stop'.")
                    except Exception as ee:
                        await ctx.send(f"Error: {ee}")
                else:
                    await ctx.send("Please type 'yes', 'no [instructions]', or 'stop'.")
            except Exception as e:
                await ctx.send(f"Error or timeout: {e}")
                break

    def store_learn_solutions(self, fname: str, suggestion: str) -> str:
        """Parses { 'solutions':[...]} from the suggestion, appends them to <fname>, skipping duplicates."""
        pat = r"```json\s*(\{.*?\})\s*```"
        mt = re.search(pat, suggestion)
        raw = suggestion
        if mt:
            raw = mt.group(1)
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Not a dict.")
            arr = data.get("solutions", [])
            if not isinstance(arr, list):
                raise ValueError("No valid 'solutions' array.")

            old = self.load_tag_file(fname)
            changed=0
            for fix in arr:
                if fix not in old:
                    old.append(fix)
                    changed+=1
            self.save_tag_file(fname, old)
            return f"Added {changed} new solution(s) to {fname}"
        except Exception:
            # fallback -> store entire text in "general.json"
            gf = "general.json"
            gf_data = self.load_tag_file(gf)
            if suggestion not in gf_data:
                gf_data.append(suggestion)
                self.save_tag_file(gf, gf_data)
                return f"Invalid JSON. Entire text stored in {gf}."
            else:
                return f"Invalid JSON, already in {gf}."
