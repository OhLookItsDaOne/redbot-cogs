import discord
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path
import requests
import json
import re
import os

###################################
# HELPER: chunkify
###################################
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

###################################
# HELPER: chunked send
###################################
async def send_chunked_or_single(ctx, text: str):
    """Sends the LLM's response in one message if short, or in chunks if > 2000 chars."""
    if len(text) <= 2000:
        await ctx.send(f"```\n{text}\n```")
    else:
        parts = chunkify(text, 1900)
        for i, c in enumerate(parts, start=1):
            await ctx.send(f"**(Part {i}/{len(parts)}):**\n```\n{c}\n```")

class LLMManager(commands.Cog):
    """
    Cog that stores each 'tag' in a separate JSON file in `tags/`.
    The LLM picks one file for the final answer or says 'none'.
    """

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1234567890)
        self.config.register_global(model="gemma3:12b", api_url="http://localhost:11434")

        # We'll store separate JSON files in `tags/`
        self.tags_folder = cog_data_path(self) / "tags"
        if not self.tags_folder.exists():
            self.tags_folder.mkdir(parents=True, exist_ok=True)

    ############################################################
    # 1) Listing tag files
    ############################################################
    def list_tag_files(self) -> list:
        """Returns a list of all .json files in the `tags/` folder, e.g. ['crash.json','enb.json']. """
        all_files = []
        for fname in os.listdir(self.tags_folder):
            if fname.lower().endswith(".json"):
                all_files.append(fname)
        return all_files

    def load_tag_file(self, fname: str) -> list:
        """Loads a single tag file as a Python list. If you prefer dict, adjust accordingly."""
        path = self.tags_folder / fname
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                # If you store a dict or something else, adjust logic
                return []
        return data

    def save_tag_file(self, fname: str, content: list):
        path = self.tags_folder / fname
        with path.open("w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)

    ############################################################
    # 2) Asking the LLM which tag file is relevant
    ############################################################
    async def decide_tag_file(self, user_question: str, tag_files: list) -> str:
        """
        Asks the LLM which single file from tag_files best addresses user_question.
        If none is relevant, the LLM should respond 'none'.
        We'll prompt the LLM: 'Here are tag files: ...; pick exactly one or say none.'
        """
        filenames_str = "\n".join(tag_files)
        prompt = (
            "We have several JSON files, each containing specialized info:\n"
            f"{filenames_str}\n\n"
            "A user asked:\n"
            f"{user_question}\n\n"
            "Pick exactly ONE filename from the above list that best addresses the question. "
            "If none is relevant, say 'none'. Return only the filename or 'none'."
        )
        answer = await self.query_llm(prompt)
        cleaned = answer.strip().lower()
        return cleaned

    async def query_llm(self, prompt: str) -> str:
        """Sends prompt to LLM, removing double newlines from the result."""
        model = await self.config.model()
        api_url = await self.config.api_url()

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        headers = {"Content-Type":"application/json"}
        resp = requests.post(f"{api_url}/api/chat", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("message",{}).get("content","No valid response received.")
        # remove double spacing
        text = text.replace("\n\n","\n")
        return text

    ############################################################
    # 3) After deciding file, load it & ask for final answer
    ############################################################
    async def answer_from_chosen_file(self, user_question: str, chosen_file: str) -> str:
        """
        Loads chosen_file from tags folder, sends it to LLM with the user's question.
        Example data format is a list of lines, so we do json.dumps.
        """
        data_list = self.load_tag_file(chosen_file)
        knowledge_str = json.dumps(data_list, indent=2)
        prompt = (
            "We have chosen this JSON file as relevant. Use its content to answer.\n\n"
            f"File: {chosen_file}\n"
            f"Content:\n{knowledge_str}\n\n"
            f"User question: {user_question}\n"
            "Don't guess. If no direct info, say so."
        )
        final_answer = await self.query_llm(prompt)
        return final_answer

    ############################################################
    # Simple "ask" command that picks the file, loads it, etc.
    ############################################################
    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """
        1) List all .json in tags/
        2) LLM picks one or 'none'
        3) If 'none', say "No relevant data."
        4) Otherwise, load the file and do final answer
        5) chunkify the final response
        """
        tag_files = self.list_tag_files()
        if not tag_files:
            return await ctx.send("No tag files exist. Please create some in `tags/` folder.")

        # 1) LLM picks a single file
        chosen = await self.decide_tag_file(question, tag_files)
        if chosen == "none" or chosen not in tag_files:
            return await ctx.send("No relevant file found or the LLM said 'none'.")

        # 2) Load that file, produce final answer
        answer = await self.answer_from_chosen_file(question, chosen)

        # 3) chunkify if needed
        if len(answer) <= 2000:
            await ctx.send(f"```\n{answer}\n```")
        else:
            parts = chunkify(answer, 1900)
            for i, c in enumerate(parts,1):
                await ctx.send(f"**(Part {i}/{len(parts)}):**\n```\n{c}\n```")

    ############################################################
    # If you want to add a method to create new tag files, etc.
    ############################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def createtag(self, ctx, tagname: str):
        """
        Create an empty JSON file for a new 'tag'.
        e.g. !createtag crash
        => tags/crash.json
        """
        filename = f"{tagname.lower()}.json"
        path = self.tags_folder / filename
        if path.exists():
            return await ctx.send("That file/tag already exists.")
        with path.open("w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        await ctx.send(f"Created tag file: {filename}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def tagadd(self, ctx, tagname: str, *, info: str):
        """
        Add a line of info to an existing tag file.
        e.g. !tagadd crash "Try using crash analyzer xyz"
        """
        filename = f"{tagname.lower()}.json"
        path = self.tags_folder / filename
        if not path.exists():
            return await ctx.send("That tag file doesn't exist. Create it first or check spelling.")

        data = self.load_tag_file(filename)  # returns a list
        if info in data:
            return await ctx.send("That info already exists in the file.")
        data.append(info)
        self.save_tag_file(filename, data)
        await ctx.send(f"Appended info to {filename}")

    @commands.command()
    async def tagshow(self, ctx):
        """
        Lists all .json files and their contents
        """
        files = self.list_tag_files()
        if not files:
            return await ctx.send("No tag files exist.")
        out = "Existing Tag Files:\n"
        for f in files:
            out += f"{f}\n"
            content = self.load_tag_file(f)
            for i, line in enumerate(content):
                out += f"  [{i}] {line}\n"
            out += "\n"

        if len(out)<=2000:
            await ctx.send(f"```\n{out}\n```")
        else:
            chunks = chunkify(out,1900)
            for idx, chunk in enumerate(chunks,1):
                await ctx.send(f"**(Part {idx}/{len(chunks)}):**\n```\n{chunk}\n```")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def tagdelete(self, ctx, tagname: str, index: int):
        """
        Delete an entry from a given tag file by index
        e.g. !tagdelete crash 0
        """
        fname = f"{tagname.lower()}.json"
        path = self.tags_folder / fname
        if not path.exists():
            return await ctx.send("That tag file doesn't exist.")
        data = self.load_tag_file(fname)
        if 0<=index<len(data):
            removed = data.pop(index)
            self.save_tag_file(fname, data)
            await ctx.send(f"Removed: {removed}")
        else:
            await ctx.send("Invalid index.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def tagdeletetag(self, ctx, tagname: str):
        """
        Completely remove a tag file
        e.g. !tagdeletetag crash
        """
        fname = f"{tagname.lower()}.json"
        path = self.tags_folder / fname
        if not path.exists():
            return await ctx.send("That tag doesn't exist.")
        path.unlink()
        await ctx.send(f"Deleted file: {fname}")
