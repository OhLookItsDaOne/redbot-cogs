import discord
import os
import json
import re
import requests
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path

###################################
# chunkify: Aufteilen langer Strings
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

class LLMManager(commands.Cog):
    """
    Ein Cog, in dem jedes "Tag" als eigene JSON-Datei in ./tags/ abgelegt wird.
    Die LLM kann so per Command genau eine Datei auswählen (oder 'none' sagen).
    """

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1234567890)
        self.config.register_global(model="gemma3:12b", api_url="http://localhost:11434")

        # Ordner für die Tag-Dateien
        self.tags_folder = cog_data_path(self) / "tags"
        if not self.tags_folder.exists():
            self.tags_folder.mkdir(parents=True, exist_ok=True)

    ############################################################
    # Model- und API-Verwaltung
    ############################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmodel(self, ctx, model: str):
        """Setzt das Standard-LLM-Modell."""
        await self.config.model.set(model)
        await ctx.send(f"LLM model set to {model}")

    @commands.command()
    async def modellist(self, ctx):
        """Zeigt verfügbare Modelle (laut Ollama)."""
        api_url = await self.config.api_url()
        try:
            r = requests.get(f"{api_url}/api/tags")
            r.raise_for_status()
            dat = r.json()
            models = [m["name"] for m in dat.get("models", [])]
            if models:
                await ctx.send(f"Available models: {', '.join(models)}")
            else:
                await ctx.send("No models found.")
        except Exception as e:
            await ctx.send(f"Error: {e}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setapi(self, ctx, url: str):
        """Setzt die Ollama API-URL."""
        url = url.rstrip("/")
        await self.config.api_url.set(url)
        await ctx.send(f"Ollama API URL set to {url}")

    ############################################################
    # Tools zum Umgang mit Tag-Dateien (tags/*.json)
    ############################################################
    def list_tag_files(self) -> list:
        """Gibt eine Liste aller .json-Dateien im 'tags/'-Ordner zurück."""
        all_files = []
        for fname in os.listdir(self.tags_folder):
            if fname.lower().endswith(".json"):
                all_files.append(fname)
        return all_files

    def load_tag_file(self, fname: str) -> list:
        """Lädt eine Tag-Datei (z. B. 'crash.json') als Liste (oder leer bei Fehler)."""
        path = self.tags_folder / fname
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return []

    def save_tag_file(self, fname: str, data_list: list):
        """Speichert die Liste data_list in 'fname' als JSON."""
        path = self.tags_folder / fname
        with path.open("w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=2)

    ############################################################
    # Commands zum Verwalten von Tag-Dateien
    ############################################################
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def createtag(self, ctx, tagname: str):
        """Erzeugt eine leere JSON-Datei (z. B. 'crash.json') in 'tags/'."""
        fname = f"{tagname.lower()}.json"
        path = self.tags_folder / fname
        if path.exists():
            return await ctx.send("Diese Datei / Tag existiert bereits.")
        with path.open("w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        await ctx.send(f"Created new tag file: {fname}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def tagadd(self, ctx, tagname: str, *, info: str):
        """Fügt einen Eintrag (info) zur bestehenden Tagdatei (z. B. crash.json) hinzu."""
        fname = f"{tagname.lower()}.json"
        path = self.tags_folder / fname
        if not path.exists():
            return await ctx.send("That tag file doesn't exist. Use !createtag first.")
        data = self.load_tag_file(fname)
        if info in data:
            return await ctx.send("That info already exists in the file.")
        data.append(info)
        self.save_tag_file(fname, data)
        await ctx.send(f"Appended info to {fname}")

    @commands.command()
    async def tagshow(self, ctx):
        """Listet alle Tag-Dateien mit ihren Inhalten."""
        files = self.list_tag_files()
        if not files:
            return await ctx.send("No tag files exist.")
        out = "Existing Tag Files:\n"
        for f in files:
            out += f"{f}:\n"
            content = self.load_tag_file(f)
            for i, line in enumerate(content):
                out += f"  [{i}] {line}\n"
            out += "\n"

        if len(out) <= 2000:
            await ctx.send(f"```\n{out}\n```")
        else:
            chunks = chunkify(out, 1900)
            for idx, c in enumerate(chunks, 1):
                await ctx.send(f"**(Part {idx}/{len(chunks)}):**\n```\n{c}\n```")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def tagdelete(self, ctx, tagname: str, index: int):
        """Löscht einen Eintrag nach Index aus der Tagdatei (z. B. !tagdelete crash 0)."""
        fname = f"{tagname.lower()}.json"
        path = self.tags_folder / fname
        if not path.exists():
            return await ctx.send("That tag file doesn't exist.")
        data = self.load_tag_file(fname)
        if 0 <= index < len(data):
            removed = data.pop(index)
            self.save_tag_file(fname, data)
            await ctx.send(f"Removed: {removed}")
        else:
            await ctx.send("Invalid index.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def tagdeletetag(self, ctx, tagname: str):
        """Löscht die gesamte Tag-Datei (z. B. 'crash.json')."""
        fname = f"{tagname.lower()}.json"
        path = self.tags_folder / fname
        if not path.exists():
            return await ctx.send("That tag doesn't exist.")
        path.unlink()
        await ctx.send(f"Deleted file: {fname}")

    ############################################################
    # LLM: wähle EINE Tag-Datei oder 'none'
    ############################################################
    async def decide_tag_file(self, user_question: str, tag_files: list) -> str:
        """
        Fragt das LLM: "Hier sind Tag-Dateien. Wähle GENAU EINE, die am besten passt,
        oder gib 'none' zurück, falls nichts relevant ist."
        """
        filenames_str = "\n".join(tag_files)
        prompt = (
            "We have multiple JSON files, each with specialized knowledge.\n"
            f"{filenames_str}\n\n"
            "A user asked:\n"
            f"{user_question}\n\n"
            "Pick exactly ONE filename from the list that best addresses the question. "
            "If none is relevant, say 'none'. Return only that filename or 'none'."
        )
        answer = await self.query_llm(prompt)
        return answer.strip().lower()

    async def answer_from_chosen_file(self, user_question: str, chosen_file: str) -> str:
        """
        Lädt 'chosen_file' und gibt den Inhalt ans LLM, damit es eine finale Antwort generiert.
        """
        data_list = self.load_tag_file(chosen_file)  # e.g. a list of lines
        knowledge_str = json.dumps(data_list, indent=2)
        prompt = (
            f"We have chosen file '{chosen_file}' as relevant.\n"
            "Below is its content:\n"
            f"{knowledge_str}\n\n"
            f"User question: {user_question}\n"
            "Please answer using only this data. If there's nothing relevant, say so."
        )
        final_answer = await self.query_llm(prompt)
        return final_answer

    ############################################################
    # Endgültiger Befehl: !askllm
    ############################################################
    @commands.command()
    async def askllm(self, ctx, *, question: str):
        """
        1) Listet alle Tag-Dateien (.json)
        2) LLM wählt EINE aus (oder 'none')
        3) Lädt nur diese Datei, generiert finale Antwort
        4) chunkify wenn >2000 Zeichen
        """
        tag_files = self.list_tag_files()
        if not tag_files:
            return await ctx.send("No tag files found in './tags'. Create some with !createtag <name>.")

        chosen = await self.decide_tag_file(question, tag_files)
        if chosen == "none" or chosen not in tag_files:
            return await ctx.send("LLM says 'none' or no valid file was chosen. No relevant tag found.")

        # Load + final answer
        final_answer = await self.answer_from_chosen_file(question, chosen)
        if len(final_answer) <= 2000:
            await ctx.send(f"```\n{final_answer}\n```")
        else:
            parts = chunkify(final_answer, 1900)
            for i, c in enumerate(parts,1):
                await ctx.send(f"**(Part {i}/{len(parts)}):**\n```\n{c}\n```")
