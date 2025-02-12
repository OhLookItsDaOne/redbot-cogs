import discord
from redbot.core import commands, Config
import aiohttp, io, re, os, json, asyncio
import sys
import subprocess

# Versuche, PyPDF2 zu importieren – falls nicht vorhanden, wird es installiert.
try:
    from PyPDF2 import PdfReader
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    from PyPDF2 import PdfReader

class AIHelp(commands.Cog):
    """
    AIHelp Cog

    This cog integrates a custom AI API (e.g. lamacpp) to generate support responses using
    Retrieval Augmented Generation (RAG). The RAG sources (tips, guides etc.) are stored globally
    in a directory structure inside the cog folder. For example, you can create a folder "Performance"
    to store performance tips and guides. When adding a source via the command, you may optionally
    specify the category (directory) and keywords (tags). If no category is specified, the source is
    stored in the "general" category.

    At runtime the bot scans the stored sources based on tags and/or an explicit category prefix in your query.
    While processing a request, the bot edits its progress message and finally posts the AI-generated answer.
    Additionally, there is a rating system so that users can rate (up/down) which RAG source was helpful.
    """

    ALLOWED_EXTENSIONS = ('.pdf', '.txt', '.json', '.md')

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9876543210)
        default_global = {
            "api_endpoint": "",      # API endpoint, e.g. "http://192.168.10.5:8999/"
            "allowed_roles": [],     # (Optional) List of role IDs allowed to use !aihelp
            "support_channels": []   # Channels where proactive support is enabled
        }
        self.config.register_global(**default_global)

        # Directory for storing sources (RAG data)
        self.sources_dir = os.path.join(os.path.dirname(__file__), "rag_sources")
        if not os.path.exists(self.sources_dir):
            os.makedirs(self.sources_dir)

        # Metadata file for storing tags for each source
        self.metadata_file = os.path.join(self.sources_dir, "metadata.json")
        self.load_metadata()

        # Ratings file for storing user ratings for sources
        self.ratings_file = os.path.join(os.path.dirname(__file__), "rag_ratings.json")
        self.load_ratings()

        # Global sources: list of dicts with keys: category, filename, content, tags, rel_path
        self.global_sources = []
        self.load_all_sources()

        # Set to track user IDs that have opted out of proactive support.
        self.opted_out = set()

    # Decorator to check if the user is the owner or has administrator permissions
    def owner_or_admins():
        async def predicate(ctx):
            return await ctx.bot.is_owner(ctx.author) or ctx.author.guild_permissions.administrator
        return commands.check(predicate)

    # ─── METADATA & RATINGS MANAGEMENT ───────────────────────────────────────────
    def load_metadata(self):
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}
        else:
            self.metadata = {}

    def save_metadata(self):
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def load_ratings(self):
        if os.path.exists(self.ratings_file):
            try:
                with open(self.ratings_file, "r", encoding="utf-8") as f:
                    self.ratings = json.load(f)
            except Exception:
                self.ratings = {}
        else:
            self.ratings = {}

    def save_ratings(self):
        with open(self.ratings_file, "w", encoding="utf-8") as f:
            json.dump(self.ratings, f, indent=2)

    # ─── SOURCES LOADING ──────────────────────────────────────────────────────────
    def load_all_sources(self):
        """
        Scans the sources directory (and its subdirectories) and loads all files along with metadata.
        """
        global_sources = []
        for root, dirs, files in os.walk(self.sources_dir):
            for file in files:
                if file.lower().endswith(self.ALLOWED_EXTENSIONS):
                    full_path = os.path.join(root, file)
                    # Category: relative path from sources_dir (if in a subfolder, else "general")
                    category = os.path.relpath(root, self.sources_dir)
                    if category == ".":
                        category = "general"
                    # Load file content
                    if file.lower().endswith(".pdf"):
                        try:
                            with open(full_path, "rb") as f:
                                reader = PdfReader(f)
                                text = ""
                                for page in reader.pages:
                                    page_text = page.extract_text()
                                    if page_text:
                                        text += page_text + "\n"
                        except Exception as e:
                            text = f"Error reading PDF: {e}"
                    else:
                        try:
                            with open(full_path, "r", encoding="utf-8") as f:
                                text = f.read()
                        except Exception as e:
                            text = f"Error reading file: {e}"
                    # Relative path key (use forward slashes for consistency)
                    rel_path = os.path.relpath(full_path, self.sources_dir).replace(os.sep, "/")
                    # Get tags from metadata (if present)
                    tags = self.metadata.get(rel_path, {}).get("tags", [])
                    global_sources.append({
                        "category": category,
                        "filename": file,
                        "content": text,
                        "tags": tags,
                        "rel_path": rel_path
                    })
        self.global_sources = global_sources

    # ─── API CALL & RAG CONTEXT ───────────────────────────────────────────────────
    async def call_api(self, prompt: str) -> str:
        """
        Calls the configured API endpoint with the given prompt and returns the response.
        Expects a JSON payload with key "prompt" and a JSON response with key "response".
        """
        endpoint = await self.config.api_endpoint()
        if not endpoint:
            raise Exception("API endpoint is not configured. Please set it with !aihelpowner endpoint <url>")
        payload = {
            "prompt": prompt,
            "temperature": 0.2,
            "max_tokens": 500
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"API request error (Status {response.status}): {text}")
                data = await response.json()
        try:
            answer = data["response"].strip()
            return answer
        except Exception as e:
            raise Exception(f"Error processing API response: {e}")

    def retrieve_context(self, query: str, documents: list, top_k: int = 3) -> str:
        """
        Retrieves context from the provided documents based on the query using simple keyword matching.
        """
        scores = []
        query_lower = query.lower()
        for doc in documents:
            content = doc.get("content", "")
            score = content.lower().count(query_lower)
            scores.append((score, content))
        scores.sort(key=lambda x: x[0], reverse=True)
        selected = [content for score, content in scores if score > 0][:top_k]
        if not selected:
            selected = [content for score, content in scores][:top_k]
        return "\n---\n".join(selected)

    def generate_augmented_query(self, query: str) -> (str, str):
        """
        Checks whether the query specifies a category using the format "Category: actual query".
        Returns a tuple (category, query) where category may be None.
        """
        m = re.match(r"^(\w+):\s*(.*)", query)
        if m:
            return m.group(1), m.group(2)
        return None, query

    async def generate_ai_response(self, query: str) -> str:
        """
        Generates an AI response for the given query.
        If a category is specified (e.g. "Performance:"), only sources from that category are used.
        Otherwise, the query is scanned for keywords that match source tags.
        """
        category, actual_query = self.generate_augmented_query(query)
        if category:
            docs = [doc for doc in self.global_sources if doc["category"].lower() == category.lower()]
        else:
            # Suche nach passenden Quellen anhand der in den Quellen hinterlegten Tags
            matching_docs = []
            for doc in self.global_sources:
                for tag in doc.get("tags", []):
                    if tag.lower() in actual_query.lower():
                        matching_docs.append(doc)
                        break
            docs = matching_docs if matching_docs else self.global_sources
        if docs:
            context = self.retrieve_context(actual_query, docs)
            augmented_query = f"Context:\n{context}\n\nQuestion: {actual_query}"
        else:
            augmented_query = actual_query
        return await self.call_api(augmented_query)

    # ─── OWNER-ONLY COMMANDS ─────────────────────────────────────────────────────
    @commands.group()
    @commands.is_owner()
    async def aihelpowner(self, ctx):
        """
        Owner-only commands for AIHelp.
        Subcommand: endpoint – Sets the API endpoint.
        """
        if ctx.invoked_subcommand is None:
            await ctx.send("Available subcommand: endpoint")

    @aihelpowner.command(name="endpoint")
    async def aihelpowner_endpoint(self, ctx, endpoint: str):
        """
        Sets the API endpoint.
        Example: `!aihelpowner endpoint http://192.168.10.5:8999/`
        """
        if not (endpoint.startswith("http://") or endpoint.startswith("https://")):
            await ctx.send("Invalid endpoint. It must start with http:// or https://")
            return
        await self.config.api_endpoint.set(endpoint)
        await ctx.send(f"API endpoint set to: {endpoint}")

    # ─── PUBLIC COMMAND: AIHELP (with progress updates) ─────────────────────────
    @commands.command()
    async def aihelp(self, ctx, *, query: str):
        """
        Generates an AI response for the provided query using the configured API with RAG context.
        Every response includes a notice: if you do not wish to receive support,
        say 'no help' and I will stop assisting you until you mention me (@BOTNAME).
        """
        progress_message = await ctx.send("Processing query... (0%)")
        try:
            await asyncio.sleep(0.3)
            await progress_message.edit(content="Scanning sources... (20%)")
            await asyncio.sleep(0.3)
            await progress_message.edit(content="Generating AI response... (50%)")
            response = await self.generate_ai_response(query)
            await progress_message.edit(content="Finalizing answer... (90%)")
            await asyncio.sleep(0.3)
        except Exception as e:
            await progress_message.edit(content=f"Error: {e}")
            return
        info_message = (
            f"Note: If you do not wish to receive support, please say 'no help' and I will refrain from assisting you until you mention me (@{self.bot.user.name}).\n\n"
        )
        await progress_message.edit(content=info_message + response)

    # ─── CONFIGURATION COMMANDS (Owner-only) ─────────────────────────────────────
    @commands.group()
    @commands.is_owner()
    async def aihelpconfig(self, ctx):
        """
        Configuration commands for AIHelp.

        Subcommands:
         - addsource <url> [category] [tags] : Adds a source file. Category defaults to "general"; tags are comma‑separated.
         - listsources [<category>]
         - removesource <category> <filename>
         - updatesourcetags <category> <filename> <tags>
         - movesource <old_category> <filename> <new_category>
         - setchannels <channel ids>
         - reloadsources
        """
        if ctx.invoked_subcommand is None:
            await ctx.send("Available subcommands: addsource, listsources, removesource, updatesourcetags, movesource, setchannels, reloadsources")

    @commands.check_any(commands.is_owner(), commands.has_permissions(administrator=True))
    @aihelpconfig.command(name="addsource")
    async def aihelpconfig_addsource(self, ctx, url: str = None, category: str = "general", *, tags: str = ""):
    # Implementierung des Befehls...

        """
        Downloads a source file from a URL or uses an attached file from your message and saves it in the specified category.
        If no category is provided, it defaults to "general". Optionally, add comma‑separated tags.
        
        **Examples:**
          • URL source:
             `!aihelpconfig addsource https://github.com/user/repo/blob/branch/file.txt Performance performance,guide`
          • Attached file:
             (Attach a file to your message, then run)
             `!aihelpconfig addsource Performance performance,guide`
        """
        # Falls der User eine Datei angehängt hat, wird diese bevorzugt verwendet.
        if ctx.message.attachments:
            url = ctx.message.attachments[0].url

        if not url:
            await ctx.send("Please provide a URL or attach a file to use as a source.")
            return

        try:
            content = await self.download_source(url)
        except Exception as e:
            await ctx.send(f"Error downloading source: {e}")
            return

        # Bestimme den Dateinamen anhand der URL (oder des Attachment-Links).
        filename = url.rstrip("/").split("/")[-1]
        category_dir = os.path.join(self.sources_dir, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        file_path = os.path.join(category_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Aktualisiere die Metadata: Relativer Pfad im Format "category/filename"
        rel_path = os.path.relpath(file_path, self.sources_dir).replace(os.sep, "/")
        tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        self.metadata[rel_path] = {"tags": tags_list}
        self.save_metadata()
        self.load_all_sources()
        await ctx.send(f"Source added in category '{category}' as file '{filename}' with tags: {tags_list}")

    async def download_source(self, url: str) -> str:
        """
        Downloads a file from the given URL and extracts its text content.
        Supported file types: PDF, TXT, JSON, and GitHub links.
        """
        if "github.com" in url and not url.startswith("https://raw.githubusercontent.com/"):
            raw_url = await self.convert_github_url(url)
            if raw_url:
                url = raw_url
            else:
                raise Exception("Unknown GitHub URL format.")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Error downloading file (Status: {response.status})")
                data = await response.read()
        if url.lower().endswith(".pdf"):
            try:
                reader = PdfReader(io.BytesIO(data))
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
            except Exception as e:
                raise Exception(f"Error processing PDF: {e}")
        elif url.lower().endswith(".txt") or url.lower().endswith(".json"):
            try:
                return data.decode("utf-8")
            except Exception as e:
                raise Exception(f"Error decoding text file: {e}")
        else:
            raise Exception("Unsupported file type. Only GitHub links, PDF, TXT, and JSON are supported.")

    async def convert_github_url(self, url: str) -> str:
        """
        Converts a typical GitHub file URL to its corresponding raw URL.
        """
        pattern = r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)"
        match = re.match(pattern, url)
        if match:
            user, repo, branch, path = match.groups()
            raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
            return raw_url
        else:
            return None

    @aihelpconfig.command(name="listsources")
    async def aihelpconfig_listsources(self, ctx, category: str = None):
        """
        Lists source files in a specific category or all sources if no category is provided.
        Examples:
          `!aihelpconfig listsources`
          `!aihelpconfig listsources Performance`
        """
        if category:
            category_dir = os.path.join(self.sources_dir, category)
            if not os.path.exists(category_dir):
                await ctx.send(f"No sources found for category '{category}'.")
                return
            files = os.listdir(category_dir)
            if not files:
                await ctx.send(f"No sources found in category '{category}'.")
            else:
                await ctx.send(f"Sources in category '{category}':\n" + "\n".join(files))
        else:
            result = ""
            for root, dirs, files in os.walk(self.sources_dir):
                rel_dir = os.path.relpath(root, self.sources_dir)
                if files:
                    result += f"Category '{rel_dir}':\n" + "\n".join(files) + "\n"
            if not result:
                result = "No sources found."
            await ctx.send(result)

    @aihelpconfig.command(name="removesource")
    async def aihelpconfig_removesource(self, ctx, category: str, filename: str):
        """
        Removes a source file from the specified category.
        Example:
          `!aihelpconfig removesource Performance tips.txt`
        """
        category_dir = os.path.join(self.sources_dir, category)
        file_path = os.path.join(category_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            # Remove metadata entry, falls vorhanden.
            rel_path = os.path.relpath(file_path, self.sources_dir).replace(os.sep, "/")
            if rel_path in self.metadata:
                del self.metadata[rel_path]
                self.save_metadata()
            self.load_all_sources()
            await ctx.send(f"Removed source '{filename}' from category '{category}'.")
        else:
            await ctx.send(f"File '{filename}' not found in category '{category}'.")

    @aihelpconfig.command(name="updatesourcetags")
    async def aihelpconfig_updatesourcetags(self, ctx, category: str, filename: str, *, tags: str):
        """
        Updates the tags for a source file.
        Example:
          `!aihelpconfig updatesourcetags Performance tips.txt performance,guide,optimization`
        """
        category_dir = os.path.join(self.sources_dir, category)
        file_path = os.path.join(category_dir, filename)
        if not os.path.exists(file_path):
            await ctx.send(f"File '{filename}' not found in category '{category}'.")
            return
        rel_path = os.path.relpath(file_path, self.sources_dir).replace(os.sep, "/")
        tags_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        self.metadata[rel_path] = {"tags": tags_list}
        self.save_metadata()
        self.load_all_sources()
        await ctx.send(f"Updated tags for '{filename}' in category '{category}' to: {tags_list}")

    @aihelpconfig.command(name="movesource")
    async def aihelpconfig_movesource(self, ctx, old_category: str, filename: str, new_category: str):
        """
        Moves a source file from one category to another.
        Example:
          `!aihelpconfig movesource general tips.txt Performance`
        """
        old_dir = os.path.join(self.sources_dir, old_category)
        new_dir = os.path.join(self.sources_dir, new_category)
        old_path = os.path.join(old_dir, filename)
        if not os.path.exists(old_path):
            await ctx.send(f"File '{filename}' not found in category '{old_category}'.")
            return
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        new_path = os.path.join(new_dir, filename)
        os.rename(old_path, new_path)
        # Update metadata: remove old entry and add new entry with same tags.
        old_rel = os.path.relpath(old_path, self.sources_dir).replace(os.sep, "/")
        new_rel = os.path.relpath(new_path, self.sources_dir).replace(os.sep, "/")
        if old_rel in self.metadata:
            self.metadata[new_rel] = self.metadata.pop(old_rel)
            self.save_metadata()
        self.load_all_sources()
        await ctx.send(f"Moved source '{filename}' from '{old_category}' to '{new_category}'.")

    @aihelpconfig.command(name="setchannels")
    async def aihelpconfig_setchannels(self, ctx, *channel_ids: int):
        """
        Sets the channels where proactive support is enabled.
        Example:
          `!aihelpconfig setchannels 123456789012345678 987654321098765432`
        """
        await self.config.support_channels.set(list(channel_ids))
        await ctx.send(f"Support channels set to: {channel_ids}")

    @aihelpconfig.command(name="reloadsources")
    async def aihelpconfig_reloadsources(self, ctx):
        """
        Reloads all sources from disk.
        """
        self.load_metadata()
        self.load_all_sources()
        await ctx.send("Sources reloaded from disk.")

    # ─── RATING COMMAND (for users) ─────────────────────────────────────────────
    @commands.command()
    async def aihelprate(self, ctx, category: str, filename: str, rating: str):
        """
        Rates a source file as helpful or not.
        Usage: `!aihelprate <category> <filename> <up|down>`
        Example: `!aihelprate Performance tips.txt up`
        """
        key = f"{category}/{filename}"
        rating = rating.lower()
        if rating not in ["up", "down"]:
            await ctx.send("Please specify rating as 'up' or 'down'.")
            return
        if key not in self.ratings:
            self.ratings[key] = {"up": 0, "down": 0}
        self.ratings[key][rating] += 1
        self.save_ratings()
        await ctx.send(f"Recorded rating for '{filename}' in category '{category}': {rating} (Total: {self.ratings[key]})")

    # ─── HELP COMMAND ─────────────────────────────────────────────────────────────
    @commands.command(name="aihelphelp")
    async def aihelphelp(self, ctx):
        """
        Displays help information for the AIHelp cog.
        """
        help_text = (
            "**AIHelp Cog Information**\n\n"
            "This cog integrates a custom AI API (e.g. lamacpp) to generate support responses using Retrieval Augmented Generation (RAG).\n"
            "The sources for RAG are stored in a directory structure inside the cog folder and can be organized by category (e.g. 'Performance').\n"
            "You can add sources with optional tags so they can later be managed (updated, moved, or deleted).\n\n"
            "The bot also provides progress feedback while generating responses and supports a rating system to evaluate which sources were helpful.\n\n"
            "```\n"
            "+--------------------------------------------------------------+------------------------------------------------------------+\n"
            "| Command                                                      | Description                                                |\n"
            "+--------------------------------------------------------------+------------------------------------------------------------+\n"
            "| !aihelpowner endpoint <url>                                  | Sets the API endpoint (Owner only)                         |\n"
            "| !aihelp                                                      | Generates an AI response with RAG context (Everyone)       |\n"
            "| !aihelpconfig addsource <url> [category] [tags]                | Adds a source (defaults to 'general'); tags are comma‑separated. |\n"
            "| !aihelpconfig listsources [<category>]                        | Lists source files (Owner only)                            |\n"
            "| !aihelpconfig removesource <category> <filename>              | Removes a source file (Owner only)                         |\n"
            "| !aihelpconfig updatesourcetags <category> <filename> <tags>     | Updates the tags for a source (Owner only)                 |\n"
            "| !aihelpconfig movesource <old_cat> <filename> <new_cat>         | Moves a source file to another category (Owner only)       |\n"
            "| !aihelpconfig setchannels <channel ids>                       | Sets proactive support channels (Owner only)               |\n"
            "| !aihelpconfig reloadsources                                   | Reloads all sources from disk (Owner only)                 |\n"
            "| !aihelprate <category> <filename> <up|down>                   | Rates a source as helpful (Everyone)                       |\n"
            "| !aihelphelp                                                  | Displays this help information                             |\n"
            "+--------------------------------------------------------------+------------------------------------------------------------+\n"
            "```\n\n"
            "Replace `<url>` with your API endpoint (e.g. http://192.168.10.5:8999/).\n"
        )
        await ctx.send(help_text)

    # ─── LISTENER FOR PROACTIVE SUPPORT ─────────────────────────────────────────
    @commands.Cog.listener()
    async def on_message(self, message):
        # Ignore bot messages and commands (those starting with "!")
        if message.author.bot or message.content.startswith("!"):
            return

        # Check if the message is in a configured support channel.
        support_channels = await self.config.support_channels()
        if support_channels and message.channel.id not in support_channels:
            return

        # If the message mentions the bot, remove the user from the opted-out set.
        if self.bot.user in message.mentions:
            if message.author.id in self.opted_out:
                self.opted_out.remove(message.author.id)
                await message.channel.send(f"{message.author.mention}, support re-enabled as requested.")
            # Continue processing.

        # If the user has opted out, do not provide proactive support.
        if message.author.id in self.opted_out:
            return

        lower_content = message.content.lower()

        # Check if the user explicitly indicates that they do not want help.
        opt_out_phrases = [
            "no help", "i don't need help", "don't help me",
            "keine hilfe", "ich brauche keine hilfe", "keine unterstützung"
        ]
        if any(phrase in lower_content for phrase in opt_out_phrases):
            self.opted_out.add(message.author.id)
            await message.channel.send(
                f"{message.author.mention}, understood. I will not provide support until you mention me (@{self.bot.user.name})."
            )
            return

        # Simple keyword matching for help requests.
        help_keywords = ["help", "hilfe", "problem", "support", "error", "issue"]
        if any(keyword in lower_content for keyword in help_keywords):
            info_message = (
                f"{message.author.mention}, it seems you might be asking for help. I can provide AI-generated support.\n"
                f"Note: If you do not wish to receive help, please say 'no help' and I will refrain from assisting you until you mention me (@{self.bot.user.name}).\n"
            )
            try:
                ai_response = await self.generate_ai_response(message.content)
            except Exception as e:
                ai_response = f"Error generating AI support: {e}"
            final_response = info_message + "\n" + ai_response
            await message.channel.send(final_response)

