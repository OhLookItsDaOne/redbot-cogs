import aiohttp
import aiofiles
import asyncio
import json
import os
from typing import Optional, List, Dict
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timedelta

import discord
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path

class DeepSeekCog(commands.Cog):
    """A Cog that uses DeepSeek API for specific topic queries with RAG learning functionality."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1234567890)
        default_global = {
            "api_key": None,
            "base_url": "https://api.deepseek.com/v1",
            "context_data": "",
            "context_source": "internal",
            "github_url": None,
            "github_branch": "main",
            "github_path": "",
            "cache_duration": 300,
            "timeout": 30,
            "rag_context_messages": 15,
            "learning_enabled": True
        }
        
        default_guild = {
            "learned_solutions": {},
            "learning_role": None
        }
        
        self.config.register_global(**default_global)
        self.config.register_guild(**default_guild)
        
        self.data_path = cog_data_path(self) / "context_data.txt"
        self.learned_db_path = cog_data_path(self) / "learned_solutions.json"
        self.session = None
        self.cache = {"data": "", "timestamp": None}
        self.learned_data = {}

    async def cog_load(self):
        """Called when the cog is loaded."""
        timeout = aiohttp.ClientTimeout(total=await self.config.timeout())
        self.session = aiohttp.ClientSession(timeout=timeout)
        await self.load_learned_data()

    async def cog_unload(self):
        """Called when the cog is unloaded."""
        if self.session and not self.session.closed:
            await self.session.close()
        await self.save_learned_data()

    async def load_learned_data(self):
        """Loads learned solutions."""
        try:
            if self.learned_db_path.exists():
                async with aiofiles.open(self.learned_db_path, 'r', encoding='utf-8') as f:
                    self.learned_data = json.loads(await f.read())
        except Exception as e:
            print(f"Error loading learned data: {e}")
            self.learned_data = {}

    async def save_learned_data(self):
        """Saves learned solutions."""
        try:
            async with aiofiles.open(self.learned_db_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.learned_data, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Error saving learned data: {e}")

    async def get_github_raw_content(self, url: str) -> str:
        """Fetches raw content from GitHub."""
        try:
            if "github.com" in url and "raw.githubusercontent.com" not in url:
                url = url.replace("github.com", "raw.githubusercontent.com")
                url = url.replace("/blob/", "/")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    return f"Error: GitHub responded with status {response.status}"
        except Exception as e:
            return f"Error fetching GitHub content: {str(e)}"

    async def get_github_repo_content(self, repo_url: str, path: str = "", branch: str = "main") -> str:
        """Fetches content from a GitHub repository with rate limit handling."""
        try:
            parsed = urlparse(repo_url)
            path_parts = parsed.path.strip('/').split('/')
            
            if len(path_parts) < 2:
                return "Invalid GitHub URL"
            
            user, repo = path_parts[0], path_parts[1]
            api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{path}?ref={branch}"
            
            headers = {}
            if os.environ.get('GITHUB_TOKEN'):
                headers['Authorization'] = f"token {os.environ['GITHUB_TOKEN']}"
            
            async with self.session.get(api_url, headers=headers) as response:
                response_text = await response.text()
                
                if response.status == 403 and 'rate limit' in response_text.lower():
                    return "GitHub rate limit reached. Please try again later."
                
                if response.status == 200:
                    content = json.loads(response_text)
                    
                    if isinstance(content, dict) and content.get('type') == 'file':
                        download_url = content.get('download_url')
                        if download_url:
                            async with self.session.get(download_url) as file_response:
                                if file_response.status == 200:
                                    return await file_response.text()
                    
                    elif isinstance(content, list):
                        file_contents = []
                        for item in content:
                            if item.get('type') == 'file' and item.get('name', '').endswith(('.txt', '.md', '.json')):
                                file_url = item.get('download_url')
                                if file_url:
                                    async with self.session.get(file_url) as file_response:
                                        if file_response.status == 200:
                                            file_contents.append(await file_response.text())
                                        await asyncio.sleep(0.1)
                        
                        return "\n\n".join(file_contents) if file_contents else "No suitable files found."
                    
                    return "Could not extract content."
                else:
                    return f"Error: GitHub API responded with status {response.status}"
        except Exception as e:
            return f"Error fetching repository content: {str(e)}"

    async def get_context_data(self) -> str:
        """Fetches context data from configured source with caching."""
        cache_duration = await self.config.cache_duration()
        if (self.cache["timestamp"] and 
            datetime.now() - self.cache["timestamp"] < timedelta(seconds=cache_duration)):
            return self.cache["data"]
        
        source = await self.config.context_source()
        result = ""
        
        if source == "github_raw":
            github_url = await self.config.github_url()
            if github_url:
                result = await self.get_github_raw_content(github_url)
        
        elif source == "github":
            github_url = await self.config.github_url()
            github_path = await self.config.github_path()
            github_branch = await self.config.github_branch()
            if github_url:
                result = await self.get_github_repo_content(github_url, github_path, github_branch)
        
        elif source == "txt":
            if self.data_path.exists():
                async with aiofiles.open(self.data_path, 'r', encoding='utf-8') as f:
                    result = await f.read()
        
        else:
            result = await self.config.context_data()
        
        self.cache = {"data": result, "timestamp": datetime.now()}
        return result

    async def get_message_context(self, message: discord.Message, context_messages: int = 15) -> List[Dict]:
        """Gets message context around a specific message."""
        try:
            messages_before = []
            async for msg in message.channel.history(limit=context_messages//2, before=message, oldest_first=False):
                messages_before.append({
                    "author": msg.author.display_name,
                    "content": msg.clean_content,
                    "timestamp": msg.created_at.isoformat()
                })
            
            messages_after = []
            async for msg in message.channel.history(limit=context_messages//2, after=message, oldest_first=True):
                messages_after.append({
                    "author": msg.author.display_name,
                    "content": msg.clean_content,
                    "timestamp": msg.created_at.isoformat()
                })
            
            all_messages = messages_before[::-1] + [{
                "author": message.author.display_name,
                "content": message.clean_content,
                "timestamp": message.created_at.isoformat(),
                "is_target": True
            }] + messages_after
            
            return all_messages
            
        except Exception as e:
            print(f"Error fetching message context: {e}")
            return []

    async def can_learn(self, user: discord.Member) -> bool:
        """Checks if a user can learn."""
        app_info = await self.bot.application_info()
        if user.id == app_info.owner.id:
            return True
        
        if user.guild_permissions.administrator:
            return True
        
        learning_role_id = await self.config.guild(user.guild).learning_role()
        if learning_role_id:
            learning_role = user.guild.get_role(learning_role_id)
            if learning_role and learning_role in user.roles:
                return True
        
        return False

    async def learn_solution(self, problem_message: discord.Message, solution: str, learner: discord.Member) -> Dict:
        """Saves a learned solution."""
        try:
            context_messages = await self.config.rag_context_messages()
            message_context = await self.get_message_context(problem_message, context_messages)
            
            problem_key = problem_message.clean_content[:100].lower().strip()
            
            learned_entry = {
                "problem": problem_message.clean_content,
                "solution": solution,
                "context": message_context,
                "learned_by": learner.display_name,
                "learned_at": datetime.now().isoformat(),
                "message_id": problem_message.id,
                "channel_id": problem_message.channel.id,
                "guild_id": problem_message.guild.id if problem_message.guild else None
            }
            
            self.learned_data[problem_key] = learned_entry
            await self.save_learned_data()
            
            return learned_entry
            
        except Exception as e:
            print(f"Error saving solution: {e}")
            return None

    async def find_solution(self, question: str) -> Optional[Dict]:
        """Finds a matching solution for a question."""
        question_lower = question.lower().strip()
        
        for problem_key, solution_data in self.learned_data.items():
            if problem_key in question_lower or any(keyword in question_lower for keyword in problem_key.split()[:3]):
                return solution_data
        
        return None

    def split_message(self, text: str, max_length: int = 2000) -> list:
        """Splits a message intelligently without breaking markdown."""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += '\n\n'
                current_chunk += paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    @commands.group()
    @commands.is_owner()
    async def deepseek(self, ctx):
        """Settings for the DeepSeek Cog."""
        pass

    @deepseek.command()
    async def apikey(self, ctx, api_key: str):
        """Sets the DeepSeek API key."""
        await self.config.api_key.set(api_key)
        await ctx.send("API key has been set.")

    @deepseek.command()
    async def context(self, ctx, *, text: str):
        """Sets the context text for queries."""
        await self.config.context_data.set(text)
        self.cache = {"data": "", "timestamp": None}
        await ctx.send("Context has been updated.")

    @deepseek.command()
    async def source(self, ctx, source_type: str):
        """Sets the data source (internal, txt, github, github_raw)."""
        if source_type.lower() in ["internal", "txt", "github", "github_raw"]:
            await self.config.context_source.set(source_type.lower())
            self.cache = {"data": "", "timestamp": None}
            await ctx.send(f"Data source set to {source_type}.")
        else:
            await ctx.send("Invalid source. Allowed: internal, txt, github, github_raw")

    @deepseek.command()
    async def github(self, ctx, url: str):
        """Sets the GitHub URL for data."""
        await self.config.github_url.set(url)
        self.cache = {"data": "", "timestamp": None}
        await ctx.send("GitHub URL has been set.")

    @deepseek.command()
    async def branch(self, ctx, branch: str):
        """Sets the GitHub branch (default: main)."""
        await self.config.github_branch.set(branch)
        self.cache = {"data": "", "timestamp": None}
        await ctx.send(f"GitHub branch set to {branch}.")

    @deepseek.command()
    async def path(self, ctx, path: str):
        """Sets the path in the GitHub repository."""
        await self.config.github_path.set(path)
        self.cache = {"data": "", "timestamp": None}
        await ctx.send(f"GitHub path set to {path}.")

    @deepseek.command()
    async def cache(self, ctx, duration: int):
        """Sets cache duration in seconds (0 disables cache)."""
        await self.config.cache_duration.set(duration)
        self.cache = {"data": "", "timestamp": None}
        await ctx.send(f"Cache duration set to {duration} seconds.")

    @deepseek.command()
    async def timeout(self, ctx, timeout: int):
        """Sets timeout for API requests in seconds."""
        await self.config.timeout.set(timeout)
        if self.session and not self.session.closed:
            await self.session.close()
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        self.session = aiohttp.ClientSession(timeout=timeout_obj)
        await ctx.send(f"Timeout set to {timeout} seconds.")

    @deepseek.command()
    async def contextmessages(self, ctx, count: int):
        """Sets the number of context messages for RAG."""
        if 5 <= count <= 50:
            await self.config.rag_context_messages.set(count)
            await ctx.send(f"Context messages set to {count}.")
        else:
            await ctx.send("Please enter a number between 5 and 50.")

    @deepseek.command()
    @commands.admin_or_permissions(administrator=True)
    async def learnrole(self, ctx, role: discord.Role = None):
        """Sets a role that can learn."""
        if role:
            await self.config.guild(ctx.guild).learning_role.set(role.id)
            await ctx.send(f"Learning role set to {role.name}.")
        else:
            await self.config.guild(ctx.guild).learning_role.set(None)
            await ctx.send("Learning role has been removed.")

    @deepseek.command()
    async def learning(self, ctx, enabled: bool):
        """Enables/disables the learning function."""
        await self.config.learning_enabled.set(enabled)
        status = "enabled" if enabled else "disabled"
        await ctx.send(f"Learning function {status}.")

    @deepseek.command()
    async def reload(self, ctx):
        """Reloads the context data."""
        self.cache = {"data": "", "timestamp": None}
        data = await self.get_context_data()
        await ctx.send(f"Context data reloaded. Length: {len(data)} characters.")

    @commands.command()
    @commands.guild_only()
    async def learn(self, ctx, *, solution: str):
        """Saves a solution for the previous problem."""
        if not await self.config.learning_enabled():
            await ctx.send("The learning function is disabled.")
            return
        
        if not await self.can_learn(ctx.author):
            await ctx.send("You don't have permission to learn.")
            return
        
        target_message = None
        if ctx.message.reference and ctx.message.reference.message_id:
            try:
                target_message = await ctx.channel.fetch_message(ctx.message.reference.message_id)
            except discord.NotFound:
                await ctx.send("Reference message not found.")
                return
        else:
            async for message in ctx.channel.history(limit=10, before=ctx.message):
                if message.author != ctx.author and not message.author.bot:
                    target_message = message
                    break
        
        if not target_message:
            await ctx.send("No suitable message found for learning.")
            return
        
        learned_entry = await self.learn_solution(target_message, solution, ctx.author)
        
        if learned_entry:
            context_preview = "\n".join([
                f"{msg['author']}: {msg['content'][:50]}..."
                for msg in learned_entry['context'][-3:]
            ])
            
            embed = discord.Embed(
                title="✅ Solution Saved",
                description=f"**Problem:** {learned_entry['problem'][:100]}...",
                color=discord.Color.green()
            )
            embed.add_field(name="Solution", value=learned_entry['solution'][:500] + "..." if len(learned_entry['solution']) > 500 else learned_entry['solution'], inline=False)
            embed.add_field(name="Context", value=context_preview or "No context", inline=False)
            embed.add_field(name="Learned by", value=learned_entry['learned_by'], inline=True)
            
            await ctx.send(embed=embed)
        else:
            await ctx.send("Error saving the solution.")

    @commands.command()
    async def ask(self, ctx, *, question: str):
        """Asks a question about the specific topic."""
        if await self.config.learning_enabled():
            solution = await self.find_solution(question)
            if solution:
                embed = discord.Embed(
                    title="🎓 Learned Solution",
                    description=solution['solution'],
                    color=discord.Color.blue()
                )
                embed.set_footer(text=f"Learned by {solution['learned_by']}")
                await ctx.send(embed=embed)
                return
        
        api_key = await self.config.api_key()
        if not api_key:
            await ctx.send("API key is not configured.")
            return

        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=await self.config.timeout())
            self.session = aiohttp.ClientSession(timeout=timeout)

        context_data = await self.get_context_data()
        
        prompt = f"""
        Based on the following information about FUS SkyrimVR modlist:
        
        {context_data}
        
        Answer this question: {question}
