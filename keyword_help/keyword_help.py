import discord
import time
import difflib
from redbot.core import commands, Config
import logging
import re

class KeywordHelp(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=123456789)  # Unique identifier for the cog
        self.user_help_log = {}  # Track user keyword responses (in memory for now)

        # Default configuration
        default_global = {
            "keywords": {},  # {keyword: response}
            "channel_ids": [],  # List of channel IDs to monitor
            "timeout_minutes": 10,  # Timeout duration in minutes
            "debug_channel_id": None,  # ID for the debug channel to log errors
            "user_help_times": {},  # User help log, persist across restarts
            "ignored_roles": []  # List of role IDs to ignore
        }
        self.config.register_global(**default_global)

        # Set up logging
        self.logger = logging.getLogger(__name__)

    async def can_help_user(self, user_id, keyword, timeout_minutes):
        """Check if the user can be helped again based on the cooldown."""
        current_time = time.time()

        # Retrieve the user's last help time from the persistent config
        user_help_times = await self.config.user_help_times()  # Dictionary of user ID => keyword => timestamp
        last_help_time = user_help_times.get(str(user_id), {}).get(keyword, 0)
        time_diff = current_time - last_help_time
        timeout_seconds = timeout_minutes * 60
        return time_diff > timeout_seconds

    async def log_help(self, user_id, keyword):
        """Log the time when a user was helped with a keyword."""
        current_time = time.time()
        user_help_times = await self.config.user_help_times()
        if str(user_id) not in user_help_times:
            user_help_times[str(user_id)] = {}

        # Log the help time for this keyword
        user_help_times[str(user_id)][keyword] = current_time
        await self.config.user_help_times.set(user_help_times)

    async def log_error(self, error):
        """Logs the error to the debug channel if specified."""
        debug_channel_id = await self.config.debug_channel_id()
        if debug_channel_id:
            channel = self.bot.get_channel(debug_channel_id)
            if channel:
                await channel.send(f"Error: {error}")
        self.logger.error(error)

    def normalize_string(self, string):
        """Normalize strings by removing excessive spaces and converting to lowercase."""
        return re.sub(r'\s+', ' ', string.lower()).strip()

    def match_keywords_in_sentence(self, content, keywords, mentioned):
        """Match keywords in a sentence, allowing for minor typos or missing spaces."""
        matched_keywords = []
        normalized_content = self.normalize_string(content)
        
        print(f"Normalized content: {normalized_content}")  # Debug: Check normalized content
        
        for keyword, response in keywords.items():
            normalized_keyword = self.normalize_string(keyword)
            
            # Debugging logs for keyword normalization
            print(f"Checking keyword: '{keyword}' -> '{normalized_keyword}'")  # Debug: Checking normalized keyword

            # Check for exact match first
            if normalized_keyword in normalized_content:
                matched_keywords.append((keyword, response))
                continue

            # Fuzzy matching for more tolerance (if exact match fails) - only if mentioned
            if mentioned:
                ratio = difflib.SequenceMatcher(None, normalized_content, normalized_keyword).ratio()
                print(f"Fuzzy match ratio: {ratio}")  # Debug: Check fuzzy match ratio
                if ratio > 0.8:  # 80% similarity threshold
                    matched_keywords.append((keyword, response))

        return matched_keywords

    async def user_has_ignored_role(self, user):
        """Check if the user has any ignored roles."""
        ignored_roles = await self.config.ignored_roles()
        user_roles = [role.id for role in user.roles]
        return any(role in ignored_roles for role in user_roles)

    @commands.Cog.listener()
    async def on_message(self, message):
        """Listen for messages and respond to keywords."""
        if message.author.bot:
            return  # Ignore bot messages

        # Check if the message is in a monitored channel
        channel_ids = await self.config.channel_ids()
        if message.channel.id not in channel_ids:
            return

        # If the bot is mentioned, skip timeout checks
        mentioned = any(mention.id == self.bot.user.id for mention in message.mentions)

        content = message.content.lower()
        keywords = await self.config.keywords()  # Fetch keywords directly from config

        # Check if the user has an ignored role
        if await self.user_has_ignored_role(message.author):
            return  # Do not respond to users with ignored roles

        # Use the match_keywords_in_sentence function for better keyword matching
        matched_keywords = self.match_keywords_in_sentence(content, keywords, mentioned)

        if matched_keywords:
            response_message = f"<@{message.author.id}> I found the following keywords:\n"
            for keyword, response in matched_keywords:
                # Ensure that the user can only be helped after the timeout, unless the bot is mentioned
                timeout_minutes = await self.config.timeout_minutes()
                if mentioned or await self.can_help_user(message.author.id, keyword, timeout_minutes):
                    response_message += f"**{keyword.capitalize()}**: {response}\n"
                    await self.log_help(message.author.id, keyword)  # Log the help time for this keyword
                # If the user is on cooldown and the bot is not mentioned, skip responding
                else:
                    continue  # Skip sending any message for this keyword if it's on cooldown

            # Only send response if there are matched keywords
            if response_message.strip() != f"<@{message.author.id}> I found the following keywords:\n":
                await message.channel.send(response_message)
        else:
            # No valid keyword matched, do nothing (only reply if timeout not exceeded and mentioned)
            timeout_minutes = await self.config.timeout_minutes()
            if mentioned:
                await message.channel.send(f"<@{message.author.id}> No matching keywords found.")
            # If no mention and no match, don't reply
            print(f"No keywords matched for message: {message.content}")  # Debug: No matches case

    @commands.group(name="kw")
    async def kw(self, ctx):
        """Base command for managing keywords and monitored channels."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help(ctx.command)

    @kw.command()
    async def addkeyword(self, ctx, keyword: str, response: str):
        """Add a new keyword and response pair."""
        if not await self.bot.is_owner(ctx.author):
            await ctx.send("You do not have permission to add keywords.")
            return

        # Validate the keyword format (check if multi-word keywords are enclosed in quotes)
        if " " in keyword and not (keyword.startswith('"') and keyword.endswith('"')):
            await ctx.send(f"Invalid keyword format. Please wrap multi-word keywords in quotes like this: \"black square\".")
            return

        # Retrieve current keywords from the config
        keywords = await self.config.keywords()
        if keyword in keywords:
            # Update the existing keyword
            keywords[keyword] = response
            await self.config.keywords.set(keywords)
            await ctx.send(f"Updated keyword: `{keyword}` with new response: `{response}`")
        else:
            # Add new keyword
            keywords[keyword] = response
            await self.config.keywords.set(keywords)
            await ctx.send(f"Added new keyword: `{keyword}` with response: `{response}`")

    @kw.command()
    async def removekeyword(self, ctx, keyword: str):
        """Remove a keyword from the configuration."""
        if not await self.bot.is_owner(ctx.author):
            await ctx.send("You do not have permission to remove keywords.")
            return

        # Retrieve current keywords from the config
        keywords = await self.config.keywords()
        if keyword in keywords:
            del keywords[keyword]
            await self.config.keywords.set(keywords)
            await ctx.send(f"Removed keyword: `{keyword}`")
        else:
            await ctx.send(f"Keyword `{keyword}` not found.")

    @kw.command()
    async def settimeout(self, ctx, timeout_minutes: int):
        """Set the timeout (in minutes) between user help responses."""
        if not await self.bot.is_owner(ctx.author):
            await ctx.send("You do not have permission to set the timeout.")
            return

        await self.config.timeout_minutes.set(timeout_minutes)
        await ctx.send(f"Timeout set to {timeout_minutes} minutes.")

    @kw.command()
    async def addchannel(self, ctx, channel_id: int):
        """Add a channel to the monitored list."""
        if not await self.bot.is_owner(ctx.author):
            await ctx.send("You do not have permission to add channels.")
            return

        channel_ids = await self.config.channel_ids()
        if channel_id not in channel_ids:
            channel_ids.append(channel_id)
            await self.config.channel_ids.set(channel_ids)
            await ctx.send(f"Added channel <#{channel_id}> to the monitored channels.")
        else:
            await ctx.send(f"Channel <#{channel_id}> is already in the monitored list.")

    @kw.command()
    async def removechannel(self, ctx, channel_id: int):
        """Remove a channel from the monitored list."""
        if not await self.bot.is_owner(ctx.author):
            await ctx.send("You do not have permission to remove channels.")
            return

        channel_ids = await self.config.channel_ids()
        if channel_id in channel_ids:
            channel_ids.remove(channel_id)
            await self.config.channel_ids.set(channel_ids)
            await ctx.send(f"Removed channel <#{channel_id}> from the monitored list.")
        else:
            await ctx.send(f"Channel <#{channel_id}> not found in the monitored list.")

    @kw.command()
    async def setdebugchannel(self, ctx, channel_id: int):
        """Set a debug channel to log errors.""" 
        if not await self.bot.is_owner(ctx.author):
            await ctx.send("You do not have permission to set the debug channel.")
            return

        await self.config.debug_channel_id.set(channel_id)
        await ctx.send(f"Debug channel set to <#{channel_id}>.")

    @kw.command()
    async def addignoredrole(self, ctx, role_id: int):
        """Add a role to the ignored roles list."""
        if not await self.bot.is_owner(ctx.author):
            await ctx.send("You do not have permission to modify ignored roles.")
            return

        ignored_roles = await self.config.ignored_roles()
        if role_id not in ignored_roles:
            ignored_roles.append(role_id)
            await self.config.ignored_roles.set(ignored_roles)
            await ctx.send(f"Added role <@&{role_id}> to ignored roles.")
        else:
            await ctx.send(f"Role <@&{role_id}> is already in the ignored list.")

    @kw.command()
    async def removeignoredrole(self, ctx, role_id: int):
        """Remove a role from the ignored roles list."""
        if not await self.bot.is_owner(ctx.author):
            await ctx.send("You do not have permission to modify ignored roles.")
            return

        ignored_roles = await self.config.ignored_roles()
        if role_id in ignored_roles:
            ignored_roles.remove(role_id)
            await self.config.ignored_roles.set(ignored_roles)
            await ctx.send(f"Removed role <@&{role_id}> from ignored roles.")
        else:
            await ctx.send(f"Role <@&{role_id}> not found in ignored roles.")

def setup(bot):
    bot.add_cog(KeywordHelp(bot))
