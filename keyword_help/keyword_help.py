import discord
import time
import difflib
from redbot.core import commands, Config
import re
import logging

class KeywordHelp(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=123456789)
        self.logger = logging.getLogger(__name__)

        # Default settings
        default_global = {
            "keywords": {},  # {keyword: response}
            "channel_ids": [],  # List of monitored channel IDs
            "timeout_minutes": 10,  # Cooldown time
            "debug_channel_id": None,  # Debug channel ID
            "user_help_times": {},  # Tracks user help cooldowns
            "ignored_roles": []  # Role IDs to ignore
        }
        self.config.register_global(**default_global)

    async def can_help_user(self, user_id, keyword, timeout_minutes):
        """Check if user can be helped again based on cooldown."""
        current_time = time.time()
        user_help_times = await self.config.user_help_times()
        last_help_time = user_help_times.get(str(user_id), {}).get(keyword, 0)
        return (current_time - last_help_time) > (timeout_minutes * 60)

    async def log_help(self, user_id, keyword):
        """Log the time when a user was helped."""
        current_time = time.time()
        user_help_times = await self.config.user_help_times()
        if str(user_id) not in user_help_times:
            user_help_times[str(user_id)] = {}
        user_help_times[str(user_id)][keyword] = current_time
        await self.config.user_help_times.set(user_help_times)

    def normalize_string(self, string):
        """Normalize a string by removing extra spaces and converting to lowercase."""
        return re.sub(r'\s+', ' ', string.lower()).strip()

    def match_keywords(self, content, keywords, mentioned):
        """Match keywords with tolerance for errors."""
        matched_keywords = []
        normalized_content = self.normalize_string(content)

        for keyword, response in keywords.items():
            normalized_keyword = self.normalize_string(keyword)

            # Exact match
            if normalized_keyword in normalized_content:
                matched_keywords.append((keyword, response))
            # Fuzzy match (only if mentioned)
            elif mentioned:
                # Fuzzy matching with SequenceMatcher for slight variations
                similarity = difflib.SequenceMatcher(None, normalized_content, normalized_keyword).ratio()
                if similarity > 0.5:  # Lowered the threshold for fuzzy matching
                    matched_keywords.append((keyword, response))
            # Alternative: Fuzzy match even without mention but for highly relevant cases
            else:
                similarity = difflib.SequenceMatcher(None, normalized_content, normalized_keyword).ratio()
                if similarity > 0.5:  # Further lowered threshold for non-mentioned cases
                    matched_keywords.append((keyword, response))

        return matched_keywords

    async def user_has_ignored_role(self, user):
        """Check if user has an ignored role."""
        ignored_roles = await self.config.ignored_roles()
        return any(role.id in ignored_roles for role in user.roles)

    async def log_error(self, error):
        """Log errors to a debug channel."""
        debug_channel_id = await self.config.debug_channel_id()
        if debug_channel_id:
            channel = self.bot.get_channel(debug_channel_id)
            if channel:
                await channel.send(f"Error: {error}")
        self.logger.error(error)

    @commands.Cog.listener()
    async def on_message(self, message):
        """Listen for keywords and respond appropriately."""
        if message.author.bot or message.channel.id not in await self.config.channel_ids():
            return

        mentioned = self.bot.user in message.mentions
        if await self.user_has_ignored_role(message.author):
            return

        keywords = await self.config.keywords()
        matched_keywords = self.match_keywords(message.content, keywords, mentioned)

        if not matched_keywords:
            return

        response_message = f"<@{message.author.id}> I found the following keywords:\n"
        timeout_minutes = await self.config.timeout_minutes()
        valid_responses = []

        for keyword, response in matched_keywords:
            if mentioned or await self.can_help_user(message.author.id, keyword, timeout_minutes):
                valid_responses.append(f"**{keyword.capitalize()}**: {response}")
                await self.log_help(message.author.id, keyword)

        if valid_responses:
            response_message += "\n".join(valid_responses)
            await message.channel.send(response_message)

    @commands.Cog.listener()
    async def on_thread_create(self, thread: discord.Thread):
        """Handles new thread creation and scans the first 3 messages for keywords."""
        # Get the creator of the thread
        creator = thread.owner

        # Get the first 3 messages in the thread
        messages = []
        async for message in thread.history(limit=3):  # Limit to first 3 messages
            messages.append(message)

        # Check if we should skip the cooldown check for these first messages
        timeout_minutes = await self.config.timeout_minutes()

        keywords = await self.config.keywords()
        for message in messages:
            if message.author == creator:
                mentioned = self.bot.user in message.mentions
                matched_keywords = self.match_keywords(message.content, keywords, mentioned)

                if matched_keywords:
                    response_message = f"<@{message.author.id}> I found the following keywords in your thread:\n"
                    valid_responses = []

                    for keyword, response in matched_keywords:
                        valid_responses.append(f"**{keyword.capitalize()}**: {response}")
                        await self.log_help(message.author.id, keyword)

                    if valid_responses:
                        response_message += "\n".join(valid_responses)
                        await message.channel.send(response_message)

    @commands.group(name="kw")
    async def kw(self, ctx):
        """Manage keywords and settings."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help(ctx.command)

    @kw.command()
    async def list(self, ctx):
        """List all available commands for the keyword manager."""
        commands_list = """
        Here are the available commands for managing keywords:

        **!kw addkeyword <keyword> <response>** - Add a new keyword and response
        **!kw removekeyword <keyword>** - Remove a keyword
        **!kw settimeout <minutes>** - Set the cooldown period for user responses
        **!kw addchannel <channel>** - Add a channel to the monitored list
        **!kw removechannel <channel>** - Remove a channel from the monitored list
        **!kw setdebugchannel <channel>** - Set a debug channel for logging errors
        **!kw addignoredrole <role>** - Add a role to the ignored roles list
        **!kw removeignoredrole <role>** - Remove a role from the ignored roles list

        Usage: Type `!kw <command>` to execute any of the above actions.
        """
        await ctx.send(commands_list)

    @kw.command()
    async def conf(self, ctx):
        """Display the current configuration of keywords and monitored channels."""
        keywords = await self.config.keywords()
        channel_ids = await self.config.channel_ids()
        timeout_minutes = await self.config.timeout_minutes()

        # Get the channel names for the IDs
        channel_mentions = [self.bot.get_channel(channel_id).mention for channel_id in channel_ids]

        response_message = "Current Keyword Configuration:\n"
        response_message += f"**Timeout (Cooldown)**: {timeout_minutes} minutes\n\n"

        if keywords:
            response_message += "**Keywords:**\n"
            for keyword, response in keywords.items():
                response_message += f"**{keyword}**: {response}\n"
        else:
            response_message += "**No keywords configured.**\n"

        if channel_mentions:
            response_message += "\n**Monitored Channels:**\n" + "\n".join(channel_mentions)
        else:
            response_message += "\n**No channels monitored.**\n"

        await ctx.send(response_message)

    @kw.command()
    async def addkeyword(self, ctx, keyword: str, response: str):
        """Add a keyword-response pair."""
        if not ctx.author.guild_permissions.administrator:
            await ctx.send("You need to be an admin to add keywords.")
            return

        keywords = await self.config.keywords()
        keywords[keyword] = response
        await self.config.keywords.set(keywords)
        await ctx.send(f"Added keyword: `{keyword}` with response: `{response}`")

    @kw.command()
    async def removekeyword(self, ctx, keyword: str):
        """Remove a keyword."""
        if not ctx.author.guild_permissions.administrator:
            await ctx.send("You need to be an admin to remove keywords.")
            return

        keywords = await self.config.keywords()
        if keyword in keywords:
            del keywords[keyword]
            await self.config.keywords.set(keywords)
            await ctx.send(f"Removed keyword: `{keyword}`")
        else:
            await ctx.send(f"Keyword `{keyword}` not found.")

    @kw.command()
    async def settimeout(self, ctx, minutes: int):
        """Set the cooldown duration in minutes."""
        if not ctx.author.guild_permissions.administrator:
            await ctx.send("You need to be an admin to set the timeout.")
            return

        await self.config.timeout_minutes.set(minutes)
        await ctx.send(f"Timeout set to {minutes} minutes.")

    @kw.command()
    async def addchannel(self, ctx, channel: discord.TextChannel):
        """Add a channel to the monitored list."""
        if not ctx.author.guild_permissions.administrator:
            await ctx.send("You need to be an admin to manage monitored channels.")
            return

        channel_ids = await self.config.channel_ids()
        if channel.id not in channel_ids:
            channel_ids.append(channel.id)
            await self.config.channel_ids.set(channel_ids)
            await ctx.send(f"Added channel {channel.mention} to the monitored list.")

    @kw.command()
    async def removechannel(self, ctx, channel: discord.TextChannel):
        """Remove a channel from the monitored list."""
        if not ctx.author.guild_permissions.administrator:
            await ctx.send("You need to be an admin to manage monitored channels.")
            return

        channel_ids = await self.config.channel_ids()
        if channel.id in channel_ids:
            channel_ids.remove(channel.id)
            await self.config.channel_ids.set(channel_ids)
            await ctx.send(f"Removed channel {channel.mention} from the monitored list.")

    @kw.command()
    async def setdebugchannel(self, ctx, channel: discord.TextChannel):
        """Set the debug log channel."""
        if not ctx.author.guild_permissions.administrator:
            await ctx.send("You need to be an admin to set the debug channel.")
            return

        await self.config.debug_channel_id.set(channel.id)
        await ctx.send(f"Set debug channel to {channel.mention}.")

    @kw.command()
    async def addignoredrole(self, ctx, role: discord.Role):
        """Add a role to the ignored list."""
        if not ctx.author.guild_permissions.administrator:
            await ctx.send("You need to be an admin to manage ignored roles.")
            return

        ignored_roles = await self.config.ignored_roles()
        if role.id not in ignored_roles:
            ignored_roles.append(role.id)
            await self.config.ignored_roles.set(ignored_roles)
            await ctx.send(f"Added role {role.name} to ignored list.")

    @kw.command()
    async def removeignoredrole(self, ctx, role: discord.Role):
        """Remove a role from the ignored list."""
        if not ctx.author.guild_permissions.administrator:
            await ctx.send("You need to be an admin to manage ignored roles.")
            return

        ignored_roles = await self.config.ignored_roles()
        if role.id in ignored_roles:
            ignored_roles.remove(role.id)
            await self.config.ignored_roles.set(ignored_roles)
            await ctx.send(f"Removed role {role.name} from ignored list.")

def setup(bot):
    bot.add_cog(KeywordHelp(bot))
