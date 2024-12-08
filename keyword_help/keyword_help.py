import discord
import time
from redbot.core import commands, Config
import logging

class KeywordHelp(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=123456789)  # Unique identifier for the cog
        self.user_help_log = {}  # Track user keyword responses

        # Default configuration
        default_global = {
            "keywords": {},  # {keyword: response}
            "channel_ids": [],  # List of channel IDs to monitor
            "timeout_minutes": 10,  # Timeout duration in minutes
            "debug_channel_id": None  # ID for the debug channel to log errors
        }
        self.config.register_global(**default_global)

        # Set up logging
        self.logger = logging.getLogger(__name__)

    def can_help_user(self, user_id, keyword, timeout_minutes):
        """Check if the user can be helped again based on the cooldown."""
        current_time = time.time()
        last_help_time = self.user_help_log.get((user_id, keyword), 0)
        time_diff = current_time - last_help_time
        timeout_seconds = timeout_minutes * 60
        return time_diff > timeout_seconds

    def log_help(self, user_id, keyword):
        """Log the time when a user was helped with a keyword."""
        self.user_help_log[(user_id, keyword)] = time.time()

    async def log_error(self, error):
        """Logs the error to the debug channel if specified."""
        debug_channel_id = await self.config.debug_channel_id()
        if debug_channel_id:
            channel = self.bot.get_channel(debug_channel_id)
            if channel:
                await channel.send(f"Error: {error}")
        self.logger.error(error)

    @commands.Cog.listener()
    async def on_message(self, message):
        """Listen for messages and respond to keywords."""
        if message.author.bot:
            return  # Ignore bot messages

        # Check if the message is in a monitored channel
        channel_ids = await self.config.channel_ids()
        if message.channel.id not in channel_ids:
            return

        content = message.content.lower()
        keywords = await self.config.keywords()

        for keyword, response in keywords.items():
            if self.bot.user.mentioned_in(message) and keyword in content:
                await message.channel.send(f"<@{message.author.id}> {response}")
                break

            if isinstance(message.channel, discord.TextChannel) and keyword in content:
                user_id = message.author.id
                timeout_minutes = await self.config.timeout_minutes()
                if self.can_help_user(user_id, keyword, timeout_minutes):
                    await message.channel.send(f"<@{message.author.id}> {response}")
                    self.log_help(user_id, keyword)
                break

    @commands.Cog.listener()
    async def on_thread_create(self, thread):
        """Respond when a new thread is created in a monitored channel."""
        try:
            # Only respond if the thread is in a monitored channel
            channel_ids = await self.config.channel_ids()
            if thread.parent.id not in channel_ids:
                return

            # Fetch the first message of the thread
            first_message = await thread.fetch_message(thread.id)
            content = first_message.content.lower()
            keywords = await self.config.keywords()

            # Check for keywords in the thread's first message
            for keyword, response in keywords.items():
                if keyword in content:
                    await thread.send(f"<@{first_message.author.id}> {response}")
                    self.log_help(first_message.author.id, keyword)
                    break
        except Exception as e:
            await self.log_error(f"Error in on_thread_create: {e}")

    @commands.group()
    async def kwhelp(self, ctx):
        """Base command for managing keywords and monitored channels."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help(ctx.command)

    @kwhelp.command()
    async def addkeyword(self, ctx, keyword: str, response: str):
        """Add a new keyword and response pair."""
        if not await self.bot.is_owner(ctx.author):
            await ctx.send("You do not have permission to add keywords.")
            return

        # Retrieve current keywords from the config
        keywords = await self.config.keywords()
        keywords[keyword] = response
        await self.config.keywords.set(keywords)
        await ctx.send(f"Added new keyword: `{keyword}` with response: `{response}`")

    @kwhelp.command()
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

    @kwhelp.command()
    async def settimeout(self, ctx, timeout_minutes: int):
        """Set the timeout (in minutes) between user help responses."""
        if not await self.bot.is_owner(ctx.author):
            await ctx.send("You do not have permission to set the timeout.")
            return

        await self.config.timeout_minutes.set(timeout_minutes)
        await ctx.send(f"Timeout set to {timeout_minutes} minutes.")

    @kwhelp.command()
    async def addchannel(self, ctx, channel_id: int):
        """Add a channel to the monitored list."""
        if not await self.bot.is_owner(ctx.author):
            await ctx.send("You do not have permission to add channels.")
            return

        channel_ids = await self.config.channel_ids()
        if channel_id not in channel_ids:
            channel_ids.append(channel_id)
            await self.config.channel_ids.set(channel_ids)
            await ctx.send(f"Added channel ID {channel_id} to the monitored channels.")
        else:
            await ctx.send(f"Channel ID {channel_id} is already in the monitored list.")

    @kwhelp.command()
    async def removechannel(self, ctx, channel_id: int):
        """Remove a channel from the monitored list."""
        if not await self.bot.is_owner(ctx.author):
            await ctx.send("You do not have permission to remove channels.")
            return

        channel_ids = await self.config.channel_ids()
        if channel_id in channel_ids:
            channel_ids.remove(channel_id)
            await self.config.channel_ids.set(channel_ids)
            await ctx.send(f"Removed channel ID {channel_id} from the monitored channels.")
        else:
            await ctx.send(f"Channel ID {channel_id} not found in the monitored list.")

    @kwhelp.command()
    async def showconfig(self, ctx):
        """Show the current configuration for the cog."""
        timeout_minutes = await self.config.timeout_minutes()
        keywords = await self.config.keywords()
        channel_ids = await self.config.channel_ids()
        
        # Show configuration details
        config_message = (
            f"Timeout: {timeout_minutes} minutes\n"
            f"Keywords: {', '.join(keywords.keys()) if keywords else 'No keywords added'}\n"
            f"Monitored channels: {', '.join(map(str, channel_ids)) if channel_ids else 'No channels added'}"
        )
        await ctx.send(config_message)

    @kwhelp.command()
    async def setdebugchannel(self, ctx, channel_id: int):
        """Set the channel where errors will be logged."""
        if not await self.bot.is_owner(ctx.author):
            await ctx.send("You do not have permission to set the debug channel.")
            return

        await self.config.debug_channel_id.set(channel_id)
        await ctx.send(f"Debugging channel set to {channel_id}. All errors will be logged there.")
