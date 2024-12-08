import discord
import time
from redbot.core import commands, Config
import logging

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
            "user_help_times": {}  # User help log, persist across restarts
        }
        self.config.register_global(**default_global)

        # Set up logging
        self.logger = logging.getLogger(__name__)

    def can_help_user(self, user_id, keyword, timeout_minutes):
        """Check if the user can be helped again based on the cooldown."""
        current_time = time.time()

        # Retrieve the user's last help time from the persistent config
        user_help_times = self.config.user_help_times()  # Dictionary of user ID => keyword => timestamp
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

    def is_valid_keyword(self, keyword):
        """Check if a keyword is valid (must be wrapped in quotes if it contains spaces)."""
        if " " in keyword and not (keyword.startswith('"') and keyword.endswith('"')):
            return False
        return True

    async def get_all_keywords(self):
        """Get all keywords and responses."""
        keywords = await self.config.keywords()
        return keywords

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
        keywords = await self.get_all_keywords()

        matched_keywords = []
        # Check for exact matches (multi-word should come first)
        for keyword, response in keywords.items():
            if keyword in content:
                matched_keywords.append((keyword, response))

        if matched_keywords:
            # Generate a unique response for each matched keyword
            response_message = ""
            for keyword, response in matched_keywords:
                # Ensure that the user can only be helped after the timeout
                timeout_minutes = await self.config.timeout_minutes()
                if await self.can_help_user(message.author.id, keyword, timeout_minutes):
                    response_message += f"**{keyword.capitalize()}**: {response}\n"
                    await self.log_help(message.author.id, keyword)  # Log the help time for this keyword
                else:
                    # Notify user about the cooldown if needed
                    response_message += f"**{keyword.capitalize()}**: You need to wait before I can help again.\n"

            await message.channel.send(f"<@{message.author.id}> {response_message}")
        else:
            # Only respond if there are keywords, no generic response
            pass  # Nothing happens here unless a keyword is found.

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
    async def showconfig(self, ctx):
        """Show the current configuration for the cog."""
        if not await self.bot.is_owner(ctx.author):
            await ctx.send("You do not have permission to view the configuration.")
            return

        timeout_minutes = await self.config.timeout_minutes()
        keywords = await self.config.keywords()
        channel_ids = await self.config.channel_ids()
        
        # Show configuration details
        config_message = (
            f"Timeout: {timeout_minutes} minutes\n"
            f"Keywords: {', '.join(keywords.keys()) if keywords else 'No keywords added'}\n"
            f"Monitored channels: {', '.join([f'<#{channel_id}>' for channel_id in channel_ids]) if channel_ids else 'No channels added'}"
        )
        await ctx.send(config_message)

    @kw.command(name="list")
    async def kwlist(self, ctx):
        """List all available commands for the keyword help cog."""
        command_list = """
**Keyword Help Cog Commands:**
- `!kw addkeyword <keyword> <response>`: Add a new keyword and response pair.
- `!kw removekeyword <keyword>`: Remove a keyword from the configuration.
- `!kw settimeout <timeout>`: Set the timeout (in minutes) for user help responses.
- `!kw addchannel <channel_id>`: Add a channel to the monitored list.
- `!kw removechannel <channel_id>`: Remove a channel from the monitored list.
- `!kw showconfig`: Show the current configuration of the cog.
"""
        await ctx.send(command_list)

    @kw.command()
    async def setdebugchannel(self, ctx, channel_id: int):
        """Set a debug channel to log errors."""
        if not await self.bot.is_owner(ctx.author):
            await ctx.send("You do not have permission to set the debug channel.")
            return

        await self.config.debug_channel_id.set(channel_id)
        await ctx.send(f"Debug channel set to <#{channel_id}>.")

def setup(bot):
    bot.add_cog(KeywordHelp(bot))
