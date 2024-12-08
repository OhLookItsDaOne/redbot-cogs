import discord
import time
from redbot.core import commands, Config
from discord.ext import tasks

class KeywordHelp(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=123456789)  # Replace with a unique identifier for your cog
        self.user_help_log = {}  # To track when users were last helped

        # Default configuration
        default_global = {
            "keywords": {},  # {keyword: response}
            "channel_ids": [],  # List of channel IDs to monitor
            "timeout_minutes": 10,  # Default timeout between keyword responses in minutes
        }
        self.config.register_global(**default_global)

    def can_help_user(self, user_id, keyword, timeout_minutes):
        """Check if the user can be helped again based on the cooldown."""
        current_time = time.time()  # Get the current time in seconds
        # Get the last time the user was helped with this keyword
        last_help_time = self.user_help_log.get((user_id, keyword), 0)
        # Calculate the time difference
        time_diff = current_time - last_help_time
        timeout_seconds = timeout_minutes * 60  # Convert timeout from minutes to seconds
        return time_diff > timeout_seconds  # Return True if cooldown is over

    def log_help(self, user_id, keyword):
        """Log the time when a user was helped with a keyword."""
        self.user_help_log[(user_id, keyword)] = time.time()  # Store the current time

    @commands.Cog.listener()
    async def on_message(self, message):
        """Listen for messages and respond to keywords."""
        if message.author.bot:
            return  # Ignore messages from bots, including itself.

        # Check if the message is in a monitored channel
        channel_ids = await self.config.channel_ids()
        if message.channel.id not in channel_ids:
            return  # Ignore messages from unmonitored channels.

        content = message.content.lower()
        keywords = await self.config.keywords()

        # Check for keywords in the message
        for keyword, response in keywords.items():
            # Handle mention with keyword (not thread-specific)
            if self.bot.user.mentioned_in(message) and keyword in content:
                # Mention user by ID to ensure they are @mentioned
                await message.channel.send(f"<@{message.author.id}> {response}")
                break  # Avoid responding multiple times for multiple keywords

            # If it's not a thread and keyword matches
            if isinstance(message.channel, discord.TextChannel) and keyword in content:
                user_id = message.author.id

                # Check if the user was recently helped
                timeout_minutes = await self.config.timeout_minutes()
                if self.can_help_user(user_id, keyword, timeout_minutes):
                    # Mention user by ID to ensure they are @mentioned
                    await message.channel.send(f"<@{message.author.id}> {response}")
                    self.log_help(user_id, keyword)
                break  # Avoid responding multiple times for multiple keywords

    @commands.Cog.listener()
    async def on_thread_create(self, thread):
        """Automatically respond when a thread is created in a monitored channel."""
        # Only respond if the thread is in a monitored channel
        channel_ids = await self.config.channel_ids()
        if thread.parent.id not in channel_ids:
            return

        # Fetch all messages in the thread, starting with the first one
        try:
            # Fetch messages in the thread
            await thread.fetch()
            first_message = thread.messages[0]
            content = first_message.content.lower()
            keywords = await self.config.keywords()

            # Check for any keyword in the first message
            for keyword, response in keywords.items():
                if keyword in content:
                    # Mention user by ID to ensure they are @mentioned
                    await thread.send(f"<@{first_message.author.id}> {response}")
                    self.log_help(first_message.author.id, keyword)
                    break
        except Exception as e:
            # Log an error if fetching messages fails
            print(f"Error fetching messages in thread {thread.id}: {e}")

    @commands.group()
    async def keyword_help(self, ctx):
        """Base command for managing keywords and monitored channels."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help(ctx.command)

    @keyword_help.command()
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

    @keyword_help.command()
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

    @keyword_help.command()
    async def settimeout(self, ctx, timeout_minutes: int):
        """Set the timeout (in minutes) between user help responses."""
        if not await self.bot.is_owner(ctx.author):
            await ctx.send("You do not have permission to set the timeout.")
            return

        await self.config.timeout_minutes.set(timeout_minutes)
        await ctx.send(f"Timeout set to {timeout_minutes} minutes.")

    @keyword_help.command()
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

    @keyword_help.command()
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

    @keyword_help.command()
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
