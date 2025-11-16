import logging
import discord
import asyncio
from redbot.core import commands, Config

logging.basicConfig(level=logging.INFO)

class ForumPostNotifier(commands.Cog):
    """A cog to send troubleshooting steps in response to new forum posts."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1234567890)
        default_global = {
            "parent_channel_id": None,
            "troubleshooting_message": "Default troubleshooting message."
        }
        self.config.register_global(**default_global)

    # Command to set the parent channel ID (numeric)
    @commands.command()
    async def setthreadid(self, ctx, channel_id: int):
        """Sets the parent channel ID dynamically via command."""
        await self.config.parent_channel_id.set(channel_id)
        # Versuche, den Kanal zuerst über den Cache zu holen
        channel = ctx.guild.get_channel(channel_id)
        if not channel:
            try:
                channel = await ctx.guild.fetch_channel(channel_id)
            except Exception:
                channel = None
        if channel:
            await ctx.send(f"Parent channel ID has been set to: {channel.mention}")
        else:
            await ctx.send(f"Parent channel ID has been set to: {channel_id} (channel not found)")

    # Command to display the currently tracked parent channel
    @commands.command()
    async def getthreadid(self, ctx):
        """Displays the currently tracked parent channel ID."""
        channel_id = await self.config.parent_channel_id()
        if channel_id is None:
            await ctx.send("No parent channel ID has been set.")
            return

        channel = ctx.guild.get_channel(channel_id)
        if not channel:
            try:
                channel = await ctx.guild.fetch_channel(channel_id)
            except Exception:
                channel = None
        if channel:
            await ctx.send(f"Currently tracked parent channel: {channel.mention}")
        else:
            await ctx.send("The stored parent channel ID is invalid or no longer accessible.")

    # Command to set the troubleshooting message (Admin only)
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmessage(self, ctx, *, message: str):
        """Sets the troubleshooting message."""
        if not message.strip():
            await ctx.send("⚠️ The message cannot be empty!")
            return
        await self.config.troubleshooting_message.set(message)
        await ctx.send("Troubleshooting message has been updated.")

    # Command to display the current troubleshooting message
    @commands.command()
    async def getmessage(self, ctx):
        """Displays the currently set troubleshooting message."""
        message = await self.config.troubleshooting_message()
        await ctx.send(f"Current troubleshooting message: {message}")

    @commands.Cog.listener()
    async def on_thread_create(self, thread: discord.Thread):
        """Listener for when a new thread is created in a forum channel."""
        channel_id = await self.config.parent_channel_id()
        if channel_id is None:
            logging.error("No parent channel ID set. Please set it using the command.")
            return

        # Check if the thread belongs to the configured parent channel
        if thread.parent_id == channel_id:
            logging.info(f"New thread created: {thread.name} (ID: {thread.id})")
            await asyncio.sleep(3)  # Wait a bit for initialization
            message = await self.config.troubleshooting_message()
            if not message:
                message = "No troubleshooting message set. Use !setmessage to configure it."
            try:
                await thread.send(message)
                logging.info(f"Message sent successfully in thread: {thread.name}")
            except discord.Forbidden:
                logging.error(f"Bot lacks permissions to send messages in thread: {thread.name}")
            except discord.HTTPException as e:
                logging.error(f"Failed to send message in thread {thread.name}: {e}")
