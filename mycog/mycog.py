import logging
import discord
import asyncio
from redbot.core import commands

# Configure logging
logging.basicConfig(level=logging.INFO)

class ForumPostNotifier(commands.Cog):
    """A cog to send troubleshooting steps in response to new forum posts."""

    def __init__(self, bot):
        self.bot = bot
        self.parent_channel_id = None  # Store the parent channel ID
        self.message_content = "Default troubleshooting message."  # Default message

    # Command to set the parent channel ID (using an integer)
    @commands.command()
    async def setthreadid(self, ctx, channel_id: int):
        """Sets the parent channel ID dynamically via command."""
        self.parent_channel_id = channel_id
        # Versuch, den Kanal über den Cache zu holen
        channel = ctx.guild.get_channel(channel_id)
        # Falls nicht im Cache, versuche den Kanal direkt abzurufen
        if not channel:
            try:
                channel = await ctx.guild.fetch_channel(channel_id)
            except Exception as e:
                logging.error(f"Error fetching channel: {e}")
                channel = None
        if channel:
            await ctx.send(f"Parent channel ID has been set to: {channel.mention}")
        else:
            await ctx.send(f"Parent channel ID has been set to: {channel_id} (channel not found)")

    # Command to display the currently tracked parent channel
    @commands.command()
    async def getthreadid(self, ctx):
        """Displays the currently tracked parent channel ID."""
        if self.parent_channel_id is not None:
            channel = ctx.guild.get_channel(self.parent_channel_id)
            if not channel:
                try:
                    channel = await ctx.guild.fetch_channel(self.parent_channel_id)
                except Exception as e:
                    logging.error(f"Error fetching channel: {e}")
                    channel = None
            if channel:
                await ctx.send(f"Currently tracked parent channel: {channel.mention}")
            else:
                await ctx.send("The stored parent channel ID is invalid or no longer accessible.")
        else:
            await ctx.send("No parent channel ID has been set.")

    # Command to set the troubleshooting message (Admin only)
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setmessage(self, ctx, *, message: str):
        """Sets the troubleshooting message."""
        self.message_content = message
        await ctx.send("Troubleshooting message has been updated.")

    # Command to get the current troubleshooting message
    @commands.command()
    async def getmessage(self, ctx):
        """Displays the currently set troubleshooting message."""
        await ctx.send(f"Current troubleshooting message: {self.message_content}")

    @commands.Cog.listener()
    async def on_thread_create(self, thread: discord.Thread):
        """Listener for when a new thread is created in a forum channel."""
        if self.parent_channel_id is None:
            logging.error("No parent channel ID set. Please set it using the command.")
            return

        # Check if the thread belongs to the configured parent channel
        if thread.parent_id == self.parent_channel_id:
            logging.info(f"New thread created: {thread.name} (ID: {thread.id})")
            # Wait a bit to ensure the thread is fully initialized
            await asyncio.sleep(3)
            # Send the troubleshooting message
            try:
                await thread.send(self.message_content)
                logging.info(f"Message sent successfully in thread: {thread.name}")
            except discord.Forbidden:
                logging.error(f"Bot lacks permissions to send messages in thread: {thread.name}")
            except discord.HTTPException as e:
                logging.error(f"Failed to send message in thread {thread.name}: {e}")
