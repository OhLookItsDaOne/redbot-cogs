import logging
import discord
import asyncio
from redbot.core import commands, Config

# Configure logging
logging.basicConfig(level=logging.INFO)

class ForumPostNotifier(commands.Cog):
    """A cog to send troubleshooting steps in response to new forum posts."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1234567890)
        self.config.register_global(parent_channel_id=None, troubleshooting_message="Default troubleshooting message")

    # Command to set the parent channel
    @commands.command()
    @commands.admin_or_permissions(administrator=True)
    async def setthreadid(self, ctx, channel: discord.TextChannel):
        """Sets the parent channel dynamically via command. (Admin only)"""
        try:
            await self.config.parent_channel_id.set(channel.id)
            await ctx.send(f"✅ Parent channel has been set to: {channel.mention}")
        except Exception as e:
            logging.error(f"Error setting parent channel: {e}")
            await ctx.send("❌ Failed to set parent channel.")

    # Command to display the currently tracked thread channel
    @commands.command()
    async def getthreadid(self, ctx):
        """Displays the currently tracked parent channel."""
        channel_id = await self.config.parent_channel_id()
        if channel_id is not None:
            channel = ctx.guild.get_channel(channel_id)
            if channel:
                await ctx.send(f"📌 Currently tracked parent channel: {channel.mention}")
            else:
                await ctx.send(f"⚠️ The configured channel (ID: {channel_id}) could not be found.")
        else:
            await ctx.send("⚠️ No parent channel has been set.")

    # Command to set the troubleshooting message (Admin only)
    @commands.command()
    @commands.admin_or_permissions(administrator=True)
    async def setmessage(self, ctx, *, message: str):
        """Sets the troubleshooting message that will be sent in new threads. (Admin only)"""
        try:
            if not message.strip():
                await ctx.send("⚠️ The message cannot be empty!")
                return
            await self.config.troubleshooting_message.set(message)
            await ctx.send("✅ Troubleshooting message has been updated.")
        except Exception as e:
            logging.error(f"Error setting troubleshooting message: {e}")
            await ctx.send("❌ Failed to update troubleshooting message.")

    # Command to get the current troubleshooting message
    @commands.command()
    async def getmessage(self, ctx):
        """Displays the currently set troubleshooting message."""
        message = await self.config.troubleshooting_message()
        if message:
            await ctx.send(f"📢 Current troubleshooting message: {message}")
        else:
            await ctx.send("⚠️ No troubleshooting message has been set.")

    @commands.Cog.listener()
    async def on_thread_create(self, thread: discord.Thread):
        """Listener for when a new thread is created in a forum channel."""
        parent_channel_id = await self.config.parent_channel_id()
        if parent_channel_id is None:
            logging.error("No parent channel ID set. Please set it using the command.")
            return

        # Check if the thread belongs to the configured parent channel
        if thread.parent_id == parent_channel_id:
            logging.info(f"New thread created: {thread.name} (ID: {thread.id})")
            await asyncio.sleep(3)  # Slightly longer delay to account for potential network lag

            # Send the troubleshooting message
            message = await self.config.troubleshooting_message()
            if not message:
                message = "⚠️ No troubleshooting message set. Use !setmessage to configure it."
            try:
                await thread.send(message)
                logging.info(f"Message sent successfully in thread: {thread.name}")
            except discord.Forbidden:
                logging.error(f"Bot lacks permissions to send messages in thread: {thread.name}")
            except discord.HTTPException as e:
                logging.error(f"Failed to send message in thread {thread.name}: {e}")
