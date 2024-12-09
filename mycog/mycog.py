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

    # Command to set the parent channel ID
    @commands.command()
    async def setthreadid(self, ctx, channel_id: int):
        """Sets the parent channel ID dynamically via command."""
        self.parent_channel_id = channel_id
        await ctx.send(f"Parent channel ID has been set to: {self.parent_channel_id}")

    # Command to display the currently tracked thread ID
    @commands.command()
    async def getthreadid(self, ctx):
        """Displays the currently tracked parent channel ID."""
        if self.parent_channel_id is not None:
            await ctx.send(f"Currently tracked parent channel ID: {self.parent_channel_id}")
        else:
            await ctx.send("No parent channel ID has been set.")

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
            await asyncio.sleep(3)  # Slightly longer delay to account for potential network lag

            # Send the troubleshooting message
            try:
                await thread.send(self.create_troubleshooting_message())
                logging.info(f"Message sent successfully in thread: {thread.name}")
            except discord.Forbidden:
                logging.error(f"Bot lacks permissions to send messages in thread: {thread.name}")
            except discord.HTTPException as e:
                logging.error(f"Failed to send message in thread {thread.name}: {e}")

    def create_troubleshooting_message(self):
        """Creates the troubleshooting message."""
        message = (
            "Hello! 👋\n\n"
            "Provide info to help us help you!\n\n"
            "Please answer all these, unless we say otherwise:\n"
            "1. **GPU**\n"
            "2. **CPU**\n"
            "3. **RAM**\n"
            "4. **Which VR headset do you use?**\n"
            "5. **Where is SkyrimVR installed?** (provide us with full path and a screenshot of the content inside)\n"
            "6. **Where is FUS installed?** (provide us with full path and a screenshot)\n"
            "7. **Which FUS profile are you using?**\n"
            "8. **Did you move SkyrimVR or FUS after installing FUS?**\n"
            "9. **Have you downloaded and added any mods?**\n"
            "10. **Which version of the modlist are you on?** (This can be found highlighted near the top of the list in MO2)\n\n"
            "If your game is crashing please provide us with your crashlog. It can be found at:\n"
            "```Documents\\My Games\\Skyrim VR\\SKSE\\sksevr.log```\n"
            "Look for the most recent file starting with crash- + the date and time of the crash.\n\n"
            "Lastly, if you use a pirated version of the game, the list will not work. We do not support piracy, buy the game. It is regularly on sale anyways."
        )
        return message
