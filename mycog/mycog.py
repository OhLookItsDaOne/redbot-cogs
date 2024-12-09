import logging
import discord  # Import discord module
import asyncio  # Import asyncio for delay
from redbot.core import commands

# Configure logging
logging.basicConfig(level=logging.INFO)

class ForumPostNotifier(commands.Cog):
    """A cog to send troubleshooting steps in response to new forum posts."""

    def __init__(self, bot):
        self.bot = bot
        self.thread_id = None  # Store the thread ID for easy configuration

    # Command to set the thread ID
    @commands.command()
    async def set_thread_id(self, ctx, thread_id: int):
        """Sets the thread ID dynamically via command."""
        self.thread_id = thread_id
        await ctx.send(f"Thread ID has been set to: {self.thread_id}")

    @commands.Cog.listener()
    async def on_thread_create(self, thread):
        """Listener for when a new thread is created in the forum."""
        # Ensure a thread ID has been set
        if self.thread_id is None:
            logging.error("No thread ID set. Please set the thread ID using the command.")
            return
        
        # Check if the thread matches the configured ID
        if thread.parent_id == self.thread_id:
            logging.info(f"New thread created: {thread.name} (ID: {thread.id})")
            
            # Wait a bit to ensure thread and any attachments (like images) are fully loaded
            await asyncio.sleep(2)  # Increased to 2 seconds to handle images
            
            # Fetch the thread again to ensure it's fully initialized
            thread = await self.bot.get_channel(thread.id)
            
            # Check if the thread is valid
            if isinstance(thread, discord.Thread):
                logging.info(f"Thread {thread.name} is valid and ready for interaction.")
                await thread.send(self.create_troubleshooting_message())
            else:
                logging.error(f"Thread {thread.id} is not valid or accessible.")
                return

    def create_troubleshooting_message(self):
        """Creates the troubleshooting message."""
        message = (
            """Hello! 👋
            
Provide info to help us help you!
        
Please answer all these, unless we say otherwise:
1. **GPU**
2. **CPU**
3. **RAM**
4. **Which VR headset do you use?**
5. **Where is SkyrimVR installed?** (provide us with full path and a screenshot of the content inside)
6. **Where is FUS installed?** (provide us with full path and a screenshot)
7. **Which FUS profile are you using?**
8. **Did you move SkyrimVR or FUS after installing FUS?**
9. **Have you downloaded and added any mods?**
10. **Which version of the modlist are you on?** (This can be found highlighted near the top of the list in MO2)

If your game is crashing please provide us with your crashlog. It can be found at:
```Documents\My Games\Skyrim VR\SKSE\sksevr.log```
Look for the most recent file starting with crash- + the date and time of the crash.

Lastly, if you use a pirated version of the game, the list will not work. We do not support piracy, buy the game. It is regularly on sale anyways.
"""
        )
        return message
