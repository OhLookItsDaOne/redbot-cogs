import logging
import discord  # Import discord module
from redbot.core import commands

# Configure logging
logging.basicConfig(level=logging.INFO)

class ForumPostNotifier(commands.Cog):
    """A cog to send troubleshooting steps in response to new forum posts."""

    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_thread_create(self, thread):
        """Listener for when a new thread is created in the forum."""
        # Check if the thread is part of the specific forum
        if thread.parent_id == 1172448935772704788:  # Forum ID
            logging.info(f"New thread created: {thread.name} (ID: {thread.id})")
            await thread.send(self.create_troubleshooting_message())

    @commands.Cog.listener()
    async def on_message(self, message):
        """Listener for when a new message is posted in the thread."""
        # This is commented out to reduce log output
        # if isinstance(message.channel, discord.Thread) and message.channel.parent_id == 1172448935772704788:
        #     logging.info(f"Message detected in thread: {message.content} from {message.author}")

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
"""
        )
        return message
