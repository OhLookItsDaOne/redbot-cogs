from discord import ForumChannel
from redbot.core import commands

class ForumMessage(commands.Cog):
    """A cog to send messages in a newly created forum post."""

    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_thread_create(self, thread):
        """Send a message when a new forum thread is created."""
        # Check if the thread is in the specific forum channel
        if isinstance(thread.parent, ForumChannel) and thread.parent.id == 1172448935772704788:
            message_content = (
                "Hey there! Thanks for creating a post in the support forum.\n\n"
                "Before we get started, please take a look at the troubleshooting guide below to help narrow down your issue:\n\n"
                "# Provide info to help us help you\n"
                "```\n"
                "Please answer all these, unless we say otherwise-\n"
                "1. GPU\n"
                "2. CPU\n"
                "3. RAM\n"
                "4. Which VR headset do you use\n"
                "5. Where is SkyrimVR installed? (provide us with full path and a screenshot of the content inside)\n"
                "6. Where is FUS installed? (provide us with full path)\n"
                "7. Which FUS profile are you using?\n"
                "8. Did you move SkyrimVR or FUS after installing FUS?\n"
                "9. Have you downloaded and added any mods?\n"
                "10. Which version of the modlist are you on? (This can be found highlighted near the top of the list in MO2)\n"
                "```\n"
                "If you've tried these steps and still need assistance, feel free to provide more details and we'll help you out!"
            )
            # Send the message to the newly created thread
            await thread.send(message_content)