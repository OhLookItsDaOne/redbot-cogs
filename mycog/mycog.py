from redbot.core import commands, checks
from redbot.core.utils import chat_formatting as cf
import discord

class ForumPostNotifier(commands.Cog):
    """A cog to send troubleshooting steps in response to new forum posts."""

    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_forum_post_created(self, post):
        """Listener for when a new forum post is created."""
        channel_id = post.channel_id  # Get the channel ID of the forum post
        message = self.create_troubleshooting_message()  # Create the troubleshooting message
        
        # Find the channel where you want to send the message
        channel = self.bot.get_channel(channel_id)
        if channel:
            await channel.send(message)

    def create_troubleshooting_message(self):
        """Creates the troubleshooting message."""
        message = (
            Hello! 👋
            Provide info to help us help you
            ```
            Please answer all these, unless we say otherwise-
            1  GPU
            2  CPU
            3  RAM
            4  Which VR headset do you use
            5  Where is SkyrimVR installed? (provide us with full path and a screenshot of the content inside)
            6  Where is FUS installed? (provide us with full path and a screenshot)
            7  Which FUS profile are you using?
            8  Did you move SkyrimVR or FUS after installing FUS?
            9  Have you downloaded and added any mods?
            10 Which version of the modlist are you on? (This can be found highlighted near the top of the list in MO2)
            ```
            

        )
        return message

# Add the cog to your bot
def setup(bot):
    bot.add_cog(ForumPostNotifier(bot))
