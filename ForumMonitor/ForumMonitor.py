import discord
from redbot.core import commands

class ForumMonitor(commands.Cog):
    """Monitor support forum posts and send troubleshooting guide on new post creation."""

    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_thread_create(self, thread: discord.Thread):
        # Replace this with your actual forum channel ID
        forum_channel_id = 1172448935772704788  # This is your forum channel ID

        # Check if the thread was created in the forum channel you want to monitor
        if thread.parent_id == forum_channel_id:
            troubleshoot_message = """
            Hey there! Thanks for creating a post in the support forum.
            
            Before we get started, please take a look at the troubleshooting guide below to help narrow down your issue:

            # Provide info to help us help you
            ```
            Please answer all these, unless we say otherwise-
            1. GPU
            2. CPU
            3. RAM
            4. Which VR headset do you use
            5. Where is SkyrimVR installed? (provide us with full path and a screenshot of the content inside)
            6. Where is FUS installed? (provide us with full path)
            7. Which FUS profile are you using?
            8. Did you move SkyrimVR or FUS after installing FUS?
            9. Have you downloaded and added any mods?
            10.Which version of the modlist are you on? (This can be found highlighted near the top of the list in MO2)
            ```
            If you've tried these steps and still need assistance, feel free to provide more details and we'll help you out!
            """

            # Send the message in the newly created thread
            await thread.send(troubleshoot_message)

# Setup function for Redbot to load the cog
def setup(bot):
    bot.add_cog(ForumMonitor(bot))