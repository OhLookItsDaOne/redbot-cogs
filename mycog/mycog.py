from redbot.core import commands

class ForumPostNotifier(commands.Cog):
    """A cog to send troubleshooting steps in response to new forum posts."""

    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_message(self, message):
        """Listener for when a new message is posted in the forum channel."""
        # Check if the message is in the correct forum channel
        if message.channel.id == 1172448935772704788 and not message.author.bot:
            # Create and send the troubleshooting message
            troubleshooting_message = self.create_troubleshooting_message()
            await message.channel.send(troubleshooting_message)

    def create_troubleshooting_message(self):
        """Creates the troubleshooting message."""
        message = (
            """Hello! 👋
            
            Provide info to help us help you!
            ```
            Please answer all these, unless we say otherwise:
            1. GPU
            2. CPU
            3. RAM
            4. Which VR headset do you use?
            5. Where is SkyrimVR installed? (provide us with full path and a screenshot of the content inside)
            6. Where is FUS installed? (provide us with full path and a screenshot)
            7. Which FUS profile are you using?
            8. Did you move SkyrimVR or FUS after installing FUS?
            9. Have you downloaded and added any mods?
            10. Which version of the modlist are you on? (This can be found highlighted near the top of the list in MO2)
            ```"""
        )
        return message
