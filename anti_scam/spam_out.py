import discord
from redbot.core import commands, Config
import asyncio
import datetime
import logging

logging.basicConfig(level=logging.INFO)

class ChannelGuard(commands.Cog):
    """A cog to guard a specified channel against spammers.

    When a user writes in the guarded channel:
      - On the first offense, the user is timed out for 10 minutes and all of their messages
        from the last 10 minutes in all text channels are deleted.
      - On a second offense, the user is kicked.
    """

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=112233445566)
        default_global = {
            "guard_channel_id": None
        }
        self.config.register_global(**default_global)
        # Offenses are stored in memory: {user_id: offense_count}
        self.offenses = {}
    
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setguardchannel(self, ctx, channel: discord.TextChannel):
        """Sets the channel to be guarded (Admin only)."""
        await self.config.guard_channel_id.set(channel.id)
        await ctx.send(f"Guard channel has been set to: {channel.mention}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def resetoffenses(self, ctx):
        """Resets all offense counts (Admin only)."""
        self.offenses = {}
        await ctx.send("All offense counts have been reset.")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        # Ignore messages from bots or if the message is not in a guild
        if message.author.bot or not message.guild:
            return

        guard_channel_id = await self.config.guard_channel_id()
        if guard_channel_id is None:
            return
        # Only react if the message was sent in the guard channel.
        if message.channel.id != guard_channel_id:
            return

        member = message.author
        user_id = member.id

        # Determine offense count
        offense_count = self.offenses.get(user_id, 0)

        # For 10 minutes ago
        time_delta = datetime.timedelta(minutes=10)
        time_threshold = discord.utils.utcnow() - time_delta

        if offense_count == 0:
            # First offense: timeout for 10 minutes and delete messages in last 10 minutes in all text channels.
            self.offenses[user_id] = 1
            until = discord.utils.utcnow() + time_delta
            try:
                await member.edit(timeout=until, reason="Guard channel first offense timeout.")
            except Exception as e:
                logging.error(f"Error timing out {member}: {e}")
            # Iterate over all text channels in the guild.
            for channel in message.guild.text_channels:
                try:
                    async for msg in channel.history(after=time_threshold, limit=None):
                        if msg.author.id == user_id:
                            try:
                                await msg.delete()
                            except Exception as e:
                                logging.error(f"Error deleting message {msg.id} in #{channel.name} from {member}: {e}")
                except Exception as e:
                    logging.error(f"Error iterating messages in channel #{channel.name}: {e}")
            try:
                await message.channel.send(
                    f"{member.mention} has been timed out for 10 minutes and all your messages from the last 10 minutes have been deleted."
                )
            except Exception as e:
                logging.error(f"Error sending timeout message: {e}")

        else:
            # Second offense: kick the user.
            try:
                await message.channel.send(f"{member.mention} has committed a second offense and will be kicked.")
            except Exception as e:
                logging.error(f"Error sending kick message: {e}")
            try:
                await member.kick(reason="Guard channel second offense kick.")
            except Exception as e:
                logging.error(f"Error kicking user {member}: {e}")
            # Remove offense count so that if the user rejoins, the count resets.
            self.offenses.pop(user_id, None)

def setup(bot):
    bot.add_cog(ChannelGuard(bot))
