import discord
from redbot.core import commands, Config
import asyncio
import datetime
import logging

logging.basicConfig(level=logging.INFO)

class ChannelGuard(commands.Cog):
    """A cog to guard a specified channel against spammers.

    - First offense: Timeout for a configurable duration, messages deleted.
    - Second offense: Messages deleted, then user is kicked.
    """
    
    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=112233445566)
        default_global = {
            "guard_channel_id": None,
            "kick_channel_id": None,
            "punishment_duration": 10  # Default 10 minutes
        }
        self.config.register_global(**default_global)
        self.offenses = {}

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setguardchannel(self, ctx, channel: discord.TextChannel):
        """Sets the channel to be guarded (Admin only)."""
        await self.config.guard_channel_id.set(channel.id)
        await ctx.send(f"Guard channel set to: {channel.mention}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setkickchannel(self, ctx, channel: discord.TextChannel):
        """Sets the channel for logging kicks/timeouts (Admin only)."""
        await self.config.kick_channel_id.set(channel.id)
        await ctx.send(f"Kick log channel set to: {channel.mention}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setpunishmenttime(self, ctx, minutes: int):
        """Sets the duration for timeouts and message deletion (Admin only)."""
        if minutes <= 0:
            await ctx.send("Time must be greater than 0 minutes.")
            return
        await self.config.punishment_duration.set(minutes)
        await ctx.send(f"Punishment time set to {minutes} minutes.")
    
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def resetoffenses(self, ctx):
        """Resets all offense counts (Admin only)."""
        self.offenses = {}
        await ctx.send("All offense counts have been reset.")

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return

        guard_channel_id = await self.config.guard_channel_id()
        if guard_channel_id is None or message.channel.id != guard_channel_id:
            return

        punishment_minutes = await self.config.punishment_duration()
        time_delta = datetime.timedelta(minutes=punishment_minutes)
        time_threshold = discord.utils.utcnow() - time_delta
        member = message.author
        user_id = member.id
        offense_count = self.offenses.get(user_id, 0)
        
        if offense_count == 0:
            self.offenses[user_id] = 1
            until = discord.utils.utcnow() + time_delta
            try:
                await member.timeout(until, reason="First offense: Timeout applied.")
            except Exception as e:
                logging.error(f"Error timing out {member}: {e}")
            
            kick_channel_id = await self.config.kick_channel_id()
            if kick_channel_id:
                kick_channel = self.bot.get_channel(kick_channel_id)
                if kick_channel:
                    await kick_channel.send(f"{member.mention} has been timed out for {punishment_minutes} minutes (first offense).")
            
            await asyncio.sleep(punishment_minutes * 60)
            
        for channel in message.guild.text_channels:
            try:
                async for msg in channel.history(after=time_threshold, limit=None):
                    if msg.author.id == user_id:
                        await msg.delete()
            except Exception as e:
                logging.error(f"Error deleting messages in #{channel.name}: {e}")
        
        if offense_count == 1:
            kick_channel_id = await self.config.kick_channel_id()
            if kick_channel_id:
                kick_channel = self.bot.get_channel(kick_channel_id)
                if kick_channel:
                    await kick_channel.send(f"{member.mention} committed a second offense and was kicked.")
            
            try:
                await member.kick(reason="Second offense: Kicked.")
            except Exception as e:
                logging.error(f"Error kicking user {member}: {e}")
            
            self.offenses.pop(user_id, None)


def setup(bot):
    bot.add_cog(ChannelGuard(bot))
