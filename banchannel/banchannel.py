import asyncio
import datetime
import logging
import discord
from redbot.core import commands, Config, app_commands


class BanChannel(commands.Cog):
    """Ban anyone who posts in the configured channel.

    Any message posted in the ban channel instantly results in:
    - the user being banned (auto-unbanned after a configurable duration)
    - Discord deleting the user's messages from the last configurable period
      (``delete_message_seconds``) server-side, in a single API call.
    """

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=112233445566)
        default_global = {
            "ban_channel_id": None,
            "log_channel_id": None,           # log channel
            "delete_message_seconds": 3600,   # how far back Discord deletes on ban (1 hour)
            "ban_duration_hours": 24,         # auto-unban after this many hours
            "scheduled_unbans": {},           # {guild_id: {user_id: unban_timestamp}}
        }
        self.config.register_global(**default_global)
        self._unban_task = None

    async def cog_load(self):
        self._unban_task = asyncio.create_task(self._unban_loop())

    async def cog_unload(self):
        if self._unban_task:
            self._unban_task.cancel()

    # ─── Unban background task ────────────────────────────────────────────
    async def _unban_loop(self):
        await self.bot.wait_until_ready()
        while True:
            try:
                await self._process_scheduled_unbans()
            except Exception:
                logging.exception("Error processing scheduled unbans")
            await asyncio.sleep(60)

    async def _process_scheduled_unbans(self):
        scheduled = await self.config.scheduled_unbans()
        now = discord.utils.utcnow().timestamp()
        changed = False
        for gid_str, users in list(scheduled.items()):
            guild = self.bot.get_guild(int(gid_str))
            if guild is None:
                continue
            for uid_str, ts in list(users.items()):
                if now >= ts:
                    user_id = int(uid_str)
                    try:
                        await guild.unban(discord.Object(id=user_id), reason="Automatic unban after configured duration.")
                        await self._send_log(guild, f"🔓 **{user_id}** has been automatically unbanned.")
                        logging.info("Automatically unbanned user %s in guild %s", user_id, guild.id)
                    except discord.NotFound:
                        pass
                    except Exception:
                        logging.exception("Failed to unban user %s in guild %s", user_id, guild.id)
                    del users[uid_str]
                    changed = True
            if not users:
                del scheduled[gid_str]
                changed = True
        if changed:
            await self.config.scheduled_unbans.set(scheduled)

    async def _schedule_unban(self, guild_id: int, user_id: int, hours: int):
        scheduled = await self.config.scheduled_unbans()
        scheduled.setdefault(str(guild_id), {})[str(user_id)] = (
            discord.utils.utcnow() + datetime.timedelta(hours=hours)
        ).timestamp()
        await self.config.scheduled_unbans.set(scheduled)

    async def _send_log(self, guild, text: str):
        log_channel_id = await self.config.log_channel_id()
        if not log_channel_id:
            return
        channel = guild.get_channel(log_channel_id)
        if channel:
            try:
                await channel.send(text)
            except Exception:
                logging.exception("Failed to send log message")

    # ─── Commands ─────────────────────────────────────────────────────────
    @commands.hybrid_group(name="banchannel", invoke_without_command=True, extras={"red_force_enable": True})
    @commands.guild_only()
    @app_commands.default_permissions(administrator=True)
    async def banchannel(self, ctx):
        """Manage the instant-ban channel."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help(ctx.command)

    @banchannel.command()
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def setchannel(self, ctx, channel: discord.TextChannel):
        """Sets the channel in which posting results in an instant ban (Admin only)."""
        await self.config.ban_channel_id.set(channel.id)
        await ctx.send(f"Ban channel set to: {channel.mention}")

    @banchannel.command()
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def setlog(self, ctx, channel: discord.TextChannel):
        """Sets the channel for logging bans (Admin only)."""
        await self.config.log_channel_id.set(channel.id)
        await ctx.send(f"Log channel set to: {channel.mention}")

    @banchannel.command()
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def deleteseconds(self, ctx, seconds: int):
        """Sets how far back Discord deletes messages on ban, in seconds (Admin only).

        Examples: 3600 = 1 hour, 21600 = 6 hours, 86400 = 24 hours, up to 604800 (7 days).
        """
        if seconds < 0 or seconds > 604800:
            await ctx.send("❌ Choose a value between 0 and 604800 seconds (7 days).\n"
                           "`0` - delete nothing\n"
                           "`3600` - previous hour\n"
                           "`21600` - previous 6 hours\n"
                           "`86400` - previous 24 hours\n"
                           "`259200` - previous 3 days\n"
                           "`604800` - previous 7 days")
            return
        await self.config.delete_message_seconds.set(seconds)
        hours = seconds / 3600
        label = f"**{seconds} seconds** ({hours:.1f} hours)" if seconds > 0 else "**0** (nothing)"
        await ctx.send(f"Ban message deletion set to {label}.")

    @banchannel.command()
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def banduration(self, ctx, hours: int):
        """Sets how long a ban lasts before auto-unban (Admin only)."""
        if hours <= 0:
            await ctx.send("❌ Duration must be greater than 0 hours.")
            return
        await self.config.ban_duration_hours.set(hours)
        await ctx.send(f"Ban duration set to **{hours} hour(s)** (auto-unban after that).")

    @banchannel.command()
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def status(self, ctx):
        """Shows the current ban channel configuration (Admin only)."""
        ban_id = await self.config.ban_channel_id()
        log_id = await self.config.log_channel_id()
        del_seconds = await self.config.delete_message_seconds()
        ban_h = await self.config.ban_duration_hours()

        banch = ctx.guild.get_channel(ban_id) if ban_id else None
        logch = ctx.guild.get_channel(log_id) if log_id else None

        text = (
            f"**Ban channel:** {banch.mention if banch else '❌ Not set'}\n"
            f"**Log channel:** {logch.mention if logch else '❌ Not set'}\n"
            f"**Ban message deletion:** {del_seconds} seconds ({del_seconds/3600:.1f} hours)\n"
            f"**Ban duration:** {ban_h} hour(s) (auto-unban)"
        )
        await ctx.send(text)

    # ─── Listener ─────────────────────────────────────────────────────────
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return

        ban_channel_id = await self.config.ban_channel_id()
        if ban_channel_id is None or message.channel.id != ban_channel_id:
            return

        member = message.author
        user_id = member.id

        # Instant ban for any message in the ban channel
        try:
            del_seconds = await self.config.delete_message_seconds()
            ban_hours = await self.config.ban_duration_hours()
            await member.ban(
                reason=f"Posted in ban channel; banned for {ban_hours} hours.",
                delete_message_seconds=del_seconds,
            )
            logging.info(
                "Banned %s in guild %s (delete %s seconds)",
                member, message.guild.id, del_seconds,
            )
        except discord.Forbidden:
            logging.warning("Missing permission to ban %s", member)
            await self._send_log(
                message.guild, f"❌ Missing Ban Members permission for {member.mention}."
            )
            return
        except Exception as e:
            logging.error("Error banning %s: %s", member, e)
            await self._send_log(message.guild, f"❌ Failed to ban {member.mention}: {e}")
            return

        await self._send_log(
            message.guild,
            f"🚫 {member.mention} banned (posted in ban channel). "
            f"Auto-unban in {ban_hours} hour(s). "
            f"Discord deleting last {del_seconds} seconds of messages...",
        )

        # Schedule auto-unban
        await self._schedule_unban(message.guild.id, user_id, ban_hours)
        await self._send_log(
            message.guild, f"✅ Ban for {member.mention} scheduled for auto-unban."
        )


async def setup(bot):
    await bot.add_cog(BanChannel(bot))
