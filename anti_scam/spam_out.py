import asyncio
import datetime
import logging
import discord
from redbot.core import commands, Config, app_commands

logging.basicConfig(level=logging.INFO)


class ChannelGuard(commands.Cog):
    """Guard a channel against spammers.

    - First offense: Timeout for a configurable duration.
    - Second offense: User is banned (with Discord's native message deletion)
      and automatically unbanned after a configurable duration.

    Recent messages of the user can be purged for the last N minutes using
    bulk deletion (rate-limit friendly).
    """

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=112233445566)
        default_global = {
            "guard_channel_id": None,
            "kick_channel_id": None,       # log channel
            "punishment_duration": 10,     # first offense timeout in minutes
            "delete_message_days": 0,      # Discord ban delete_days (0-7)
            "recent_minutes": 5,           # purge recent messages within this window
            "ban_duration_hours": 24,      # auto-unban after this many hours
            "scheduled_unbans": {},        # {guild_id: {user_id: unban_timestamp}}
        }
        self.config.register_global(**default_global)
        self.offenses = {}
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
        log_channel_id = await self.config.kick_channel_id()
        if not log_channel_id:
            return
        channel = guild.get_channel(log_channel_id)
        if channel:
            try:
                await channel.send(text)
            except Exception:
                logging.exception("Failed to send log message")

    # ─── Commands ─────────────────────────────────────────────────────────
    @commands.hybrid_command(extras={"red_force_enable": True})
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def setguardchannel(self, ctx, channel: discord.TextChannel):
        """Sets the channel to be guarded (Admin only)."""
        await self.config.guard_channel_id.set(channel.id)
        await ctx.send(f"Guard channel set to: {channel.mention}")

    @commands.hybrid_command(name="setlogchannel", aliases=["setkickchannel"], extras={"red_force_enable": True})
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def setlogchannel(self, ctx, channel: discord.TextChannel):
        """Sets the channel for logging timeouts/bans (Admin only)."""
        await self.config.kick_channel_id.set(channel.id)
        await ctx.send(f"Log channel set to: {channel.mention}")

    @commands.hybrid_command(extras={"red_force_enable": True})
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def setpunishmenttime(self, ctx, minutes: int):
        """Sets the first-offense timeout duration in minutes (Admin only)."""
        if minutes <= 0:
            await ctx.send("Time must be greater than 0 minutes.")
            return
        await self.config.punishment_duration.set(minutes)
        await ctx.send(f"First-offense timeout set to {minutes} minutes.")

    @commands.hybrid_command(extras={"red_force_enable": True})
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def setdeletedays(self, ctx, days: int):
        """Sets how many days of the user's messages Discord deletes on ban (0-7)."""
        if days < 0 or days > 7:
            await ctx.send("❌ Choose a value between 0 and 7 days:\n"
                           "`0` - delete nothing\n"
                           "`1` - last 1 day\n"
                           "`2` - last 2 days\n"
                           "`3` - last 3 days\n"
                           "`4` - last 4 days\n"
                           "`5` - last 5 days\n"
                           "`6` - last 6 days\n"
                           "`7` - last 7 days")
            return
        await self.config.delete_message_days.set(days)
        await ctx.send(f"Ban message deletion set to **{days} day(s)**.")

    @commands.hybrid_command(extras={"red_force_enable": True})
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def setrecentminutes(self, ctx, minutes: int):
        """Sets how many recent minutes of the user's messages get purged (Admin only)."""
        if minutes < 0:
            await ctx.send("❌ Minutes must be 0 or greater (0 = no purge).")
            return
        await self.config.recent_minutes.set(minutes)
        await ctx.send(f"Recent message purge window set to **{minutes} minute(s)**.")

    @commands.hybrid_command(extras={"red_force_enable": True})
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def setbanduration(self, ctx, hours: int):
        """Sets how long a ban lasts before auto-unban (Admin only)."""
        if hours <= 0:
            await ctx.send("❌ Duration must be greater than 0 hours.")
            return
        await self.config.ban_duration_hours.set(hours)
        await ctx.send(f"Ban duration set to **{hours} hour(s)** (auto-unban after that).")

    @commands.hybrid_command(extras={"red_force_enable": True})
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def resetoffenses(self, ctx):
        """Resets all offense counts (Admin only)."""
        self.offenses = {}
        await ctx.send("All offense counts have been reset.")

    @commands.hybrid_command(extras={"red_force_enable": True})
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def guardstatus(self, ctx):
        """Shows the current guard configuration (Admin only)."""
        guard_id = await self.config.guard_channel_id()
        log_id = await self.config.kick_channel_id()
        punish = await self.config.punishment_duration()
        days = await self.config.delete_message_days()
        recent = await self.config.recent_minutes()
        ban_h = await self.config.ban_duration_hours()

        guard = ctx.guild.get_channel(guard_id) if guard_id else None
        logch = ctx.guild.get_channel(log_id) if log_id else None

        text = (
            f"**Guard channel:** {guard.mention if guard else '❌ Not set'}\n"
            f"**Log channel:** {logch.mention if logch else '❌ Not set'}\n"
            f"**First-offense timeout:** {punish} min\n"
            f"**Ban message deletion:** {days} day(s)\n"
            f"**Recent purge window:** {recent} min\n"
            f"**Ban duration:** {ban_h} hour(s) (auto-unban)"
        )
        await ctx.send(text)

    # ─── Listener ─────────────────────────────────────────────────────────
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return

        guard_channel_id = await self.config.guard_channel_id()
        if guard_channel_id is None or message.channel.id != guard_channel_id:
            return

        member = message.author
        user_id = member.id

        # Keep offense counts fresh within a reasonable window (e.g. 24h)
        now = discord.utils.utcnow()
        offense_window = datetime.timedelta(hours=24)
        prev = self.offenses.get(user_id, [])
        prev = [t for t in prev if (now - t) <= offense_window]
        offense_count = len(prev)

        if offense_count == 0:
            # First offense: timeout only, delete nothing (no nuke)
            self.offenses[user_id] = prev + [now]
            punishment_minutes = await self.config.punishment_duration()
            until = now + datetime.timedelta(minutes=punishment_minutes)
            try:
                await member.timeout(until, reason="First offense: Timeout applied.")
                await self._send_log(
                    message.guild,
                    f"⏰ {member.mention} timed out for {punishment_minutes} min (first offense).",
                )
            except discord.Forbidden:
                logging.warning("Missing permission to timeout %s", member)
            except Exception as e:
                logging.error("Error timing out %s: %s", member, e)
            return

        # Second offense: ban + purge + auto-unban
        try:
            delete_days = await self.config.delete_message_days()
            ban_hours = await self.config.ban_duration_hours()
            await member.ban(
                reason=f"Second offense: banned for {ban_hours} hours.",
                delete_message_days=delete_days,
            )
            logging.info("Banned %s in guild %s (delete %s days)", member, message.guild.id, delete_days)
        except discord.Forbidden:
            logging.warning("Missing permission to ban %s", member)
            await self._send_log(message.guild, f"❌ Missing Ban Members permission for {member.mention}.")
            return
        except Exception as e:
            logging.error("Error banning %s: %s", member, e)
            await self._send_log(message.guild, f"❌ Failed to ban {member.mention}: {e}")
            return

        await self._send_log(
            message.guild,
            f"🚫 {member.mention} banned (2nd offense). Auto-unban in {ban_hours} hour(s). "
            f"Deleted {delete_days} day(s) via Discord, purging recent messages...",
        )

        # Purge recent messages using bulk deletion (rate-limit friendly)
        recent_minutes = await self.config.recent_minutes()
        if recent_minutes > 0:
            threshold = now - datetime.timedelta(minutes=recent_minutes)
            for channel in message.guild.text_channels:
                try:
                    # bulk=True deletes in batches of 100 -> 1 API call per 100 messages.
                    # limit bounds how many messages we scan per channel.
                    await channel.purge(
                        after=threshold,
                        limit=500,
                        check=lambda m, uid=user_id: m.author.id == uid,
                        bulk=True,
                    )
                except discord.Forbidden:
                    pass
                except discord.HTTPException as e:
                    logging.warning("Purge rate-limited/error in #%s: %s", channel.name, e)
                except Exception as e:
                    logging.error("Error purging #%s: %s", channel.name, e)

        # Schedule auto-unban
        await self._schedule_unban(message.guild.id, user_id, ban_hours)
        self.offenses.pop(user_id, None)

        await self._send_log(message.guild, f"✅ Purge complete. Ban for {member.mention} scheduled for auto-unban.")


def setup(bot):
    bot.add_cog(ChannelGuard(bot))
