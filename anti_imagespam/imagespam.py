import discord
from redbot.core import commands, Config
from redbot.core.utils.chat_formatting import pagify
import asyncio
import datetime

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp")

class ImageSpam(commands.Cog):
    """Anti-image-spam system with per-guild settings, including thread support."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9876543210123, force_registration=True)

        default_guild = {
            "max_images": 3,
            "excluded_channels": [],
            "log_channel_id": None,
            "monitor_all": True,
            "monitor_admins": False
        }
        self.config.register_guild(**default_guild)
        self.offenses = {}  # {guild_id: {user_id: [timestamps]}}

    # -------------------------------
    # Commands
    # -------------------------------
    @commands.group(name="imageprevent", invoke_without_command=True)
    @commands.admin()
    async def imageprevent(self, ctx):
        """Manage image spam prevention settings."""
        await ctx.send_help("imageprevent")

    @imageprevent.command(name="help")
    @commands.admin()
    async def imageprevent_help(self, ctx):
        """Show Image Prevention commands (Admin only)."""
        embed = discord.Embed(
            title="Image Prevention Commands",
            description="Manage the image spam prevention system.",
            color=discord.Color.blurple()
        )
        embed.add_field(name="!imageprevent image <amount>", value="Set the maximum allowed images per message.", inline=False)
        embed.add_field(name="!imageprevent channel <#channel>", value="Set the channel where violations and logs are posted.", inline=False)
        embed.add_field(name="!imageprevent exclude <#channel> [#channel ...]", value="Exclude one or multiple channels from monitoring.", inline=False)
        embed.add_field(name="!imageprevent include <#channel> [#channel ...]", value="Include one or multiple previously excluded channels.", inline=False)
        embed.add_field(name="!imageprevent monitorall <True/False>", value="Toggle whether all non-excluded channels are monitored.", inline=False)
        embed.add_field(name="!imageprevent monitoradmins <True/False>", value="Toggle whether messages from admins are also monitored.", inline=False)
        embed.add_field(name="!imageprevent list", value="List the current configuration for the server.", inline=False)
        await ctx.send(embed=embed)

    @imageprevent.command(name="image")
    @commands.admin()
    async def set_max_images(self, ctx, amount: int):
        if amount < 1:
            await ctx.send("Must allow at least 1 image.")
            return
        await self.config.guild(ctx.guild).max_images.set(amount)
        await ctx.send(f"Max images per message set to {amount}.")

    @imageprevent.command(name="channel")
    @commands.admin()
    async def set_log_channel(self, ctx, channel: discord.TextChannel):
        await self.config.guild(ctx.guild).log_channel_id.set(channel.id)
        await ctx.send(f"Logging channel set to {channel.mention}.")

    @imageprevent.command(name="exclude")
    @commands.admin()
    async def add_excluded_channels(self, ctx, *channels: discord.abc.GuildChannel):
        """Exclude one or multiple channels from monitoring."""
        excluded = set(await self.config.guild(ctx.guild).excluded_channels())
        for channel in channels:
            excluded.add(channel.id)
        await self.config.guild(ctx.guild).excluded_channels.set(list(excluded))
        await ctx.send(f"Excluded channels: {', '.join(c.name for c in channels)}")

    @imageprevent.command(name="include")
    @commands.admin()
    async def remove_excluded_channels(self, ctx, *channels: discord.abc.GuildChannel):
        """Include one or multiple previously excluded channels."""
        excluded = set(await self.config.guild(ctx.guild).excluded_channels())
        for channel in channels:
            excluded.discard(channel.id)
        await self.config.guild(ctx.guild).excluded_channels.set(list(excluded))
        await ctx.send(f"Included channels: {', '.join(c.name for c in channels)}")

    @imageprevent.command(name="monitorall")
    @commands.admin()
    async def toggle_monitor_all(self, ctx, value: bool):
        await self.config.guild(ctx.guild).monitor_all.set(value)
        status = "enabled" if value else "disabled"
        await ctx.send(f"Monitoring all non-excluded channels is now {status}.")

    @imageprevent.command(name="monitoradmins")
    @commands.admin()
    async def toggle_monitor_admins(self, ctx, value: bool):
        await self.config.guild(ctx.guild).monitor_admins.set(value)
        status = "enabled" if value else "disabled"
        await ctx.send(f"Monitoring admin messages is now {status}.")

    @imageprevent.command(name="list")
    @commands.admin()
    async def list_settings(self, ctx):
        guild_conf = await self.config.guild(ctx.guild).all()
        max_images = guild_conf["max_images"]
        excluded = guild_conf["excluded_channels"]
        log_id = guild_conf["log_channel_id"]
        log_channel = ctx.guild.get_channel(log_id) if log_id else None
        monitor_all = guild_conf["monitor_all"]
        monitor_admins = guild_conf.get("monitor_admins", False)

        excluded_names = [ctx.guild.get_channel(c).mention if ctx.guild.get_channel(c) else str(c) for c in excluded]

        msg = (
            f"**Image Prevention Settings:**\n"
            f"- Max images per message: {max_images}\n"
            f"- Log channel: {log_channel.mention if log_channel else 'Not set'}\n"
            f"- Monitor all non-excluded channels: {monitor_all}\n"
            f"- Monitor admin messages: {monitor_admins}\n"
            f"- Excluded channels: {', '.join(excluded_names) if excluded_names else 'None'}"
        )
        for page in pagify(msg):
            await ctx.send(page)

    # -------------------------------
    # Listener
    # -------------------------------
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return

        guild_conf = await self.config.guild(message.guild).all()
        max_images = guild_conf["max_images"]
        excluded_channels = set(guild_conf["excluded_channels"])
        monitor_all = guild_conf["monitor_all"]
        monitor_admins = guild_conf.get("monitor_admins", False)
        log_channel_id = guild_conf["log_channel_id"]

        # Determine if message should be monitored
        is_excluded = message.channel.id in excluded_channels
        should_monitor = (monitor_all and not is_excluded) or (monitor_admins and message.author.guild_permissions.administrator)

        # Skip if not monitored
        if not should_monitor:
            return

        # Count images
        img_count = sum(
            1 for a in message.attachments
            if (a.content_type and a.content_type.startswith("image/")) or a.filename.lower().endswith(IMAGE_EXTENSIONS)
        )

        log_channel = message.guild.get_channel(log_channel_id) if log_channel_id else None
        if log_channel is None:
            await message.channel.send(
                f"⚠️ Admins: Logging channel is not set. Please configure using `!imageprevent channel #channel`.",
                delete_after=10
            )

        # CASE 1: Excluded channel → log only if admin monitoring is enabled
        if is_excluded and monitor_admins and message.author.guild_permissions.administrator:
            if log_channel:
                embed = self.make_embed(
                    title="Excluded Channel Post Logged",
                    description=f"Message posted in **#{message.channel.name}** (excluded).\nImages: {img_count} ⚠️ User is an admin",
                    user=message.author,
                    color=discord.Color.orange()
                )
                embed.add_field(name="Content", value=message.content or "(no text)", inline=False)
                await log_channel.send(embed=embed)
            return

        # CASE 2: Exceeded image limit
        if img_count > max_images:
            if message.author.guild_permissions.administrator and not monitor_admins:
                return
            try:
                await message.delete()
            except discord.Forbidden:
                pass

            # Log violation
            if log_channel:
                embed = self.make_embed(
                    title="Image Spam Blocked",
                    description=f"User attempted to send **{img_count} images** (limit {max_images}) in **#{message.channel.name}**" +
                                (" ⚠️ User is an admin" if message.author.guild_permissions.administrator else ""),
                    user=message.author,
                    color=discord.Color.red()
                )
                embed.add_field(name="Deleted Content", value=message.content or "(no text)", inline=False)
                await log_channel.send(embed=embed)

            # Track offenses
            now = datetime.datetime.utcnow()
            guild_offenses = self.offenses.setdefault(message.guild.id, {})
            user_offenses = guild_offenses.setdefault(message.author.id, [])
            user_offenses = [t for t in user_offenses if (now - t).total_seconds() <= 60]
            user_offenses.append(now)
            guild_offenses[message.author.id] = user_offenses

            if len(user_offenses) >= 3:
                try:
                    until = now + datetime.timedelta(minutes=5)
                    await message.author.timeout(until, reason="Triggered image spam prevention")
                    try:
                        await message.author.send(
                            f"⚠️ You have been timed out for 5 minutes for exceeding the image limit in {message.guild.name}."
                        )
                    except discord.Forbidden:
                        pass
                except discord.Forbidden:
                    if log_channel:
                        await log_channel.send(f"⚠️ Could not timeout {message.author} (missing permissions).")

    # -------------------------------
    # Embed helper
    # -------------------------------
    def make_embed(self, title: str, description: str, user: discord.Member, color: int):
        embed = discord.Embed(title=title, description=description, color=color)
        embed.set_author(name=f"{user} (ID: {user.id})", icon_url=user.display_avatar.url)
        embed.set_footer(text="Image Monitoring System")
        return embed


async def setup(bot):
    await bot.add_cog(ImageSpam(bot))
