import discord
from redbot.core import commands, Config
from redbot.core.utils.chat_formatting import pagify
import datetime

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp")

class ImageSpam(commands.Cog):
    """Anti-image-spam system with per-guild settings."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9876543210123, force_registration=True)

        default_guild = {
            "max_images": 3,
            "excluded_channels": [],
            "log_channel_id": None,
            "monitor_all": True,
            "monitored_channels": [],
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
        await ctx.send_help("imageprevent")

    @imageprevent.command(name="help")
    @commands.admin()
    async def imageprevent_help(self, ctx):
        embed = discord.Embed(
            title="Image Prevention Commands",
            description="Manage and configure the image spam prevention system.",
            color=discord.Color.blurple()
        )
        embed.add_field(name="!imageprevent image <amount>", value="Set the maximum allowed images per message (default: 3).", inline=False)
        embed.add_field(name="!imageprevent channel <#channel>", value="Set the channel where violations and logs are posted.", inline=False)
        embed.add_field(name="!imageprevent exclude <#channel>", value="Exclude a channel from monitoring.", inline=False)
        embed.add_field(name="!imageprevent include <#channel>", value="Remove a channel from the exclusion list.", inline=False)
        embed.add_field(name="!imageprevent monitorall <True/False>", value="Toggle whether all non-excluded channels are monitored.", inline=False)
        embed.add_field(name="!imageprevent list", value="List the current configuration for the server.", inline=False)
        embed.add_field(name="!imageprevent help", value="Show this help embed with available commands (Admin only).", inline=False)
        embed.add_field(name="!imageprevent monitorallchannels <add/remove>", value="Add or remove all text channels in the server to the monitored channels list.", inline=False)
        embed.add_field(name="!imageprevent monitoradmins <True/False>", value="Toggle whether messages from admins are also monitored and deleted if they exceed the image limit.", inline=False)
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
    async def add_excluded_channel(self, ctx, channel: discord.TextChannel):
        excluded = await self.config.guild(ctx.guild).excluded_channels()
        if channel.id in excluded:
            await ctx.send(f"{channel.mention} is already excluded.")
            return
        excluded.append(channel.id)
        await self.config.guild(ctx.guild).excluded_channels.set(excluded)
        await ctx.send(f"{channel.mention} is now excluded.")

    @imageprevent.command(name="include")
    @commands.admin()
    async def remove_excluded_channel(self, ctx, channel: discord.TextChannel):
        excluded = await self.config.guild(ctx.guild).excluded_channels()
        if channel.id not in excluded:
            await ctx.send(f"{channel.mention} is not excluded.")
            return
        excluded.remove(channel.id)
        await self.config.guild(ctx.guild).excluded_channels.set(excluded)
        await ctx.send(f"{channel.mention} is no longer excluded.")

    @imageprevent.command(name="monitorallchannels")
    @commands.admin()
    async def monitor_all_channels(self, ctx, action: str):
        guild_conf = await self.config.guild(ctx.guild).all()
        monitored_channels = set(guild_conf["monitored_channels"])
        if action.lower() == "add":
            for ch in ctx.guild.text_channels:
                monitored_channels.add(ch.id)
            await self.config.guild(ctx.guild).monitored_channels.set(list(monitored_channels))
            await ctx.send(f"✅ All text channels have been added to the monitored channels list.")
        elif action.lower() == "remove":
            await self.config.guild(ctx.guild).monitored_channels.set([])
            await ctx.send(f"✅ All text channels have been removed from the monitored channels list.")
        else:
            await ctx.send("❌ Invalid action. Use `add` or `remove`.")

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
        excluded = set(guild_conf["excluded_channels"])
        log_id = guild_conf["log_channel_id"]
        log_channel = ctx.guild.get_channel(log_id) if log_id else None
        monitor_all = guild_conf["monitor_all"]
        monitor_admins = guild_conf["monitor_admins"]
        monitored = set(guild_conf["monitored_channels"]) - excluded  # ignore excluded channels in list

        excluded_names = [ctx.guild.get_channel(c).mention if ctx.guild.get_channel(c) else str(c) for c in excluded]
        monitored_names = [ctx.guild.get_channel(c).mention if ctx.guild.get_channel(c) else str(c) for c in monitored]

        msg = (
            f"**Image Prevention Settings:**\n"
            f"- Max images per message: {max_images}\n"
            f"- Log channel: {log_channel.mention if log_channel else 'Not set'}\n"
            f"- Monitor all non-excluded channels: {monitor_all}\n"
            f"- Monitor admin messages: {monitor_admins}\n"
            f"- Excluded channels: {', '.join(excluded_names) if excluded_names else 'None'}\n"
            f"- Explicitly monitored channels: {', '.join(monitored_names) if monitored_names else 'None'}"
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
        monitored_channels = set(guild_conf["monitored_channels"])
        monitor_all = guild_conf["monitor_all"]
        monitor_admins = guild_conf.get("monitor_admins", False)
        log_channel_id = guild_conf["log_channel_id"]

        channel_id = message.channel.id
        is_excluded = channel_id in excluded_channels
        is_monitored = monitor_all or (channel_id in monitored_channels)

        # Count images
        img_count = sum(1 for a in message.attachments if (a.content_type and a.content_type.startswith("image/")) or a.filename.lower().endswith(IMAGE_EXTENSIONS))

        # Excluded channels
        if is_excluded:
            if img_count == 0 and not (monitor_admins and message.author.guild_permissions.administrator):
                return

            log_channel = message.guild.get_channel(log_channel_id) if log_channel_id else None
            if log_channel:
                embed = self.make_embed(
                    title="Excluded Channel Post Logged",
                    description=f"Message posted in **#{message.channel.name}** (excluded).\nImages: {img_count}" +
                                (" ⚠️ User is an admin" if message.author.guild_permissions.administrator else ""),
                    user=message.author,
                    color=discord.Color.orange()
                )
                if img_count > 0:
                    embed.add_field(name="Images", value=f"{img_count} images attached", inline=False)
                if message.content:
                    embed.add_field(name="Content", value=message.content, inline=False)
                await log_channel.send(embed=embed)
            return

        if not is_monitored and not (monitor_admins and message.author.guild_permissions.administrator):
            return

        # Check for image spam
        if img_count > max_images:
            if message.author.guild_permissions.administrator and not monitor_admins:
                return

            try:
                await message.delete()
            except discord.Forbidden:
                pass

            log_channel = message.guild.get_channel(log_channel_id) if log_channel_id else None
            if log_channel:
                embed = self.make_embed(
                    title="Image Spam Blocked",
                    description=f"User attempted to send **{img_count} images** (limit {max_images}) in **#{message.channel.name}**" +
                                (" ⚠️ User is an admin" if message.author.guild_permissions.administrator else ""),
                    user=message.author,
                    color=discord.Color.red()
                )
                if message.content:
                    embed.add_field(name="Deleted Content", value=message.content, inline=False)
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
                        await message.author.send(f"⚠️ You have been timed out for 5 minutes for exceeding the image limit in {message.guild.name}.")
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
