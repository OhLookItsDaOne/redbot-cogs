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

        self.offenses = {}

    def get_effective_channel_id(self, message: discord.Message) -> int:
        if isinstance(message.channel, discord.Thread) and message.channel.parent_id:
            return message.channel.parent_id
        return message.channel.id

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
        embed.add_field(name="!imageprevent monitorallchannels <add/remove>", value="Add or remove all text channels in the server to the monitored channels list.", inline=False)
        embed.add_field(name="!imageprevent monitoradmins <True/False>", value="Toggle whether messages from admins are also monitored.", inline=False)
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
        if channel.id not in excluded:
            excluded.append(channel.id)
            await self.config.guild(ctx.guild).excluded_channels.set(excluded)
        await ctx.send(f"{channel.mention} is now excluded (threads included).")

    @imageprevent.command(name="include")
    @commands.admin()
    async def remove_excluded_channel(self, ctx, channel: discord.TextChannel):
        excluded = await self.config.guild(ctx.guild).excluded_channels()
        if channel.id in excluded:
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
            await ctx.send("✅ All text channels added.")
        elif action.lower() == "remove":
            await self.config.guild(ctx.guild).monitored_channels.set([])
            await ctx.send("✅ Monitored channels cleared.")
        else:
            await ctx.send("❌ Invalid action. Use add/remove.")

    @imageprevent.command(name="monitorall")
    @commands.admin()
    async def toggle_monitor_all(self, ctx, value: bool):
        await self.config.guild(ctx.guild).monitor_all.set(value)
        await ctx.send(f"Monitoring all non-excluded channels is now {value}.")

    @imageprevent.command(name="monitoradmins")
    @commands.admin()
    async def toggle_monitor_admins(self, ctx, value: bool):
        await self.config.guild(ctx.guild).monitor_admins.set(value)
        await ctx.send(f"Monitoring admin messages is now {value}.")

    @imageprevent.command(name="list")
    @commands.admin()
    async def list_settings(self, ctx):
        guild_conf = await self.config.guild(ctx.guild).all()
        log = ctx.guild.get_channel(guild_conf["log_channel_id"]) if guild_conf["log_channel_id"] else None
        excluded = [ctx.guild.get_channel(c).mention for c in guild_conf["excluded_channels"] if ctx.guild.get_channel(c)]

        msg = (
            f"Max images: {guild_conf['max_images']}\n"
            f"Monitor all: {guild_conf['monitor_all']}\n"
            f"Monitor admins: {guild_conf['monitor_admins']}\n"
            f"Log channel: {log.mention if log else 'Disabled'}\n"
            f"Excluded: {', '.join(excluded) if excluded else 'None'}"
        )

        for page in pagify(msg):
            await ctx.send(page)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return

        conf = await self.config.guild(message.guild).all()

        effective_channel_id = self.get_effective_channel_id(message)

        # Excluded = do nothing
        if effective_channel_id in conf["excluded_channels"]:
            return

        # Admin ignore
        if message.author.guild_permissions.administrator and not conf["monitor_admins"]:
            return

        if not conf["monitor_all"] and effective_channel_id not in conf["monitored_channels"]:
            return

        img_count = sum(
            1 for a in message.attachments
            if (a.content_type and a.content_type.startswith("image/"))
            or a.filename.lower().endswith(IMAGE_EXTENSIONS)
        )

        if img_count <= conf["max_images"]:
            return

        try:
            await message.delete()
        except discord.Forbidden:
            pass

        if conf["log_channel_id"]:
            log_channel = message.guild.get_channel(conf["log_channel_id"])
            if log_channel:
                embed = self.make_embed(
                    "Image Spam Blocked",
                    f"#{message.channel.name}\nImages: {img_count}/{conf['max_images']}",
                    message.author,
                    discord.Color.red(),
                )
                await log_channel.send(embed=embed)

        now = datetime.datetime.utcnow()
        user_offenses = self.offenses.setdefault(message.guild.id, {}).setdefault(message.author.id, [])
        user_offenses[:] = [t for t in user_offenses if (now - t).total_seconds() <= 60]
        user_offenses.append(now)

        if len(user_offenses) >= 3:
            try:
                await message.author.timeout(now + datetime.timedelta(minutes=5))
            except discord.Forbidden:
                pass

    def make_embed(self, title: str, description: str, user: discord.Member, color: int):
        embed = discord.Embed(title=title, description=description, color=color)
        embed.set_author(name=f"{user} (ID: {user.id})", icon_url=user.display_avatar.url)
        embed.set_footer(text="Image Monitoring System")
        return embed


async def setup(bot):
    await bot.add_cog(ImageSpam(bot))
