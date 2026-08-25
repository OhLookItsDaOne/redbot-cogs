import logging
import discord
import asyncio
from redbot.core import commands, Config, app_commands

logging.basicConfig(level=logging.INFO)


class SupportForum(commands.Cog):
    """Automatically posts a troubleshooting message in new forum posts."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1234567890)
        default_global = {
            "parent_channel_id": None,
            "troubleshooting_message": "Default troubleshooting message.",
            "privacy_policy_url": "https://github.com/OhLookItsDaOne/redbot-cogs/blob/main/PRIVACY.md",
        }
        self.config.register_global(**default_global)

    @commands.hybrid_group(name="forumhelp", invoke_without_command=True, extras={"red_force_enable": True})
    @commands.guild_only()
    @app_commands.default_permissions(administrator=True)
    async def forumhelp(self, ctx):
        """Manage support forum troubleshooting settings."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help(ctx.command)

    @forumhelp.command()
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def setchannel(self, ctx, channel_id: int):
        """Sets the parent channel ID for tracked forum posts."""
        await self.config.parent_channel_id.set(channel_id)
        # Try to fetch the channel from cache first
        channel = ctx.guild.get_channel(channel_id)
        if not channel:
            try:
                channel = await ctx.guild.fetch_channel(channel_id)
            except Exception:
                channel = None
        if channel:
            await ctx.send(f"Parent channel ID has been set to: {channel.mention}")
        else:
            await ctx.send(f"Parent channel ID has been set to: {channel_id} (channel not found)")

    @forumhelp.command()
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def getchannel(self, ctx):
        """Displays the currently tracked parent channel ID."""
        channel_id = await self.config.parent_channel_id()
        if channel_id is None:
            await ctx.send("No parent channel ID has been set.")
            return

        channel = ctx.guild.get_channel(channel_id)
        if not channel:
            try:
                channel = await ctx.guild.fetch_channel(channel_id)
            except Exception:
                channel = None
        if channel:
            await ctx.send(f"Currently tracked parent channel: {channel.mention}")
        else:
            await ctx.send("The stored parent channel ID is invalid or no longer accessible.")

    @forumhelp.command()
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def setmessage(self, ctx, *, message: str):
        """Sets the troubleshooting message."""
        if not message.strip():
            await ctx.send("⚠️ The message cannot be empty!")
            return
        await self.config.troubleshooting_message.set(message)
        await ctx.send("Troubleshooting message has been updated.")

    @forumhelp.command()
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def getmessage(self, ctx):
        """Displays the currently set troubleshooting message."""
        message = await self.config.troubleshooting_message()
        await ctx.send(f"Current troubleshooting message: {message}")

    @commands.hybrid_command(extras={"red_force_enable": True})
    async def privacy(self, ctx):
        """Shows the bot's privacy policy."""
        url = await self.config.privacy_policy_url()
        embed = discord.Embed(
            title="Privacy Policy",
            description=(
                "This bot processes message content in memory only for moderation "
                "and help features. It does not store message content, transmit it "
                "off-platform, or use it to train AI models.\n\n"
                f"Full policy: {url}"
            ),
            color=discord.Color.blue(),
        )
        await ctx.send(embed=embed)

    @commands.hybrid_command(name="setprivacypolicy", extras={"red_force_enable": True})
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def setprivacypolicy(self, ctx, url: str):
        """Sets the privacy policy URL shown by /privacy (Admin only)."""
        if not url.startswith("http://") and not url.startswith("https://"):
            await ctx.send("❌ The URL must start with http:// or https://")
            return
        await self.config.privacy_policy_url.set(url)
        await ctx.send(f"✅ Privacy policy URL set to: {url}")

    @commands.Cog.listener()
    async def on_thread_create(self, thread: discord.Thread):
        """Listener for when a new thread is created in a forum channel."""
        channel_id = await self.config.parent_channel_id()
        if channel_id is None:
            logging.error("No parent channel ID set. Please set it using the command.")
            return

        # Check if the thread belongs to the configured parent channel
        if thread.parent_id == channel_id:
            logging.info(f"New thread created: {thread.name} (ID: {thread.id})")
            await asyncio.sleep(3)  # Wait a bit for initialization
            message = await self.config.troubleshooting_message()
            if not message:
                message = "No troubleshooting message set. Use !forumhelp setmessage to configure it."
            try:
                await thread.send(message)
                logging.info(f"Message sent successfully in thread: {thread.name}")
            except discord.Forbidden:
                logging.error(f"Bot lacks permissions to send messages in thread: {thread.name}")
            except discord.HTTPException as e:
                logging.error(f"Failed to send message in thread {thread.name}: {e}")


async def setup(bot):
    await bot.add_cog(SupportForum(bot))
