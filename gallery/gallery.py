import logging
import re

import discord
from redbot.core import commands, Config, app_commands

log = logging.getLogger("red.gallery")

YOUTUBE_RE = re.compile(
    r"(?:https?://)?(?:www\.|m\.|music\.)?"
    r"(?:youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|embed/|live/)|youtu\.be/)"
    r"[A-Za-z0-9_-]{11}"
)


class Gallery(commands.Cog):
    """Image gallery channels.

    In a configured gallery channel, only messages containing images, videos
    or YouTube links are allowed. Each allowed message automatically creates a
    thread where users can comment and post more media. Plain text messages
    are deleted and the author is notified with a temporary hint (30 seconds).
    Admins and configured admin roles are NOT exempt by default.
    """

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=8675309)
        default_guild = {
            "gallery_channels": [],          # list of channel IDs
            "thread_name": "Screenshot from {user}",
            "hint_duration": 30,             # seconds before hint is deleted
            "exempt_admins": False,          # whether admins are exempt from rules
            "admin_roles": [],               # role IDs treated as admins for this cog
            "allow_youtube": True,           # whether YouTube links are allowed
            "thread_message_enabled": True,  # whether to post a message in the thread
            "thread_message": "Feel free to comment on this image or post more images here.",
        }
        self.config.register_guild(**default_guild)

    @staticmethod
    def _is_admin(member: discord.Member, admin_roles: list) -> bool:
        """Whether the member counts as an admin for this cog."""
        if member.guild_permissions.administrator:
            return True
        return any(role.id in admin_roles for role in member.roles)

    @staticmethod
    def _has_image(message: discord.Message) -> bool:
        return any(
            a.content_type and a.content_type.startswith("image")
            for a in message.attachments
        )

    @staticmethod
    def _has_video(message: discord.Message) -> bool:
        return any(
            a.content_type and a.content_type.startswith("video")
            for a in message.attachments
        )

    @staticmethod
    def _has_youtube(message: discord.Message) -> bool:
        return bool(YOUTUBE_RE.search(message.content or ""))

    @staticmethod
    def _format(text: str, author: discord.Member) -> str:
        """Replace placeholders with the author's info."""
        return text.replace("{user}", author.display_name).replace(
            "{user_mention}", author.mention
        )

    # ─── Commands ─────────────────────────────────────────────────────────
    @commands.hybrid_group(name="gallery", invoke_without_command=True, extras={"red_force_enable": True})
    @commands.guild_only()
    @app_commands.default_permissions(administrator=True)
    async def gallery(self, ctx):
        """Manage image gallery channels."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help(ctx.command)

    @gallery.command(name="add")
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def setgallery(self, ctx, channel: discord.TextChannel):
        """Adds a channel as an image gallery (Admin only)."""
        async with self.config.guild(ctx.guild).gallery_channels() as channels:
            if channel.id not in channels:
                channels.append(channel.id)
                await ctx.send(f"✅ {channel.mention} is now a gallery channel.")
            else:
                await ctx.send(f"❌ {channel.mention} is already a gallery channel.")

    @gallery.command(name="remove")
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def removegallery(self, ctx, channel: discord.TextChannel):
        """Removes a channel from being an image gallery (Admin only)."""
        async with self.config.guild(ctx.guild).gallery_channels() as channels:
            if channel.id in channels:
                channels.remove(channel.id)
                await ctx.send(f"✅ {channel.mention} is no longer a gallery channel.")
            else:
                await ctx.send(f"❌ {channel.mention} is not a gallery channel.")

    @gallery.command(name="list")
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def listgallery(self, ctx):
        """Lists all gallery channels (Admin only)."""
        channels = await self.config.guild(ctx.guild).gallery_channels()
        if not channels:
            await ctx.send("No gallery channels configured.")
            return
        names = []
        for cid in channels:
            ch = ctx.guild.get_channel(cid)
            names.append(ch.mention if ch else f"`{cid}` (deleted)")
        await ctx.send("Gallery channels:\n" + "\n".join(names))

    @gallery.command(name="threadname")
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def setthreadname(self, ctx, *, name: str):
        """Sets the thread name template. Use `{user}` for the poster's name (Admin only)."""
        await self.config.guild(ctx.guild).thread_name.set(name)
        await ctx.send(f"✅ Thread name template set to: `{name}`")

    @gallery.command(name="hintduration")
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def sethintduration(self, ctx, seconds: int):
        """Sets how long the text-warning hint stays visible in seconds (Admin only)."""
        if seconds < 1 or seconds > 300:
            await ctx.send("❌ Duration must be between 1 and 300 seconds.")
            return
        await self.config.guild(ctx.guild).hint_duration.set(seconds)
        await ctx.send(f"✅ Hint duration set to **{seconds} seconds**.")

    @gallery.command(name="admins")
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def toggleadmins(self, ctx, state: str):
        """Toggles whether admins are exempt from the gallery rules (Admin only)."""
        if state.lower() in ["on", "true", "yes", "enable"]:
            await self.config.guild(ctx.guild).exempt_admins.set(True)
            await ctx.send("✅ Admins are now **exempt** from the gallery rules.")
        elif state.lower() in ["off", "false", "no", "disable"]:
            await self.config.guild(ctx.guild).exempt_admins.set(False)
            await ctx.send("✅ Admins are now **NOT exempt** from the gallery rules.")
        else:
            await ctx.send("❌ Please use `on` or `off`.")

    @gallery.group(name="adminrole", invoke_without_command=True)
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def gallery_adminrole(self, ctx):
        """Manage roles that count as admins for this cog (Admin only)."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help(ctx.command)

    @gallery_adminrole.command(name="add")
    async def adminrole_add(self, ctx, role: discord.Role):
        """Adds a role that counts as admin for this cog (Admin only)."""
        async with self.config.guild(ctx.guild).admin_roles() as roles:
            if role.id in roles:
                await ctx.send(f"❌ {role.mention} is already an admin role.")
            else:
                roles.append(role.id)
                await ctx.send(f"✅ {role.mention} now counts as an admin role.")

    @gallery_adminrole.command(name="remove")
    async def adminrole_remove(self, ctx, role: discord.Role):
        """Removes a role that counts as admin for this cog (Admin only)."""
        async with self.config.guild(ctx.guild).admin_roles() as roles:
            if role.id in roles:
                roles.remove(role.id)
                await ctx.send(f"✅ {role.mention} no longer counts as an admin role.")
            else:
                await ctx.send(f"❌ {role.mention} is not an admin role.")

    @gallery_adminrole.command(name="list")
    async def adminrole_list(self, ctx):
        """Lists all roles that count as admins for this cog (Admin only)."""
        roles = await self.config.guild(ctx.guild).admin_roles()
        if not roles:
            await ctx.send("No admin roles configured.")
            return
        names = []
        for rid in roles:
            role = ctx.guild.get_role(rid)
            names.append(role.mention if role else f"`{rid}` (deleted)")
        await ctx.send("Admin roles:\n" + "\n".join(names))

    @gallery.command(name="youtube")
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def toggleyoutube(self, ctx, state: str):
        """Toggles whether YouTube links are allowed in gallery channels (Admin only)."""
        if state.lower() in ["on", "true", "yes", "enable"]:
            await self.config.guild(ctx.guild).allow_youtube.set(True)
            await ctx.send("✅ YouTube links are now **allowed** in gallery channels.")
        elif state.lower() in ["off", "false", "no", "disable"]:
            await self.config.guild(ctx.guild).allow_youtube.set(False)
            await ctx.send("✅ YouTube links are now **blocked** in gallery channels.")
        else:
            await ctx.send("❌ Please use `on` or `off`.")

    @gallery.command(name="togglemessage")
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def togglethreadmessage(self, ctx, state: str):
        """Toggles whether a message is posted in newly created threads (Admin only)."""
        if state.lower() in ["on", "true", "yes", "enable"]:
            await self.config.guild(ctx.guild).thread_message_enabled.set(True)
            await ctx.send("✅ Thread messages are now **ON**.")
        elif state.lower() in ["off", "false", "no", "disable"]:
            await self.config.guild(ctx.guild).thread_message_enabled.set(False)
            await ctx.send("✅ Thread messages are now **OFF**.")
        else:
            await ctx.send("❌ Please use `on` or `off`.")

    @gallery.command(name="message")
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def setthreadmessage(self, ctx, *, text: str):
        """Sets the message posted in new threads (Admin only). Use {user} or {user_mention}."""
        await self.config.guild(ctx.guild).thread_message.set(text)
        await ctx.send(f"✅ Thread message set to:\n`{text}`")

    @gallery.command(name="status")
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def gallerystatus(self, ctx):
        """Shows the gallery configuration (Admin only)."""
        channels = await self.config.guild(ctx.guild).gallery_channels()
        thread_name = await self.config.guild(ctx.guild).thread_name()
        hint = await self.config.guild(ctx.guild).hint_duration()
        exempt = await self.config.guild(ctx.guild).exempt_admins()
        admin_roles = await self.config.guild(ctx.guild).admin_roles()
        youtube = await self.config.guild(ctx.guild).allow_youtube()
        msg_enabled = await self.config.guild(ctx.guild).thread_message_enabled()
        msg = await self.config.guild(ctx.guild).thread_message()
        names = []
        for cid in channels:
            ch = ctx.guild.get_channel(cid)
            names.append(ch.mention if ch else f"`{cid}` (deleted)")
        role_names = []
        for rid in admin_roles:
            role = ctx.guild.get_role(rid)
            role_names.append(role.mention if role else f"`{rid}` (deleted)")
        await ctx.send(
            "**Gallery channels:**\n" + ("\n".join(names) if names else "None") +
            f"\n**Thread name:** `{thread_name}`\n"
            f"**Hint duration:** {hint}s\n"
            f"**Admins exempt:** {'ON' if exempt else 'OFF'}\n"
            f"**Admin roles:** " + ("\n".join(role_names) if role_names else "None") +
            f"\n**YouTube links:** {'ALLOWED' if youtube else 'BLOCKED'}\n"
            f"**Thread message:** {'ON' if msg_enabled else 'OFF'}"
            + (f"\n**Message:** `{msg}`" if msg_enabled else "")
        )

    # ─── Listener ─────────────────────────────────────────────────────────
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return

        if not isinstance(message.channel, discord.TextChannel):
            return

        gallery_channels = await self.config.guild(message.guild).gallery_channels()
        if message.channel.id not in gallery_channels:
            return

        # Admin exemption
        exempt = await self.config.guild(message.guild).exempt_admins()
        if exempt:
            admin_roles = await self.config.guild(message.guild).admin_roles()
            if self._is_admin(message.author, admin_roles):
                return

        has_media = self._has_image(message) or self._has_video(message)
        has_youtube = False
        if not has_media:
            allow_youtube = await self.config.guild(message.guild).allow_youtube()
            has_youtube = allow_youtube and self._has_youtube(message)

        if has_media or has_youtube:
            # Create a thread for this media message
            thread_name = await self.config.guild(message.guild).thread_name()
            try:
                name = self._format(thread_name, message.author)
            except Exception:
                name = f"Screenshot from {message.author.display_name}"
            try:
                thread = await message.channel.create_thread(
                    name=name[:100],
                    message=message,
                    auto_archive_duration=10080,
                )
                # Optional message in the thread
                if await self.config.guild(message.guild).thread_message_enabled():
                    msg_text = await self.config.guild(message.guild).thread_message()
                    await thread.send(self._format(msg_text, message.author))
            except discord.Forbidden:
                log.warning("Missing permission to create thread in %s", message.channel)
            except Exception as e:
                log.error("Error creating thread in %s: %s", message.channel, e)
            return

        # Plain text message: delete it and notify the author
        hint_duration = await self.config.guild(message.guild).hint_duration()
        try:
            await message.delete()
        except discord.Forbidden:
            log.warning("Missing Manage Messages permission in %s", message.channel)
            return
        except discord.NotFound:
            return
        except Exception as e:
            log.error("Error deleting message in %s: %s", message.channel, e)
            return

        try:
            hint = await message.channel.send(
                f"{message.author.mention}, only messages with images, videos or "
                f"YouTube links can be sent in this channel. If you want to comment "
                f"on media, do so in its thread."
            )
            await hint.delete(delay=hint_duration)
        except Exception as e:
            log.error("Error sending/cleaning hint in %s: %s", message.channel, e)


async def setup(bot):
    await bot.add_cog(Gallery(bot))
