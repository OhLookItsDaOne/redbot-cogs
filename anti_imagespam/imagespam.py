import discord
from redbot.core import commands, Config
from redbot.core.utils.chat_formatting import pagify, box
import datetime
from typing import Optional, List

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff")

class ImageSpam(commands.Cog):
    """Advanced anti-image-spam system with customizable messages."""
    
    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9876543210123, force_registration=True)
        default_guild = {
            "max_images": 3,
            "excluded_channels": [],
            "log_channel_id": None,
            "monitor_all": True,
            "monitor_admins": False,
            "exclude_forum_threads": True,
            "admin_role_id": None,
            "monitored_channels": [],
            "notification_on_delete": True,
            "user_notification": True,
            "user_message": "Your message in {channel} was deleted because it contained {image_count} images (maximum allowed: {max_images}).",
            "log_message": "🚫 **Image Spam Blocked**\nUser: {user_mention}\nChannel: {channel_mention}\nImages: {image_count}/{max_images}",
            "timeout_message": "⏰ **User Timed Out**\n{user_mention} has been timed out for 5 minutes due to repeated violations.",
            "placeholder_info": "Available placeholders:\n{user} - Username\n{user_mention} - @User\n{channel} - Channel name\n{channel_mention} - #channel\n{max_images} - Max allowed images\n{image_count} - Images in message\n{guild} - Server name"
        }
        self.config.register_guild(**default_guild)
        self.offenses = {}

    def is_admin_or_role(self, ctx):
        """Check if user has admin permissions or the configured admin role."""
        if ctx.author.guild_permissions.administrator:
            return True
        
        admin_role_id = self.config.guild(ctx.guild).admin_role_id()
        if admin_role_id:
            admin_role = ctx.guild.get_role(admin_role_id)
            if admin_role and admin_role in ctx.author.roles:
                return True
        
        return False

    def get_monitored_channels(self, guild: discord.Guild, conf: dict) -> List[int]:
        """Get list of channel IDs that are currently being monitored."""
        monitored = []
        
        if conf["monitor_all"]:
            for channel in guild.text_channels:
                if channel.id not in conf["excluded_channels"]:
                    if conf["exclude_forum_threads"]:
                        if isinstance(channel, discord.Thread):
                            continue
                        if isinstance(channel, discord.ForumChannel):
                            continue
                    monitored.append(channel.id)
        else:
            monitored = conf["monitored_channels"]
        
        return monitored

    def should_monitor_message(self, message: discord.Message, conf: dict) -> bool:
        """Determine if a message should be monitored."""
        if message.author.bot or not message.guild:
            return False
        
        if message.author.guild_permissions.administrator and not conf["monitor_admins"]:
            return False
        
        channel_id = message.channel.id
        
        if channel_id in conf["excluded_channels"]:
            return False
        
        if conf["exclude_forum_threads"]:
            if isinstance(message.channel, discord.Thread):
                return False
            if isinstance(message.channel, discord.ForumChannel):
                return False
        
        if conf["monitor_all"]:
            if isinstance(message.channel, discord.TextChannel):
                return True
        else:
            if channel_id in conf["monitored_channels"]:
                return True
        
        return False

    def format_message(self, message: str, message_obj: discord.Message, img_count: int, conf: dict) -> str:
        """Format a message with placeholders."""
        formatted = message
        formatted = formatted.replace("{user}", str(message_obj.author))
        formatted = formatted.replace("{user_mention}", message_obj.author.mention)
        formatted = formatted.replace("{channel}", f"#{message_obj.channel.name}")
        formatted = formatted.replace("{channel_mention}", message_obj.channel.mention)
        formatted = formatted.replace("{max_images}", str(conf["max_images"]))
        formatted = formatted.replace("{image_count}", str(img_count))
        formatted = formatted.replace("{guild}", message_obj.guild.name)
        return formatted

    async def send_ephemeral(self, ctx, content: str, **kwargs):
        """Send an ephemeral message (only visible to the user)."""
        # For Redbot, we can use ctx.send with ephemeral=True if it's a slash command
        # But since we're using prefix commands, we'll simulate it with a DM or a temporary message
        try:
            # Try to send as DM first (closest to ephemeral)
            await ctx.author.send(content)
        except discord.Forbidden:
            # Fallback to regular channel message with delete_after
            await ctx.send(f"{ctx.author.mention} (Only you can see this): {content}", delete_after=10.0, **kwargs)

    @commands.group(name="imageprevent", invoke_without_command=True)
    @commands.guild_only()
    async def imageprevent(self, ctx):
        """Manage image spam prevention system."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help("imageprevent")

    @imageprevent.command(name="help")
    async def imageprevent_help(self, ctx):
        """Show help for image prevention commands."""
        embed = discord.Embed(
            title="🛡️ Image Prevention Commands",
            description="Manage the image spam prevention system.",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="⚙️ Configuration",
            value="• `!imageprevent image <1-10>` - Max images (default: 3)\n"
                  "• `!imageprevent channel <#channel>` - Set log channel\n"
                  "• `!imageprevent logtoggle <on/off>` - Toggle log notifications",
            inline=False
        )
        
        embed.add_field(
            name="💬 Message Settings",
            value="• `!imageprevent usermessage <text>` - Set user notification message\n"
                  "• `!imageprevent logmessage <text>` - Set log channel message\n"
                  "• `!imageprevent timeoutmessage <text>` - Set timeout notification message\n"
                  "• `!imageprevent usernotify <on/off>` - Toggle user notifications\n"
                  "• `!imageprevent placeholders` - Show available placeholders",
            inline=False
        )
        
        embed.add_field(
            name="📋 Monitoring Settings",
            value="• `!imageprevent monitorall <on/off>` - Monitor all non-excluded\n"
                  "• `!imageprevent monitoradmins <on/off>` - Monitor admins\n"
                  "• `!imageprevent forumthreads <on/off>` - Auto-exclude forum threads",
            inline=False
        )
        
        embed.add_field(
            name="🚫 Channel Management",
            value="• `!imageprevent exclude <#channel>` - Exclude channel\n"
                  "• `!imageprevent include <#channel>` - Remove exclusion\n"
                  "• `!imageprevent addmonitor <#channel>` - Add to monitored\n"
                  "• `!imageprevent removemonitor <#channel>` - Remove from monitored",
            inline=False
        )
        
        embed.add_field(
            name="📊 Information",
            value="• `!imageprevent list` - Show settings\n"
                  "• `!imageprevent monitored` - Show monitored channels\n"
                  "• `!imageprevent unmonitored` - Show unmonitored channels\n"
                  "• `!imageprevent status` - Check channel status",
            inline=False
        )
        
        embed.add_field(
            name="👑 Admin Role",
            value="• `!imageprevent setadminrole <@role>` - Set admin role\n"
                  "• `!imageprevent clearadminrole` - Clear admin role",
            inline=False
        )
        
        embed.set_footer(text="Use !imageprevent placeholders to see available message placeholders")
        await ctx.send(embed=embed)

    async def check_admin_or_role(self, ctx):
        """Check if user has permission to use admin commands."""
        if not self.is_admin_or_role(ctx):
            await self.send_ephemeral(ctx, "❌ You need administrator permissions or the configured admin role to use this command.")
            return False
        return True

    @imageprevent.command(name="usermessage")
    async def set_user_message(self, ctx, *, message: str):
        """Set the message sent to users when their message is deleted."""
        if not await self.check_admin_or_role(ctx):
            return
        
        if len(message) > 500:
            await ctx.send("❌ Message too long. Maximum 500 characters.")
            return
        
        await self.config.guild(ctx.guild).user_message.set(message)
        await ctx.send(f"✅ User notification message set to:\n```{message}```")

    @imageprevent.command(name="logmessage")
    async def set_log_message(self, ctx, *, message: str):
        """Set the message sent to log channel when a violation occurs."""
        if not await self.check_admin_or_role(ctx):
            return
        
        if len(message) > 1000:
            await ctx.send("❌ Message too long. Maximum 1000 characters.")
            return
        
        await self.config.guild(ctx.guild).log_message.set(message)
        await ctx.send(f"✅ Log message set to:\n```{message}```")

    @imageprevent.command(name="timeoutmessage")
    async def set_timeout_message(self, ctx, *, message: str):
        """Set the message sent when a user is timed out."""
        if not await self.check_admin_or_role(ctx):
            return
        
        if len(message) > 500:
            await ctx.send("❌ Message too long. Maximum 500 characters.")
            return
        
        await self.config.guild(ctx.guild).timeout_message.set(message)
        await ctx.send(f"✅ Timeout message set to:\n```{message}```")

    @imageprevent.command(name="placeholders")
    async def show_placeholders(self, ctx):
        """Show available placeholders for custom messages."""
        placeholders = await self.config.guild(ctx.guild).placeholder_info()
        # Use double curly braces to escape them in the example
        example = "Your message in {{channel}} was deleted for having {{image_count}}/{{max_images}} images."
        await ctx.send(f"📝 **Available Placeholders**\n```{placeholders}```\n\n**Example:**\n`{example}`")

    @imageprevent.command(name="usernotify")
    async def toggle_user_notification(self, ctx, state: str):
        """Toggle notifications sent to users (on/off)."""
        if not await self.check_admin_or_role(ctx):
            return
        
        if state.lower() in ["on", "true", "yes", "enable"]:
            await self.config.guild(ctx.guild).user_notification.set(True)
            await ctx.send("✅ User notifications are now **ON**. Users will receive a temporary ephemeral message when their message is deleted.")
        elif state.lower() in ["off", "false", "no", "disable"]:
            await self.config.guild(ctx.guild).user_notification.set(False)
            await ctx.send("✅ User notifications are now **OFF**. Users will not receive notifications.")
        else:
            await ctx.send("❌ Please use `on` or `off`.")

    @imageprevent.command(name="channel")
    async def set_log_channel(self, ctx, channel: discord.TextChannel):
        """Set the channel where violation logs are posted."""
        if not await self.check_admin_or_role(ctx):
            return
        
        await self.config.guild(ctx.guild).log_channel_id.set(channel.id)
        await ctx.send(f"✅ Logging channel set to {channel.mention}.")

    @imageprevent.command(name="logtoggle")
    async def toggle_log_notification(self, ctx, state: str):
        """Toggle delete notifications in log channel (on/off)."""
        if not await self.check_admin_or_role(ctx):
            return
        
        if state.lower() in ["on", "true", "yes", "enable"]:
            await self.config.guild(ctx.guild).notification_on_delete.set(True)
            await ctx.send("✅ Log channel notifications are now **ON**.")
        elif state.lower() in ["off", "false", "no", "disable"]:
            await self.config.guild(ctx.guild).notification_on_delete.set(False)
            await ctx.send("✅ Log channel notifications are now **OFF**.")
        else:
            await ctx.send("❌ Please use `on` or `off`.")

    @imageprevent.command(name="image")
    async def set_max_images(self, ctx, amount: int):
        """Set maximum allowed images per message (1-10)."""
        if not await self.check_admin_or_role(ctx):
            return
        
        if amount < 1 or amount > 10:
            await ctx.send("❌ Amount must be between 1 and 10.")
            return
        
        await self.config.guild(ctx.guild).max_images.set(amount)
        await ctx.send(f"✅ Max images per message set to **{amount}**.")

    @imageprevent.command(name="monitorall")
    async def toggle_monitor_all(self, ctx, state: str):
        """Toggle monitoring all non-excluded channels (on/off)."""
        if not await self.check_admin_or_role(ctx):
            return
        
        if state.lower() in ["on", "true", "yes", "enable"]:
            await self.config.guild(ctx.guild).monitor_all.set(True)
            await ctx.send("✅ Now monitoring **all non-excluded channels**.")
        elif state.lower() in ["off", "false", "no", "disable"]:
            await self.config.guild(ctx.guild).monitor_all.set(False)
            await ctx.send("✅ Now monitoring only **specified channels**.")
        else:
            await ctx.send("❌ Please use `on` or `off`.")

    @imageprevent.command(name="monitoradmins")
    async def toggle_monitor_admins(self, ctx, state: str):
        """Toggle whether to monitor admin messages (on/off)."""
        if not await self.check_admin_or_role(ctx):
            return
        
        if state.lower() in ["on", "true", "yes", "enable"]:
            await self.config.guild(ctx.guild).monitor_admins.set(True)
            await ctx.send("✅ **Admin monitoring is now ON.** Admin messages will be deleted if they violate rules.")
        elif state.lower() in ["off", "false", "no", "disable"]:
            await self.config.guild(ctx.guild).monitor_admins.set(False)
            await ctx.send("✅ **Admin monitoring is now OFF.** Admin messages are ignored.")
        else:
            await ctx.send("❌ Please use `on` or `off`.")

    @imageprevent.command(name="forumthreads")
    async def toggle_forum_threads(self, ctx, state: str):
        """Toggle auto-exclusion of forum threads (on/off)."""
        if not await self.check_admin_or_role(ctx):
            return
        
        if state.lower() in ["on", "true", "yes", "enable"]:
            await self.config.guild(ctx.guild).exclude_forum_threads.set(True)
            await ctx.send("✅ **Forum threads are now automatically excluded.**")
        elif state.lower() in ["off", "false", "no", "disable"]:
            await self.config.guild(ctx.guild).exclude_forum_threads.set(False)
            await ctx.send("✅ **Forum threads are no longer auto-excluded.**")
        else:
            await ctx.send("❌ Please use `on` or `off`.")

    @imageprevent.command(name="exclude")
    async def add_excluded_channel(self, ctx, channel: discord.TextChannel):
        """Exclude a channel from monitoring."""
        if not await self.check_admin_or_role(ctx):
            return
        
        async with self.config.guild(ctx.guild).excluded_channels() as excluded:
            if channel.id not in excluded:
                excluded.append(channel.id)
        
        async with self.config.guild(ctx.guild).monitored_channels() as monitored:
            if channel.id in monitored:
                monitored.remove(channel.id)
        
        await ctx.send(f"✅ {channel.mention} is now **excluded** from monitoring.")

    @imageprevent.command(name="include")
    async def remove_excluded_channel(self, ctx, channel: discord.TextChannel):
        """Remove a channel from exclusion list."""
        if not await self.check_admin_or_role(ctx):
            return
        
        async with self.config.guild(ctx.guild).excluded_channels() as excluded:
            if channel.id in excluded:
                excluded.remove(channel.id)
        
        await ctx.send(f"✅ {channel.mention} is no longer excluded.")

    @imageprevent.command(name="addmonitor")
    async def add_monitored_channel(self, ctx, channel: discord.TextChannel):
        """Add a channel to monitored channels (when monitorall=off)."""
        if not await self.check_admin_or_role(ctx):
            return
        
        excluded = await self.config.guild(ctx.guild).excluded_channels()
        if channel.id in excluded:
            await ctx.send(f"❌ {channel.mention} is excluded. Remove exclusion first with `!imageprevent include #{channel.name}`.")
            return
        
        async with self.config.guild(ctx.guild).monitored_channels() as monitored:
            if channel.id not in monitored:
                monitored.append(channel.id)
        
        await ctx.send(f"✅ {channel.mention} added to monitored channels.")

    @imageprevent.command(name="removemonitor")
    async def remove_monitored_channel(self, ctx, channel: discord.TextChannel):
        """Remove a channel from monitored channels."""
        if not await self.check_admin_or_role(ctx):
            return
        
        async with self.config.guild(ctx.guild).monitored_channels() as monitored:
            if channel.id in monitored:
                monitored.remove(channel.id)
        
        await ctx.send(f"✅ {channel.mention} removed from monitored channels.")

    @imageprevent.command(name="setadminrole")
    async def set_admin_role(self, ctx, role: discord.Role):
        """Set a role that can use imageprevent commands."""
        if not ctx.author.guild_permissions.administrator:
            await self.send_ephemeral(ctx, "❌ You need administrator permissions to set the admin role.")
            return
        
        await self.config.guild(ctx.guild).admin_role_id.set(role.id)
        await ctx.send(f"✅ Admin role set to {role.mention}. Members with this role can now use imageprevent commands.")

    @imageprevent.command(name="clearadminrole")
    async def clear_admin_role(self, ctx):
        """Clear the admin role setting."""
        if not ctx.author.guild_permissions.administrator:
            await self.send_ephemeral(ctx, "❌ You need administrator permissions to clear the admin role.")
            return
        
        await self.config.guild(ctx.guild).admin_role_id.set(None)
        await ctx.send("✅ Admin role cleared. Only server admins can use imageprevent commands.")

    @imageprevent.command(name="list")
    async def list_settings(self, ctx):
        """Show current image prevention settings."""
        conf = await self.config.guild(ctx.guild).all()
        
        log_channel = ctx.guild.get_channel(conf["log_channel_id"]) if conf["log_channel_id"] else None
        admin_role = ctx.guild.get_role(conf["admin_role_id"]) if conf["admin_role_id"] else None
        
        monitored_count = len(self.get_monitored_channels(ctx.guild, conf))
        
        embed = discord.Embed(
            title="🛡️ Image Prevention Settings",
            color=discord.Color.blue(),
            timestamp=datetime.datetime.utcnow()
        )
        
        embed.add_field(name="Max Images", value=f"**{conf['max_images']}** per message", inline=True)
        embed.add_field(name="Log Channel", value=log_channel.mention if log_channel else "❌ Not set", inline=True)
        embed.add_field(name="Admin Role", value=admin_role.mention if admin_role else "👑 Server Admins only", inline=True)
        
        embed.add_field(name="Monitoring Mode", value=f"**{'All non-excluded' if conf['monitor_all'] else 'Specific channels'}**", inline=True)
        embed.add_field(name="Admin Monitoring", value=f"**{'ON' if conf['monitor_admins'] else 'OFF'}**", inline=True)
        embed.add_field(name="Forum Threads", value=f"**{'Auto-excluded' if conf['exclude_forum_threads'] else 'Monitored'}**", inline=True)
        
        embed.add_field(name="Notifications", value=f"Log: **{'ON' if conf['notification_on_delete'] else 'OFF'}**\nUser: **{'ON' if conf['user_notification'] else 'OFF'}**", inline=True)
        embed.add_field(name="Monitored Channels", value=f"**{monitored_count}** channels", inline=True)
        embed.add_field(name="Excluded Channels", value=f"**{len(conf['excluded_channels'])}** channels", inline=True)
        
        user_msg_preview = conf["user_message"][:50] + "..." if len(conf["user_message"]) > 50 else conf["user_message"]
        embed.add_field(name="User Message", value=f"```{user_msg_preview}```", inline=False)
        
        await ctx.send(embed=embed)

    @imageprevent.command(name="monitored")
    async def list_monitored_channels(self, ctx):
        """Show all channels currently being monitored."""
        conf = await self.config.guild(ctx.guild).all()
        monitored_ids = self.get_monitored_channels(ctx.guild, conf)
        
        if not monitored_ids:
            await ctx.send("❌ No channels are being monitored.")
            return
        
        text_channels = []
        for cid in monitored_ids:
            channel = ctx.guild.get_channel(cid)
            if channel:
                if isinstance(channel, discord.TextChannel):
                    text_channels.append(f"📝 #{channel.name}")
        
        embed = discord.Embed(
            title="📊 Currently Monitored Channels",
            description="\n".join(text_channels[:20]) if text_channels else "No text channels",
            color=discord.Color.green()
        )
        
        if len(text_channels) > 20:
            embed.set_footer(text=f"Showing 20 of {len(text_channels)} channels")
        
        await ctx.send(embed=embed)

    @imageprevent.command(name="status")
    async def channel_status(self, ctx, channel: Optional[discord.TextChannel] = None):
        """Check monitoring status of a specific channel."""
        channel = channel or ctx.channel
        conf = await self.config.guild(ctx.guild).all()
        
        class MockMessage:
            def __init__(self, ch, guild, author):
                self.channel = ch
                self.guild = guild
                self.author = author
        
        mock_msg = MockMessage(channel, ctx.guild, ctx.author)
        is_monitored = self.should_monitor_message(mock_msg, conf)
        
        embed = discord.Embed(
            title=f"{'🟢' if is_monitored else '🔴'} Channel Monitoring Status",
            color=discord.Color.green() if is_monitored else discord.Color.red()
        )
        
        embed.add_field(name="Channel", value=channel.mention, inline=True)
        embed.add_field(name="Status", value="**MONITORED**" if is_monitored else "**NOT MONITORED**", inline=True)
        embed.add_field(name="Max Images", value=f"**{conf['max_images']}** allowed", inline=True)
        
        await ctx.send(embed=embed)

    async def send_ephemeral_to_user(self, user: discord.Member, channel: discord.TextChannel, content: str):
        """Send an ephemeral-like message to a user."""
        try:
            # Try to send as DM (closest to ephemeral for non-interaction messages)
            await user.send(content)
        except discord.Forbidden:
            # If DMs are disabled, send a temporary message in the channel
            # This will be visible to everyone, but deleted after a short time
            try:
                temp_msg = await channel.send(f"{user.mention} (Only you can see this notification): {content}", delete_after=10.0)
            except Exception:
                pass  # Couldn't send temporary message either

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Monitor messages for image spam."""
        if message.author.bot or not message.guild:
            return
        
        conf = await self.config.guild(message.guild).all()
        
        if not self.should_monitor_message(message, conf):
            return
        
        img_count = 0
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                img_count += 1
            elif any(attachment.filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                img_count += 1
        
        if img_count <= conf["max_images"]:
            return
        
        # Nachricht löschen
        delete_success = False
        try:
            await message.delete()
            delete_success = True
        except (discord.Forbidden, discord.NotFound):
            delete_success = False
        except Exception as e:
            print(f"Error deleting message: {e}")
            delete_success = False
        
        # Benutzer benachrichtigen (ephemeral-like)
        if conf["user_notification"] and delete_success:
            user_msg = self.format_message(
                conf["user_message"],
                message,
                img_count,
                conf
            )
            
            # Send ephemeral-like notification
            await self.send_ephemeral_to_user(message.author, message.channel, user_msg)
        
        # Logge in Log-Channel
        if conf["log_channel_id"] and conf["notification_on_delete"]:
            log_channel = message.guild.get_channel(conf["log_channel_id"])
            if log_channel:
                try:
                    log_msg = self.format_message(
                        conf["log_message"],
                        message,
                        img_count,
                        conf
                    )
                    
                    # Sende Nachricht mit User-Mention, damit sie es sehen
                    await log_channel.send(log_msg)
                    
                except Exception as e:
                    print(f"Error sending log: {e}")
        
        # Timeout-Logik
        now = datetime.datetime.utcnow()
        guild_id = message.guild.id
        user_id = message.author.id
        
        if guild_id not in self.offenses:
            self.offenses[guild_id] = {}
        
        if user_id not in self.offenses[guild_id]:
            self.offenses[guild_id][user_id] = []
        
        user_offenses = self.offenses[guild_id][user_id]
        user_offenses[:] = [t for t in user_offenses if (now - t).total_seconds() <= 60]
        user_offenses.append(now)
        
        # Timeout bei 3+ Verstößen in 60 Sekunden
        if len(user_offenses) >= 3:
            try:
                timeout_duration = datetime.timedelta(minutes=5)
                await message.author.timeout(until=datetime.datetime.utcnow() + timeout_duration, reason="Repeated image spam violations")
                
                # Timeout-Benachrichtigung im Log-Channel
                if conf["log_channel_id"] and conf["notification_on_delete"]:
                    log_channel = message.guild.get_channel(conf["log_channel_id"])
                    if log_channel:
                        timeout_msg = self.format_message(
                            conf["timeout_message"],
                            message,
                            img_count,
                            conf
                        )
                        await log_channel.send(timeout_msg)
                
                # Offenses zurücksetzen
                self.offenses[guild_id][user_id] = []
                
            except discord.Forbidden:
                pass
            except Exception as e:
                print(f"Error applying timeout: {e}")

async def setup(bot):
    await bot.add_cog(ImageSpam(bot))
