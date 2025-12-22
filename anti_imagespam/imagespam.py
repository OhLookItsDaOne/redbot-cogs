import discord
from redbot.core import commands, Config
from redbot.core.utils.chat_formatting import pagify, box
import datetime
from typing import Optional, List
import re

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
            "notification_on_delete": True,
            "user_notification": True,
            "user_message": "Your message in {channel} was deleted because it contained {image_count} images (maximum allowed: {max_images}).",
            "log_message": "🚫 **Image Spam Blocked**\nUser: {user_mention}\nChannel: {channel_mention}\nImages: {image_count}/{max_images}",
            "timeout_message": "⏰ **User Timed Out**\n{user_mention} has been timed out for 5 minutes due to repeated violations.",
            "placeholder_info": "Available placeholders:\n{user} - Username\n{user_mention} - @User\n{channel} - Channel name\n{channel_mention} - #channel\n{max_images} - Max allowed images\n{image_count} - Images in message\n{guild} - Server name",
            "count_discord_links": True  # Neue Einstellung für Discord-Links
        }
        self.config.register_guild(**default_guild)
        self.offenses = {}
        
        # Regex für Discord-CDN-Links
        self.discord_cdn_pattern = re.compile(
            r'https?://(?:cdn\.discordapp\.com|media\.discordapp\.net)/attachments/\d+/\d+/[^\s]+'
        )
        
        # Regex für allgemeine Bild-URLs mit Erweiterungen
        self.image_url_pattern = re.compile(
            r'https?://[^\s]+\.(?:png|jpg|jpeg|gif|webp|bmp|tiff)(?:\?[^\s]*)?',
            re.IGNORECASE
        )

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

    def count_images_in_message(self, message: discord.Message, conf: dict) -> int:
        """Count images in a message, including attachments and Discord CDN links."""
        img_count = 0
        
        # 1. Count attachments (Dateianhänge)
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith("image/"):
                img_count += 1
            elif any(attachment.filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                img_count += 1
        
        # 2. Count Discord CDN links if enabled
        if conf.get("count_discord_links", True):
            # Discord CDN links (like your example)
            discord_links = self.discord_cdn_pattern.findall(message.content)
            img_count += len(discord_links)
            
            # General image URLs with extensions
            image_urls = self.image_url_pattern.findall(message.content)
            # Filter out URLs that are already counted as Discord CDN links
            image_urls = [url for url in image_urls if not any(url.startswith(cdn_url) for cdn_url in discord_links)]
            img_count += len(image_urls)
        
        return img_count

    def get_monitored_channels(self, guild: discord.Guild, conf: dict) -> List[int]:
        """Get list of channel IDs that are currently being monitored."""
        monitored = []
        
        if conf["monitor_all"]:
            for channel in guild.text_channels:
                if isinstance(channel, discord.TextChannel):
                    if channel.id not in conf["excluded_channels"]:
                        monitored.append(channel.id)
        
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
        try:
            await ctx.author.send(content)
        except discord.Forbidden:
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
                  "• `!imageprevent forumthreads <on/off>` - Auto-exclude forum threads\n"
                  "• `!imageprevent discordlinks <on/off>` - Count Discord CDN image links",
            inline=False
        )
        
        embed.add_field(
            name="🚫 Channel Management",
            value="• `!imageprevent exclude <#channel>` - Exclude channel from monitoring\n"
                  "• `!imageprevent include <#channel>` - Remove channel from exclusion list",
            inline=False
        )
        
        embed.add_field(
            name="📊 Information",
            value="• `!imageprevent list` - Show settings\n"
                  "• `!imageprevent channels` - Show all text channels and their status\n"
                  "• `!imageprevent status` - Check channel status\n"
                  "• `!imageprevent test <message>` - Test image counting in a message",
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

    @imageprevent.command(name="discordlinks")
    async def toggle_discord_links(self, ctx, state: str):
        """Toggle whether to count Discord CDN image links (on/off)."""
        if not await self.check_admin_or_role(ctx):
            return
        
        if state.lower() in ["on", "true", "yes", "enable"]:
            await self.config.guild(ctx.guild).count_discord_links.set(True)
            await ctx.send("✅ **Discord CDN link counting is now ON.** Discord image links will be counted as images.")
        elif state.lower() in ["off", "false", "no", "disable"]:
            await self.config.guild(ctx.guild).count_discord_links.set(False)
            await ctx.send("✅ **Discord CDN link counting is now OFF.** Only image attachments will be counted.")
        else:
            await ctx.send("❌ Please use `on` or `off`.")

    @imageprevent.command(name="test")
    async def test_image_counting(self, ctx, *, message: str):
        """Test how many images would be counted in a message."""
        conf = await self.config.guild(ctx.guild).all()
        
        class MockMessage:
            def __init__(self, content, channel, guild, author):
                self.content = content
                self.channel = channel
                self.guild = guild
                self.author = author
                self.attachments = []
        
        mock_msg = MockMessage(message, ctx.channel, ctx.guild, ctx.author)
        img_count = self.count_images_in_message(mock_msg, conf)
        
        # Extrahiere alle gefundenen Links
        discord_links = self.discord_cdn_pattern.findall(message)
        image_urls = self.image_url_pattern.findall(message)
        
        embed = discord.Embed(
            title="🔍 Image Counting Test",
            color=discord.Color.blue()
        )
        
        embed.add_field(name="Test Message", value=f"```{message[:500]}{'...' if len(message) > 500 else ''}```", inline=False)
        embed.add_field(name="Images Counted", value=f"**{img_count}**", inline=True)
        embed.add_field(name="Max Allowed", value=f"**{conf['max_images']}**", inline=True)
        embed.add_field(name="Would be deleted?", value=f"**{'Yes ✅' if img_count > conf['max_images'] else 'No ❌'}**", inline=True)
        
        if discord_links:
            embed.add_field(
                name=f"Discord CDN Links ({len(discord_links)})",
                value="\n".join([f"• {link[:50]}..." if len(link) > 50 else f"• {link}" for link in discord_links[:3]]),
                inline=False
            )
        
        if image_urls:
            embed.add_field(
                name=f"Image URLs ({len(image_urls)})",
                value="\n".join([f"• {url[:50]}..." if len(url) > 50 else f"• {url}" for url in image_urls[:3]]),
                inline=False
            )
        
        embed.set_footer(text=f"Discord link counting: {'ON' if conf.get('count_discord_links', True) else 'OFF'}")
        await ctx.send(embed=embed)

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
            await ctx.send("✅ Now monitoring **all non-excluded text channels**.")
        elif state.lower() in ["off", "false", "no", "disable"]:
            await self.config.guild(ctx.guild).monitor_all.set(False)
            await ctx.send("✅ Now monitoring **NO text channels** (monitorall is off).")
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
            await ctx.send("✅ **Forum threads are now automatically excluded from monitoring.**")
        elif state.lower() in ["off", "false", "no", "disable"]:
            await self.config.guild(ctx.guild).exclude_forum_threads.set(False)
            await ctx.send("✅ **Forum threads are no longer auto-excluded.** They will be monitored like regular text channels.")
        else:
            await ctx.send("❌ Please use `on` or `off`.")

    @imageprevent.command(name="exclude")
    async def add_excluded_channel(self, ctx, channel: discord.TextChannel):
        """Exclude a text channel from monitoring."""
        if not await self.check_admin_or_role(ctx):
            return
        
        if not isinstance(channel, discord.TextChannel):
            await ctx.send("❌ You can only exclude text channels.")
            return
        
        async with self.config.guild(ctx.guild).excluded_channels() as excluded:
            if channel.id not in excluded:
                excluded.append(channel.id)
        
        await ctx.send(f"✅ {channel.mention} is now **excluded** from monitoring.")

    @imageprevent.command(name="include")
    async def remove_excluded_channel(self, ctx, channel: discord.TextChannel):
        """Remove a text channel from exclusion list."""
        if not await self.check_admin_or_role(ctx):
            return
        
        async with self.config.guild(ctx.guild).excluded_channels() as excluded:
            if channel.id in excluded:
                excluded.remove(channel.id)
                await ctx.send(f"✅ {channel.mention} is no longer excluded.")
            else:
                await ctx.send(f"❌ {channel.mention} is not in the exclusion list.")

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
        
        embed.add_field(name="Monitoring Mode", value=f"**{'All non-excluded' if conf['monitor_all'] else 'NO channels'}**", inline=True)
        embed.add_field(name="Admin Monitoring", value=f"**{'ON' if conf['monitor_admins'] else 'OFF'}**", inline=True)
        embed.add_field(name="Forum Threads", value=f"**{'Auto-excluded' if conf['exclude_forum_threads'] else 'Monitored'}**", inline=True)
        
        embed.add_field(name="Discord Links", value=f"**{'Counted' if conf.get('count_discord_links', True) else 'Ignored'}**", inline=True)
        embed.add_field(name="Excluded Channels", value=f"**{len(conf['excluded_channels'])}** channels", inline=True)
        embed.add_field(name="Notifications", value=f"Log: **{'ON' if conf['notification_on_delete'] else 'OFF'}**\nUser: **{'ON' if conf['user_notification'] else 'OFF'}**", inline=True)
        
        # Zeige die ersten 5 exkludierten Channels
        excluded_list = []
        for cid in conf["excluded_channels"][:5]:
            channel = ctx.guild.get_channel(cid)
            if channel and isinstance(channel, discord.TextChannel):
                excluded_list.append(f"• #{channel.name}")
        
        if excluded_list:
            excluded_text = "\n".join(excluded_list)
            if len(conf["excluded_channels"]) > 5:
                excluded_text += f"\n... and {len(conf['excluded_channels']) - 5} more"
            embed.add_field(name="Excluded Channels", value=excluded_text, inline=False)
        
        await ctx.send(embed=embed)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Monitor messages for image spam."""
        if message.author.bot or not message.guild:
            return
        
        conf = await self.config.guild(message.guild).all()
        
        if not self.should_monitor_message(message, conf):
            return
        
        # Bilder zählen (inklusive Discord-Links)
        img_count = self.count_images_in_message(message, conf)
        
        if img_count <= conf["max_images"]:
            return
        
        # Debug-Ausgabe
        print(f"[ImagePrevent] Detected {img_count} images in message from {message.author} in #{message.channel.name}")
        print(f"[ImagePrevent] Message content: {message.content[:100]}...")
        
        # Nachricht löschen
        delete_success = False
        try:
            await message.delete()
            delete_success = True
            print(f"[ImagePrevent] Deleted message with {img_count} images (max: {conf['max_images']})")
        except (discord.Forbidden, discord.NotFound) as e:
            delete_success = False
            print(f"[ImagePrevent] Failed to delete message: {e}")
        except Exception as e:
            print(f"[ImagePrevent] Error deleting message: {e}")
            delete_success = False
        
        # Benutzer benachrichtigen (ephemeral-like)
        if conf["user_notification"] and delete_success:
            user_msg = self.format_message(
                conf["user_message"],
                message,
                img_count,
                conf
            )
            
            try:
                await message.author.send(user_msg)
            except discord.Forbidden:
                try:
                    temp_msg = await message.channel.send(
                        f"{message.author.mention} (Only you can see this): {user_msg}",
                        delete_after=10.0
                    )
                except Exception:
                    pass
        
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
                    
                    await log_channel.send(log_msg)
                    
                except Exception as e:
                    print(f"[ImagePrevent] Error sending log: {e}")
        
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
                await message.author.timeout(
                    until=datetime.datetime.utcnow() + timeout_duration, 
                    reason="Repeated image spam violations"
                )
                
                print(f"[ImagePrevent] Timed out {message.author} for repeated violations")
                
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
                
                self.offenses[guild_id][user_id] = []
                
            except discord.Forbidden:
                print(f"[ImagePrevent] No permission to timeout {message.author}")
            except Exception as e:
                print(f"[ImagePrevent] Error applying timeout: {e}")

async def setup(bot):
    await bot.add_cog(ImageSpam(bot))
