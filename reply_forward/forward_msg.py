import discord
from redbot.core import commands, Config
import logging

logging.basicConfig(level=logging.INFO)

class UnsupportedMessageForwarder(commands.Cog):
    """A cog to forward messages using Discord's reply function with the !unsupported command.
    
    Only users with allowed roles (set via command) can use the command.
    The command must be used in reply to a message; the replied-to message is then forwarded
    to a target channel (set via command).
    """
    
    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=987654321)
        default_global = {
            "target_channel_id": None,
            "allowed_role_ids": []
        }
        self.config.register_global(**default_global)
    
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def settarget(self, ctx, channel: discord.TextChannel):
        """Sets the target channel where forwarded messages will be sent. (Admin only)"""
        await self.config.target_channel_id.set(channel.id)
        await ctx.send(f"Target channel has been set to: {channel.mention}")
    
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def addunsupportedrole(self, ctx, role: discord.Role):
        """Adds a role allowed to use the !unsupported command. (Admin only)"""
        roles = await self.config.allowed_role_ids()
        if role.id not in roles:
            roles.append(role.id)
            await self.config.allowed_role_ids.set(roles)
            await ctx.send(f"Role **{role.name}** has been added to allowed roles.")
        else:
            await ctx.send(f"Role **{role.name}** is already allowed.")
    
    @commands.command(name="removeunsupportedrole")
    @commands.has_permissions(administrator=True)
    async def _removeunsupportedrole(self, ctx, role: discord.Role):
        """Removes a role from the allowed roles. (Admin only)"""
        roles = await self.config.allowed_role_ids()
        if role.id in roles:
            roles.remove(role.id)
            await self.config.allowed_role_ids.set(roles)
            await ctx.send(f"Role **{role.name}** has been removed from allowed roles.")
        else:
            await ctx.send(f"Role **{role.name}** is not in the allowed roles.")
    
    @commands.command()
    async def listroles(self, ctx):
        """Lists the roles allowed to use the !unsupported command."""
        roles = await self.config.allowed_role_ids()
        if not roles:
            await ctx.send("No roles have been set to use this command.")
        else:
            role_names = []
            for role_id in roles:
                role = ctx.guild.get_role(role_id)
                if role:
                    role_names.append(role.name)
            await ctx.send("Allowed roles: " + ", ".join(role_names))
    
    @commands.command()
    async def unsupported(self, ctx):
        """Forwards the replied-to message to the configured target channel.
        
        This command must be used as a reply to a message.
        """
        # Check if the command message is a reply
        if ctx.message.reference is None:
            await ctx.send("❌ This command must be used as a reply to a message.")
            return
        
        # Check if the user has one of the allowed roles (if any have been set)
        allowed_roles = await self.config.allowed_role_ids()
        if allowed_roles:
            if not any(role.id in allowed_roles for role in ctx.author.roles):
                await ctx.send("❌ You do not have permission to use this command.")
                return
        
        # Retrieve the replied message
        try:
            replied_message = await ctx.channel.fetch_message(ctx.message.reference.message_id)
        except Exception as e:
            logging.error(f"Error fetching replied message: {e}")
            await ctx.send("❌ Could not retrieve the replied message.")
            return
        
        # Get the target channel from config
        target_channel_id = await self.config.target_channel_id()
        if target_channel_id is None:
            await ctx.send("❌ No target channel has been set. Use `!settarget` to configure one.")
            return
        
        target_channel = ctx.guild.get_channel(target_channel_id)
        if target_channel is None:
            try:
                target_channel = await ctx.guild.fetch_channel(target_channel_id)
            except Exception as e:
                logging.error(f"Error fetching target channel: {e}")
                await ctx.send("❌ The target channel is invalid or not accessible.")
                return
        
        # Create an embed with the original message details
        embed = discord.Embed(
            title="Forwarded Message",
            description=replied_message.content or "[No text content]",
            color=discord.Color.blue(),
            timestamp=replied_message.created_at
        )
        embed.set_author(
            name=f"{replied_message.author} in #{ctx.channel.name}",
            icon_url=replied_message.author.avatar.url if replied_message.author.avatar else discord.Embed.Empty
        )
        if replied_message.attachments:
            attachments = "\n".join([attachment.url for attachment in replied_message.attachments])
            embed.add_field(name="Attachments", value=attachments, inline=False)
        
        try:
            await target_channel.send(embed=embed)
            await ctx.send(f"✅ Message has been forwarded to {target_channel.mention}.")
        except discord.Forbidden:
            logging.error("Bot lacks permissions to send messages in the target channel.")
            await ctx.send("❌ Bot lacks permissions to send messages in the target channel.")
        except Exception as e:
            logging.error(f"Error sending forwarded message: {e}")
            await ctx.send("❌ Failed to forward the message.")
