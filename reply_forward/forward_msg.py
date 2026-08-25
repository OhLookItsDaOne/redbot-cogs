import discord
import logging
from redbot.core import commands, Config, app_commands

logging.basicConfig(level=logging.INFO)

class UnsupportedMessageForwarder(commands.Cog):
    """A cog to forward messages using Discord's message context menu.

    Users with allowed roles (set via command) can right-click a message,
    open Apps and choose "Forward to Support" to forward it to the configured
    target channel. This works without the Message Content intent.
    """

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=987654321)
        default_global = {
            "target_channel_id": None,
            "allowed_role_ids": []
        }
        self.config.register_global(**default_global)

    @commands.hybrid_command()
    @commands.has_permissions(administrator=True)
    async def settarget(self, ctx, channel: discord.TextChannel):
        """Sets the target channel where forwarded messages will be sent. (Admin only)"""
        await self.config.target_channel_id.set(channel.id)
        await ctx.send(f"Target channel has been set to: {channel.mention}")

    @commands.hybrid_command()
    @commands.has_permissions(administrator=True)
    async def addunsupportedrole(self, ctx, role: discord.Role):
        """Adds a role allowed to use the Forward to Support command. (Admin only)"""
        roles = await self.config.allowed_role_ids()
        if role.id not in roles:
            roles.append(role.id)
            await self.config.allowed_role_ids.set(roles)
            await ctx.send(f"Role **{role.name}** has been added to allowed roles.")
        else:
            await ctx.send(f"Role **{role.name}** is already allowed.")

    @commands.hybrid_command(name="removeunsupportedrole")
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

    @commands.hybrid_command()
    async def listroles(self, ctx):
        """Lists the roles allowed to use the Forward to Support command."""
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

    async def forward_message(self, interaction: discord.Interaction, message: discord.Message):
        """Shared implementation for the Forward to Support context menu."""
        if interaction.guild is None:
            await interaction.response.send_message("❌ This command must be used in a server.", ephemeral=True)
            return

        allowed_roles = await self.config.allowed_role_ids()
        if allowed_roles:
            if not any(role.id in allowed_roles for role in interaction.user.roles):
                await interaction.response.send_message("❌ You do not have permission to use this command.", ephemeral=True)
                return

        target_channel_id = await self.config.target_channel_id()
        if target_channel_id is None:
            await interaction.response.send_message("❌ No target channel has been set. Use `/settarget` to configure one.", ephemeral=True)
            return

        target_channel = interaction.guild.get_channel(target_channel_id)
        if target_channel is None:
            try:
                target_channel = await interaction.guild.fetch_channel(target_channel_id)
            except Exception as e:
                logging.error(f"Error fetching target channel: {e}")
                await interaction.response.send_message("❌ The target channel is invalid or not accessible.", ephemeral=True)
                return

        embed = discord.Embed(
            title="Forwarded Message",
            description=message.content or "[No text content]",
            color=discord.Color.blue(),
            timestamp=message.created_at
        )
        embed.set_author(
            name=f"{message.author} in #{message.channel.name}",
            icon_url=message.author.avatar.url if message.author.avatar else None
        )
        if message.attachments:
            attachments = "\n".join([attachment.url for attachment in message.attachments])
            embed.add_field(name="Attachments", value=attachments, inline=False)

        try:
            await target_channel.send(embed=embed)
            await interaction.response.send_message(f"✅ Message has been forwarded to {target_channel.mention}.", ephemeral=True)
        except discord.Forbidden:
            logging.error("Bot lacks permissions to send messages in the target channel.")
            await interaction.response.send_message("❌ Bot lacks permissions to send messages in the target channel.", ephemeral=True)
        except Exception as e:
            logging.error(f"Error sending forwarded message: {e}")
            await interaction.response.send_message("❌ Failed to forward the message.", ephemeral=True)


@app_commands.context_menu(name="Forward to Support")
async def forward_to_support(interaction: discord.Interaction, message: discord.Message):
    cog = interaction.client.get_cog("UnsupportedMessageForwarder")
    if cog is None:
        await interaction.response.send_message("❌ Cog is not loaded.", ephemeral=True)
        return
    await cog.forward_message(interaction, message)
