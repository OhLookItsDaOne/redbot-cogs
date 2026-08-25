import discord
import logging
import re
from redbot.core import commands, Config, app_commands

logging.basicConfig(level=logging.INFO)

MESSAGE_LINK_RE = re.compile(
    r"https?://(?:canary|ptb|www\.)?discord(?:app)?\.com/channels/(\d+)/(\d+)/(\d+)"
)


class MessageForwarder(commands.Cog):
    """Forward messages to a configured target channel.

    Supported ways to forward:
    - Message context menu (right click -> Apps -> Forward to Support)
    - Slash/prefix command ``/forward message <message link>`` (or as a reply)
    The forwarded message mentions the author, links back to the original message
    and includes attachments.
    """

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=987654321)
        default_global = {
            "target_channel_id": None,
            "allowed_role_ids": []
        }
        self.config.register_global(**default_global)

    @commands.hybrid_group(name="forward", invoke_without_command=True, extras={"red_force_enable": True})
    @commands.guild_only()
    async def forward(self, ctx):
        """Forward messages to the configured target channel."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help(ctx.command)

    @forward.command(name="message", aliases=["unsupported"])
    async def forward_message(self, ctx, message_link: str = None):
        """Forwards a message to the configured target channel.

        Provide a Discord message link, or use this command as a reply to a message.

        **Examples:**
        - `/forward message https://discord.com/channels/123/456/789`
        - Reply to a message and type `!forward message` (or `!unsupported`)
        """
        respond = lambda msg: ctx.send(msg)

        # Determine the target message: from link or reply
        target_message = None
        if message_link:
            target_message = await self._resolve_link(ctx, message_link, respond)
            if target_message is None:
                return
        elif ctx.message.reference and ctx.message.reference.message_id:
            try:
                target_message = await ctx.channel.fetch_message(ctx.message.reference.message_id)
            except Exception as e:
                logging.error(f"Error fetching replied message: {e}")
                await respond("❌ Could not retrieve the replied message.")
                return
        else:
            await respond(
                "❌ Please provide a Discord message link, or use this command as a reply to a message."
            )
            return

        await self._forward(ctx.author, ctx.guild, target_message, respond)

    @forward.command(name="settarget")
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def settarget(self, ctx, channel: discord.TextChannel):
        """Sets the target channel where forwarded messages will be sent (Admin only)."""
        await self.config.target_channel_id.set(channel.id)
        await ctx.send(f"Target channel has been set to: {channel.mention}")

    @forward.command(name="addrole")
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def addrole(self, ctx, role: discord.Role):
        """Adds a role allowed to use the forward command (Admin only)."""
        roles = await self.config.allowed_role_ids()
        if role.id not in roles:
            roles.append(role.id)
            await self.config.allowed_role_ids.set(roles)
            await ctx.send(f"Role **{role.name}** has been added to allowed roles.")
        else:
            await ctx.send(f"Role **{role.name}** is already allowed.")

    @forward.command(name="removerole")
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def removerole(self, ctx, role: discord.Role):
        """Removes a role from the allowed roles (Admin only)."""
        roles = await self.config.allowed_role_ids()
        if role.id in roles:
            roles.remove(role.id)
            await self.config.allowed_role_ids.set(roles)
            await ctx.send(f"Role **{role.name}** has been removed from allowed roles.")
        else:
            await ctx.send(f"Role **{role.name}** is not in the allowed roles.")

    @forward.command(name="listroles")
    @commands.has_permissions(administrator=True)
    @app_commands.default_permissions(administrator=True)
    async def listroles(self, ctx):
        """Lists the roles allowed to use the forward command."""
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

    async def _resolve_link(self, ctx, link: str, respond) -> discord.Message:
        """Parses a Discord message link and fetches the target message."""
        match = MESSAGE_LINK_RE.search(link)
        if not match:
            await respond("❌ That doesn't look like a valid Discord message link.")
            return None
        guild_id, channel_id, message_id = (int(g) for g in match.groups())

        if ctx.guild.id != guild_id:
            await respond("❌ The message link is not from this server.")
            return None

        channel = ctx.guild.get_channel(channel_id)
        if channel is None:
            try:
                channel = await ctx.guild.fetch_channel(channel_id)
            except Exception as e:
                logging.error(f"Error fetching channel from link: {e}")
                await respond("❌ Could not access the channel from the link.")
                return None

        try:
            return await channel.fetch_message(message_id)
        except Exception as e:
            logging.error(f"Error fetching message from link: {e}")
            await respond("❌ Could not fetch the message from the link.")
            return None

    async def _forward(self, author, guild, message: discord.Message, respond) -> None:
        """Shared implementation: forwards a message to the configured target channel."""
        if guild is None:
            await respond("❌ This command must be used in a server.")
            return

        allowed_roles = await self.config.allowed_role_ids()
        if allowed_roles:
            if not any(role.id in allowed_roles for role in author.roles):
                await respond("❌ You do not have permission to use this command.")
                return

        target_channel_id = await self.config.target_channel_id()
        if target_channel_id is None:
            await respond("❌ No target channel has been set. Use `/forward settarget` to configure one.")
            return

        target_channel = guild.get_channel(target_channel_id)
        if target_channel is None:
            try:
                target_channel = await guild.fetch_channel(target_channel_id)
            except Exception as e:
                logging.error(f"Error fetching target channel: {e}")
                await respond("❌ The target channel is invalid or not accessible.")
                return

        # Layout: Mention first, then text, then original link, then attachments.
        # Attachments are inserted as URLs - Discord renders images automatically
        # as embeds (no download/storage on the server).
        lines = [f"**Forwarded message from:** {message.author.mention}"]
        if message.content:
            lines.append(message.content)
        lines.append(f"🔗 [Original message]({message.jump_url})")
        for attachment in message.attachments:
            lines.append(attachment.url)
        content = "\n".join(lines)

        try:
            await target_channel.send(content=content)
            await respond(f"✅ Message has been forwarded to {target_channel.mention}.")
        except discord.Forbidden:
            logging.error("Bot lacks permissions to send messages in the target channel.")
            await respond("❌ Bot lacks permissions to send messages in the target channel.")
        except Exception as e:
            logging.error(f"Error sending forwarded message: {e}")
            await respond("❌ Failed to forward the message.")


@app_commands.context_menu(name="Forward to Support", extras={"red_force_enable": True})
async def forward_to_support(interaction: discord.Interaction, message: discord.Message):
    cog = interaction.client.get_cog("MessageForwarder")
    if cog is None:
        await interaction.response.send_message("❌ Cog is not loaded.", ephemeral=True)
        return
    respond = lambda msg: interaction.response.send_message(msg, ephemeral=True)
    await cog._forward(interaction.user, interaction.guild, message, respond)
