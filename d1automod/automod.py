import discord
from redbot.core import commands, Config

class AutoMod(commands.Cog):
    """AutoMod: Manage rules and allowed words/phrases interactively."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=2468101214)
        default_guild = {"allowed_roles": [], "shortnames": {}}
        self.config.register_guild(**default_guild)

    async def has_automod_permission(self, ctx):
        if ctx.author.guild_permissions.administrator:
            return True
        allowed_roles = await self.config.guild(ctx.guild).allowed_roles()
        return any(role.id in allowed_roles for role in ctx.author.roles)

    async def get_shortname_mapping(self, guild):
        mapping = await self.config.guild(guild).shortnames()
        rules = await guild.fetch_auto_moderation_rules()
        names_used = set()
        newmap = {}
        for rule in rules:
            words = rule.name.lower().split()
            if not words:
                continue
            short = words[0]
            if short in names_used:
                if len(words) > 1:
                    short = words[0] + words[1]
                else:
                    short = words[0] + str(rule.id)[-3:]
            while short in names_used:
                short += str(rule.id)[-1]
            newmap[short] = rule.id
            names_used.add(short)
        await self.config.guild(guild).shortnames.set(newmap)
        return newmap

    @commands.group(invoke_without_command=True)
    @commands.guild_only()
    async def automod(self, ctx, rule: str = None):
        """Manage AutoMod rules and allowed words/phrases interactively.
        
        Just use: !automod <shortname>
        Example: !automod spam
        """
        if rule is None:
            return await ctx.send_help()

        if not await self.has_automod_permission(ctx):
            return await ctx.send("You do not have permission to use this command.")

        shortmap = await self.get_shortname_mapping(ctx.guild)
        rule_id = shortmap.get(rule) if not rule.isdigit() else int(rule)
        if not rule_id:
            return await ctx.send("Rule not found by short name or ID.")

        try:
            rule_obj = await ctx.guild.fetch_auto_moderation_rule(rule_id)
        except Exception as e:
            return await ctx.send(f"Could not fetch rule: {e}")

        if rule_obj.trigger_type != discord.AutoModTriggerType.keyword:
            return await ctx.send("This rule is not a keyword rule (only keyword rules support allowed words/phrases).")

        view = AllowWordsView(self, rule_obj)
        await ctx.send(
            embed=await view.get_embed(),
            view=view
        )

    @automod.command(name="allowrole")
    @commands.has_guild_permissions(administrator=True)
    async def allowrole(self, ctx, role: discord.Role):
        """Allow a role to use automod management commands."""
        allowed_roles = await self.config.guild(ctx.guild).allowed_roles()
        if role.id not in allowed_roles:
            allowed_roles.append(role.id)
            await self.config.guild(ctx.guild).allowed_roles.set(allowed_roles)
            await ctx.send(f"{role.mention} can now use automod management commands.")
        else:
            await ctx.send(f"{role.mention} is already allowed.")

    @automod.command(name="removerole")
    @commands.has_guild_permissions(administrator=True)
    async def removerole(self, ctx, role: discord.Role):
        """Remove a role from automod management access."""
        allowed_roles = await self.config.guild(ctx.guild).allowed_roles()
        if role.id in allowed_roles:
            allowed_roles.remove(role.id)
            await self.config.guild(ctx.guild).allowed_roles.set(allowed_roles)
            await ctx.send(f"{role.mention} can no longer use automod management commands.")
        else:
            await ctx.send(f"{role.mention} is not in the allowed roles.")

    @automod.command(name="roles")
    async def roles(self, ctx):
        """List all roles allowed to use automod management commands."""
        allowed_roles = await self.config.guild(ctx.guild).allowed_roles()
        if not allowed_roles:
            return await ctx.send("No roles are currently allowed to use automod management commands.")
        mentions = [f"<@&{role_id}>" for role_id in allowed_roles if ctx.guild.get_role(role_id)]
        await ctx.send("Allowed roles: " + ", ".join(mentions) if mentions else "No valid roles found.")

    @automod.command(name="list")
    async def list_rules(self, ctx):
        """List all AutoMod rules with their short names."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("You do not have permission to use this command.")
        try:
            rules = await ctx.guild.fetch_auto_moderation_rules()
        except Exception as e:
            return await ctx.send(f"Failed to fetch rules: {e}")
        if not rules:
            return await ctx.send("No AutoMod rules found.")
        shortmap = await self.get_shortname_mapping(ctx.guild)
        msg = ""
        for shortname, ruleid in shortmap.items():
            rule = discord.utils.get(rules, id=ruleid)
            if rule:
                msg += f"**{shortname}** — {rule.name} (ID: `{rule.id}`)\n"
        await ctx.send(f"**AutoMod rules and short names:**\n{msg}")

class AllowWordsView(discord.ui.View):
    def __init__(self, cog, rule_obj):
        super().__init__(timeout=180)
        self.cog = cog
        self.rule_obj = rule_obj

    async def get_embed(self):
        allow_list = self.rule_obj.trigger_metadata.allow_list or []
        embed = discord.Embed(
            title=f"Manage allowed words/phrases for '{self.rule_obj.name}'",
            description="Use the buttons below to add or remove allowed words/phrases."
        )
        if allow_list:
            embed.add_field(
                name="Currently allowed words/phrases",
                value="\n".join(allow_list),
                inline=False
            )
        else:
            embed.add_field(
                name="Currently allowed words/phrases",
                value="*No allowed words set.*",
                inline=False
            )
        return embed

    @discord.ui.button(label="Add allowed word(s)", style=discord.ButtonStyle.green)
    async def add_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(AddWordModal(self.cog, self.rule_obj))

    @discord.ui.button(label="Remove allowed word(s)", style=discord.ButtonStyle.danger)
    async def remove_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_modal(RemoveWordModal(self.cog, self.rule_obj))

    @discord.ui.button(label="Show current list", style=discord.ButtonStyle.blurple)
    async def show_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.edit_message(embed=await self.get_embed(), view=self)

class AddWordModal(discord.ui.Modal):
    def __init__(self, cog, rule_obj):
        super().__init__(title="Add allowed word(s) or phrase(s)")
        self.cog = cog
        self.rule_obj = rule_obj
        self.add_item(discord.ui.TextInput(
            label="Enter words/phrases, comma or newline separated:",
            style=discord.TextStyle.paragraph,
            required=True
        ))

    async def on_submit(self, interaction: discord.Interaction):
        added = [w.strip() for w in self.children[0].value.replace("\n", ",").split(",") if w.strip()]
        allow_list = set(self.rule_obj.trigger_metadata.allow_list or [])
        before = list(allow_list)
        allow_list.update(added)
        try:
            await self.rule_obj.edit(
                trigger_metadata=discord.AutoModRuleTriggerMetadata(
                    allow_list=list(allow_list),
                    keyword_filter=self.rule_obj.trigger_metadata.keyword_filter
                )
            )
            msg = f"Added: {', '.join(set(added) - set(before))}" if set(added) - set(before) else "No new words added."
        except Exception as e:
            msg = f"Failed to update rule: {e}"
        await interaction.response.send_message(msg, ephemeral=True)

class RemoveWordModal(discord.ui.Modal):
    def __init__(self, cog, rule_obj):
        super().__init__(title="Remove allowed word(s) or phrase(s)")
        self.cog = cog
        self.rule_obj = rule_obj
        self.add_item(discord.ui.TextInput(
            label="Enter words/phrases to remove (comma or newline):",
            style=discord.TextStyle.paragraph,
            required=True
        ))

    async def on_submit(self, interaction: discord.Interaction):
        to_remove = [w.strip() for w in self.children[0].value.replace("\n", ",").split(",") if w.strip()]
        allow_list = set(self.rule_obj.trigger_metadata.allow_list or [])
        before = set(allow_list)
        allow_list.difference_update(to_remove)
        try:
            await self.rule_obj.edit(
                trigger_metadata=discord.AutoModRuleTriggerMetadata(
                    allow_list=list(allow_list),
                    keyword_filter=self.rule_obj.trigger_metadata.keyword_filter
                )
            )
            msg = f"Removed: {', '.join(before - set(allow_list))}" if before - set(allow_list) else "No words were removed."
        except Exception as e:
            msg = f"Failed to update rule: {e}"
        await interaction.response.send_message(msg, ephemeral=True)

async def setup(bot):
    await bot.add_cog(AutoMod(bot))

