import discord
from redbot.core import commands, Config

def automod_manage_check():
    async def predicate(ctx):
        cog = ctx.cog
        return await cog.has_automod_permission(ctx)
    return commands.check(predicate)

class D1AutoMod(commands.Cog):
    """AutoMod: Manage Discord AutoMod keyword rules and allowed words/phrases interactively."""

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
        try:
            rules = await guild.fetch_automod_rules()
        except Exception:
            return {}
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
    @automod_manage_check()
    async def automod(self, ctx, rule: str = None, action: str = None, *args):
        """
        Show info about an AutoMod rule. (All management via commands!)
        Usage:
        !automod <shortname> [enable|disable]
        !automod <shortname> allow add word1,word2
        !automod <shortname> allow remove word1,word2
        !automod list
        """
        if rule is None:
            return await ctx.send_help()

        shortmap = await self.get_shortname_mapping(ctx.guild)
        rule_id = shortmap.get(rule) if not rule.isdigit() else int(rule)
        if not rule_id:
            return await ctx.send(
                "Rule not found by short name or ID.\n"
                "Use `!automod list` to see available rules and their short names."
            )
        try:
            rule_obj = await ctx.guild.fetch_automod_rule(rule_id)
        except Exception as e:
            return await ctx.send(f"Could not fetch rule: {e}")

        trigger = getattr(rule_obj, "trigger", None)
        trigger_type = getattr(trigger, "type", None)
        readable_type = str(trigger_type)
        enabled = getattr(rule_obj, "enabled", None)
        rule_fields = []
        if hasattr(rule_obj, "trigger_metadata"):
            meta = rule_obj.trigger_metadata
            if hasattr(meta, "keyword_filter"):
                rule_fields.append(("Keyword Filter", ", ".join(meta.keyword_filter) or "*None*"))
            if hasattr(meta, "allow_list"):
                rule_fields.append(("Allowed List", ", ".join(meta.allow_list) or "*None*"))
            if hasattr(meta, "regex_patterns"):
                rule_fields.append(("Regex Patterns", ", ".join(meta.regex_patterns) or "*None*"))
            if hasattr(meta, "mention_total_limit"):
                rule_fields.append(("Mention Limit", str(meta.mention_total_limit)))

        embed = discord.Embed(
            title=f"AutoMod Rule: {rule_obj.name}",
            description=(
                f"Type: `{readable_type}`\n"
                f"Enabled: {enabled}\n"
                f"Rule ID: {rule_obj.id}"
            ),
        )
        for name, value in rule_fields:
            embed.add_field(name=name, value=value, inline=False)

        await ctx.send(embed=embed)

    @automod.command(name="enable")
    @automod_manage_check()
    async def enable_rule(self, ctx, rule: str):
        """Enable an AutoMod rule."""
        shortmap = await self.get_shortname_mapping(ctx.guild)
        rule_id = shortmap.get(rule) if not rule.isdigit() else int(rule)
        if not rule_id:
            return await ctx.send("Rule not found by short name or ID.")
        rule_obj = await ctx.guild.fetch_automod_rule(rule_id)
        await rule_obj.edit(enabled=True)
        await ctx.send(f"Rule **{rule_obj.name}** is now **enabled**.")

    @automod.command(name="disable")
    @automod_manage_check()
    async def disable_rule(self, ctx, rule: str):
        """Disable an AutoMod rule."""
        shortmap = await self.get_shortname_mapping(ctx.guild)
        rule_id = shortmap.get(rule) if not rule.isdigit() else int(rule)
        if not rule_id:
            return await ctx.send("Rule not found by short name or ID.")
        rule_obj = await ctx.guild.fetch_automod_rule(rule_id)
        await rule_obj.edit(enabled=False)
        await ctx.send(f"Rule **{rule_obj.name}** is now **disabled**.")

    @automod.group(name="allow")
    @automod_manage_check()
    async def allow(self, ctx, rule: str = None):
        """Manage allowed words/phrases for a keyword rule."""
        if rule is None:
            return await ctx.send("You must specify a rule.")

    @allow.command(name="add")
    @automod_manage_check()
    async def allow_add(self, ctx, rule: str, *, words: str):
        """Add allowed words/phrases (comma or newline separated)."""
        shortmap = await self.get_shortname_mapping(ctx.guild)
        rule_id = shortmap.get(rule) if not rule.isdigit() else int(rule)
        if not rule_id:
            return await ctx.send("Rule not found by short name or ID.")
        rule_obj = await ctx.guild.fetch_automod_rule(rule_id)
        if not hasattr(rule_obj.trigger_metadata, "allow_list"):
            return await ctx.send("This rule does not support allowed words.")
        added = [w.strip() for w in words.replace("\n", ",").split(",") if w.strip()]
        allow_list = set(rule_obj.trigger_metadata.allow_list or [])
        before = set(allow_list)
        allow_list.update(added)
        try:
            await rule_obj.edit(
                trigger_metadata=discord.AutoModRuleTriggerMetadata(
                    allow_list=list(allow_list),
                    keyword_filter=rule_obj.trigger_metadata.keyword_filter
                )
            )
            msg = f"Added: {', '.join(set(added) - before)}" if set(added) - before else "No new words added."
        except Exception as e:
            msg = f"Failed to update rule: {e}"
        await ctx.send(msg)

    @allow.command(name="remove")
    @automod_manage_check()
    async def allow_remove(self, ctx, rule: str, *, words: str):
        """Remove allowed words/phrases (comma or newline separated)."""
        shortmap = await self.get_shortname_mapping(ctx.guild)
        rule_id = shortmap.get(rule) if not rule.isdigit() else int(rule)
        if not rule_id:
            return await ctx.send("Rule not found by short name or ID.")
        rule_obj = await ctx.guild.fetch_automod_rule(rule_id)
        if not hasattr(rule_obj.trigger_metadata, "allow_list"):
            return await ctx.send("This rule does not support allowed words.")
        to_remove = [w.strip() for w in words.replace("\n", ",").split(",") if w.strip()]
        allow_list = set(rule_obj.trigger_metadata.allow_list or [])
        before = set(allow_list)
        allow_list.difference_update(to_remove)
        try:
            await rule_obj.edit(
                trigger_metadata=discord.AutoModRuleTriggerMetadata(
                    allow_list=list(allow_list),
                    keyword_filter=rule_obj.trigger_metadata.keyword_filter
                )
            )
            msg = f"Removed: {', '.join(before - set(allow_list))}" if before - set(allow_list) else "No words were removed."
        except Exception as e:
            msg = f"Failed to update rule: {e}"
        await ctx.send(msg)

    @automod.command(name="list")
    @automod_manage_check()
    async def list_rules(self, ctx):
        """List all AutoMod rules with their short names."""
        try:
            rules = await ctx.guild.fetch_automod_rules()
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
        await ctx.send(f"**AutoMod rules and short names:**\n{msg or 'None'}")

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
    @automod_manage_check()
    async def roles(self, ctx):
        """List all roles allowed to use automod management commands."""
        allowed_roles = await self.config.guild(ctx.guild).allowed_roles()
        if not allowed_roles:
            return await ctx.send("No roles are currently allowed to use automod management commands.")
        mentions = [f"<@&{role_id}>" for role_id in allowed_roles if ctx.guild.get_role(role_id)]
        await ctx.send("Allowed roles: " + ", ".join(mentions) if mentions else "No valid roles found.")

async def setup(bot):
    await bot.add_cog(D1AutoMod(bot))
