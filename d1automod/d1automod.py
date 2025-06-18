import discord
from discord.automod import AutoModTriggerMetadata
from redbot.core import commands, Config
from typing import List

class D1AutoMod(commands.Cog):
    """AutoMod: Manage Discord AutoMod rules via simple commands."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=2468101214)
        default = {"allowed_roles": [], "shortnames": {}}
        self.config.register_guild(**default)

    async def has_automod_permission(self, ctx):
        # Admins always allowed
        if ctx.author.guild_permissions.administrator:
            return True
        allowed = await self.config.guild(ctx.guild).allowed_roles()
        return any(r.id in allowed for r in ctx.author.roles)

    async def get_shortname_mapping(self, guild: discord.Guild):
        # Build shortname→ID map
        mapping = {}
        seen = set()
        try:
            rules = await guild.fetch_automod_rules()
        except Exception:
            return {}
        for rule in rules:
            parts = rule.name.lower().split()
            if not parts:
                continue
            short = parts[0]
            if short in seen and len(parts) > 1:
                short = parts[0] + parts[1]
            # ensure unique
            while short in seen:
                short += str(rule.id)[-1]
            mapping[short] = rule.id
            seen.add(short)
        await self.config.guild(guild).shortnames.set(mapping)
        return mapping

    @commands.group(name="automod", invoke_without_command=True)
    @commands.guild_only()
    async def automod(self, ctx, rule: str = None):
        """
        Show info about an AutoMod rule.

        Usage:
          !automod <shortname>
          !automod list
          !automod roles
          !automod allowrole <role>
          !automod removerole <role>
          !automod <shortname> enable|disable
          !automod <shortname> add <word1,word2>
          !automod <shortname> remove <word1,word2>
        """
        if rule is None:
            return await ctx.send_help()

        if rule.lower() in ("list",):
            return await self.list_rules.callback(self, ctx)
        if rule.lower() in ("roles",):
            return await self.roles.callback(self, ctx)
        # everything else must be a rule name/ID
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission to use this command.")

        # fetch rule
        sm = await self.get_shortname_mapping(ctx.guild)
        rid = sm.get(rule) if not rule.isdigit() else int(rule)
        if not rid:
            return await ctx.send("❌ Rule not found. Use `!automod list`.")
        try:
            rule_obj = await ctx.guild.fetch_automod_rule(rid)
        except Exception as e:
            return await ctx.send(f"❌ Could not fetch rule: {e}")

        # build info embed
        trig = rule_obj.trigger
        em = discord.Embed(title=f"AutoMod Rule: {rule_obj.name}", colour=await ctx.embed_colour())
        em.description = (
            f"**Type:** {trig.type.name}\n"
            f"**Enabled:** {rule_obj.enabled}\n"
            f"**Rule ID:** {rule_obj.id}"
        )
        # trigger metadata
        # keyword_filter
        if getattr(trig, "keyword_filter", None):
            em.add_field(
                name="Keyword Filter",
                value=", ".join(trig.keyword_filter),
                inline=False
            )
        # allow_list
        if getattr(trig, "allow_list", None):
            em.add_field(
                name="Allowed List",
                value=", ".join(trig.allow_list),
                inline=False
            )
        # regex_patterns
        if getattr(trig, "regex_patterns", None):
            em.add_field(
                name="Regex Patterns",
                value=", ".join(trig.regex_patterns),
                inline=False
            )
        # mention limit
        if getattr(trig, "mention_total_limit", None) is not None:
            em.add_field(
                name="Mention Limit",
                value=str(trig.mention_total_limit),
                inline=False
            )
        # actions
        acts = []
        for a in rule_obj.actions:
            line = f"- {a.type.name}"
            if getattr(a.metadata, "channel_id", None):
                line += f" → <#{a.metadata.channel_id}>"
            if getattr(a.metadata, "custom_message", None):
                line += f"\n  • Msg: {a.metadata.custom_message}"
            if getattr(a.metadata, "timeout_duration", None):
                line += f"\n  • Timeout: {a.metadata.timeout_duration}"
            acts.append(line)
        if acts:
            em.add_field(name="Actions", value="\n".join(acts), inline=False)

        # creator
        creator = rule_obj.creator.mention if rule_obj.creator else str(rule_obj.creator_id)
        em.add_field(name="Created by", value=creator, inline=False)

        # exemptions
        if rule_obj.exempt_roles:
            em.add_field(
                name="Exempt Roles",
                value="\n".join(r.mention for r in rule_obj.exempt_roles),
                inline=False
            )
        if rule_obj.exempt_channels:
            em.add_field(
                name="Exempt Channels",
                value="\n".join(c.mention for c in rule_obj.exempt_channels),
                inline=False
            )

        return await ctx.send(embed=em)

    @automod.command(name="list")
    async def list_rules(self, ctx):
        """List all AutoMod rules with their short names."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission to use this command.")
        try:
            rules = await ctx.guild.fetch_automod_rules()
        except Exception as e:
            return await ctx.send(f"❌ Failed to fetch rules: {e}")
        if not rules:
            return await ctx.send("No AutoMod rules found.")
        sm = await self.get_shortname_mapping(ctx.guild)
        lines = [
            f"**{short}** — {discord.utils.get(rules, id=rid).name} (`{rid}`)"
            for short, rid in sm.items()
            if discord.utils.get(rules, id=rid)
        ]
        await ctx.send("**AutoMod rules and short names:**\n" + "\n".join(lines))

    @automod.command(name="roles")
    async def roles(self, ctx):
        """List roles allowed to use these commands."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission to use this command.")
        allowed = await self.config.guild(ctx.guild).allowed_roles()
        if not allowed:
            return await ctx.send("No roles currently allowed.")
        mentions = [f"<@&{rid}>" for rid in allowed if ctx.guild.get_role(rid)]
        await ctx.send("Allowed roles: " + ", ".join(mentions))

    @automod.command(name="allowrole")
    @commands.has_guild_permissions(administrator=True)
    async def allowrole(self, ctx, role: discord.Role):
        """Give a role access to automod management."""
        guild_conf = self.config.guild(ctx.guild)
        allowed = await guild_conf.allowed_roles()
        if role.id in allowed:
            return await ctx.send(f"{role.mention} is already allowed.")
        allowed.append(role.id)
        await guild_conf.allowed_roles.set(allowed)
        await ctx.send(f"{role.mention} can now use automod commands.")

    @automod.command(name="removerole")
    @commands.has_guild_permissions(administrator=True)
    async def removerole(self, ctx, role: discord.Role):
        """Revoke a role’s access."""
        guild_conf = self.config.guild(ctx.guild)
        allowed = await guild_conf.allowed_roles()
        if role.id not in allowed:
            return await ctx.send(f"{role.mention} wasn’t allowed.")
        allowed.remove(role.id)
        await guild_conf.allowed_roles.set(allowed)
        await ctx.send(f"{role.mention} can no longer use automod commands.")

    @automod.command(name="enable")
    async def enable_rule(self, ctx, rule: str):
        """Enable a rule."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ No permission.")
        rid = (await self.get_shortname_mapping(ctx.guild)).get(rule) or int(rule)
        rule_obj = await ctx.guild.fetch_automod_rule(rid)
        await rule_obj.edit(enabled=True)
        await ctx.send(f"✅ Rule **{rule_obj.name}** is now **ENABLED**.")

    @automod.command(name="disable")
    async def disable_rule(self, ctx, rule: str):
        """Disable a rule."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ No permission.")
        rid = (await self.get_shortname_mapping(ctx.guild)).get(rule) or int(rule)
        rule_obj = await ctx.guild.fetch_automod_rule(rid)
        await rule_obj.edit(enabled=False)
        await ctx.send(f"❌ Rule **{rule_obj.name}** is now **DISABLED**.")

    @automod.command(name="add")
    async def add_words(self, ctx, rule: str, *, words: str):
        """Add to a rule’s allow list (keyword rules only)."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ No permission.")
        sm = await self.get_shortname_mapping(ctx.guild)
        rid = sm.get(rule) or int(rule)
        rule_obj = await ctx.guild.fetch_automod_rule(rid)
        if rule_obj.trigger.type is not discord.AutoModTriggerType.keyword:
            return await ctx.send("❌ Only keyword rules support an allow list.")
        # parse words
        new = {w.strip() for w in words.replace("\n", ",").split(",") if w.strip()}
        old = set(rule_obj.trigger.allow_list or [])
        updated = old | new
        # build metadata
        meta = AutoModTriggerMetadata(
            keyword_filter=rule_obj.trigger.keyword_filter,
            regex_patterns=rule_obj.trigger.regex_patterns,
            allow_list=list(updated)
        )
        await rule_obj.edit(trigger_metadata=meta)
        added = new - old
        await ctx.send(f"✅ Added: {', '.join(added) or '— none —'}\nCurrent list: {', '.join(sorted(updated)) or '— empty —'}")

    @automod.command(name="remove")
    async def remove_words(self, ctx, rule: str, *, words: str):
        """Remove from a rule’s allow list."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ No permission.")
        sm = await self.get_shortname_mapping(ctx.guild)
        rid = sm.get(rule) or int(rule)
        rule_obj = await ctx.guild.fetch_automod_rule(rid)
        if rule_obj.trigger.type is not discord.AutoModTriggerType.keyword:
            return await ctx.send("❌ Only keyword rules support an allow list.")
        to_rem = {w.strip() for w in words.replace("\n", ",").split(",") if w.strip()}
        old = set(rule_obj.trigger.allow_list or [])
        updated = old - to_rem
        meta = AutoModTriggerMetadata(
            keyword_filter=rule_obj.trigger.keyword_filter,
            regex_patterns=rule_obj.trigger.regex_patterns,
            allow_list=list(updated)
        )
        await rule_obj.edit(trigger_metadata=meta)
        removed = old & to_rem
        await ctx.send(f"✅ Removed: {', '.join(removed) or '— none —'}\nCurrent list: {', '.join(sorted(updated)) or '— empty —'}")

async def setup(bot):
    await bot.add_cog(D1AutoMod(bot))
