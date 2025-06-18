import discord
from redbot.core import commands, Config
from typing import Set

class D1AutoMod(commands.Cog):
    """AutoMod: Manage Discord AutoMod rules via simple commands."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=2468101214)
        self.config.register_guild(allowed_roles=[], shortnames={})

    async def has_automod_permission(self, ctx):
        if ctx.author.guild_permissions.administrator:
            return True
        allowed = await self.config.guild(ctx.guild).allowed_roles()
        return any(r.id in allowed for r in ctx.author.roles)

    async def get_shortname_mapping(self, guild: discord.Guild):
        """Build short-name → rule ID map, store in config."""
        seen = set()
        mapping = {}
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
            while short in seen:
                short += str(rule.id)[-1]
            seen.add(short)
            mapping[short] = rule.id
        await self.config.guild(guild).shortnames.set(mapping)
        return mapping

    @commands.group(name="automod", invoke_without_command=True)
    @commands.guild_only()
    async def automod(self, ctx, rule: str = None):
        """
        Show info about an AutoMod rule or subcommand.

        Usage:
          • !automod list
          • !automod roles
          • !automod allowrole <role>
          • !automod removerole <role>
          • !automod <rule>                     # info
          • !automod <rule> enable|disable      # toggle
          • !automod <rule> add  w1,w2,w3       # add to allow list
          • !automod <rule> remove  w1,w2       # remove from allow list
        """
        if not rule:
            return await ctx.send_help()

        cmd = rule.lower()
        # dispatch subcommands
        if cmd == "list":
            return await self.list_rules(ctx)
        if cmd == "roles":
            return await self.roles(ctx)
        # next ones require admin perms
        if cmd == "allowrole" or cmd == "removerole":
            return await ctx.send_help()

        # everything else is a rule name or ID
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")

        sm = await self.get_shortname_mapping(ctx.guild)
        rid = sm.get(rule) if not rule.isdigit() else int(rule)
        if not rid:
            return await ctx.send("❌ Rule not found. Use `!automod list`.")
        try:
            rule_obj = await ctx.guild.fetch_automod_rule(rid)
        except Exception as e:
            return await ctx.send(f"❌ Could not fetch rule: {e}")

        # render info embed
        trig = rule_obj.trigger
        em = discord.Embed(
            title=f"AutoMod Rule: {rule_obj.name}",
            colour=await ctx.embed_colour()
        )
        em.description = (
            f"**Type:** {trig.type.name}\n"
            f"**Enabled:** {rule_obj.enabled}\n"
            f"**Rule ID:** {rule_obj.id}"
        )
        # metadata fields
        if getattr(trig, "keyword_filter", None):
            em.add_field(
                name="Keyword Filter",
                value=", ".join(trig.keyword_filter), inline=False
            )
        if getattr(trig, "allow_list", None):
            em.add_field(
                name="Allowed List",
                value=", ".join(trig.allow_list), inline=False
            )
        if getattr(trig, "regex_patterns", None):
            em.add_field(
                name="Regex Patterns",
                value=", ".join(trig.regex_patterns), inline=False
            )
        if getattr(trig, "mention_total_limit", None) is not None:
            em.add_field(
                name="Mention Limit",
                value=str(trig.mention_total_limit), inline=False
            )
        # actions
        lines = []
        for a in rule_obj.actions:
            part = f"- {a.type.name}"
            if getattr(a, "channel_id", None):
                part += f" → <#{a.channel_id}>"
            if getattr(a, "custom_message", None):
                part += f"\n  • Msg: {a.custom_message}"
            if getattr(a, "duration", None):
                part += f"\n  • Timeout: {a.duration}"
            lines.append(part)
        if lines:
            em.add_field(name="Actions", value="\n".join(lines), inline=False)

        # creator + exemptions
        creator = rule_obj.creator.mention if rule_obj.creator else str(rule_obj.creator_id)
        em.add_field(name="Created by", value=creator, inline=False)

        if rule_obj.exempt_roles:
            em.add_field(
                name="Exempt Roles",
                value="\n".join(r.mention for r in rule_obj.exempt_roles), inline=False
            )
        if rule_obj.exempt_channels:
            em.add_field(
                name="Exempt Channels",
                value="\n".join(c.mention for c in rule_obj.exempt_channels), inline=False
            )

        return await ctx.send(embed=em)

    @automod.command(name="list")
    async def list_rules(self, ctx):
        """List AutoMod rules + their short names."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")
        try:
            rules = await ctx.guild.fetch_automod_rules()
        except Exception as e:
            return await ctx.send(f"❌ Failed to fetch rules: {e}")
        if not rules:
            return await ctx.send("No AutoMod rules.")
        sm = await self.get_shortname_mapping(ctx.guild)
        lines = [
            f"**{short}** — {discord.utils.get(rules, id=rid).name} (`{rid}`)"
            for short, rid in sm.items()
            if discord.utils.get(rules, id=rid)
        ]
        await ctx.send("**AutoMod rules:**\n" + "\n".join(lines))

    @automod.command(name="roles")
    async def roles(self, ctx):
        """List roles allowed to manage automod."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")
        allowed = await self.config.guild(ctx.guild).allowed_roles()
        if not allowed:
            return await ctx.send("No roles currently allowed.")
        mentions = [f"<@&{rid}>" for rid in allowed if ctx.guild.get_role(rid)]
        await ctx.send("Allowed roles: " + ", ".join(mentions))

    @automod.command(name="allowrole")
    @commands.has_guild_permissions(administrator=True)
    async def allowrole(self, ctx, role: discord.Role):
        """Grant a role automod access."""
        g = self.config.guild(ctx.guild)
        lst = await g.allowed_roles()
        if role.id in lst:
            return await ctx.send("That role is already allowed.")
        lst.append(role.id)
        await g.allowed_roles.set(lst)
        await ctx.send(f"{role.mention} can now use automod commands.")

    @automod.command(name="removerole")
    @commands.has_guild_permissions(administrator=True)
    async def removerole(self, ctx, role: discord.Role):
        """Revoke a role’s automod access."""
        g = self.config.guild(ctx.guild)
        lst = await g.allowed_roles()
        if role.id not in lst:
            return await ctx.send("That role wasn’t allowed.")
        lst.remove(role.id)
        await g.allowed_roles.set(lst)
        await ctx.send(f"{role.mention} can no longer use automod commands.")

    @automod.command(name="enable")
    async def enable_rule(self, ctx, rule: str):
        """Enable a rule."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")
        rid = (await self.get_shortname_mapping(ctx.guild)).get(rule) or int(rule)
        r = await ctx.guild.fetch_automod_rule(rid)
        await r.edit(enabled=True)
        await ctx.send(f"✅ **{r.name}** enabled.")

    @automod.command(name="disable")
    async def disable_rule(self, ctx, rule: str):
        """Disable a rule."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")
        rid = (await self.get_shortname_mapping(ctx.guild)).get(rule) or int(rule)
        r = await ctx.guild.fetch_automod_rule(rid)
        await r.edit(enabled=False)
        await ctx.send(f"❌ **{r.name}** disabled.")

    @automod.command(name="add")
    async def add_words(self, ctx, rule: str, *, words: str):
        """Add words to a keyword rule’s allow list."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")
        sm = await self.get_shortname_mapping(ctx.guild)
        rid = sm.get(rule) or int(rule)
        r = await ctx.guild.fetch_automod_rule(rid)
        if r.trigger.type is not discord.AutoModTriggerType.keyword:
            return await ctx.send("❌ Only keyword rules support an allow list.")
        new: Set[str] = {w.strip() for w in words.replace("\n", ",").split(",") if w.strip()}
        old = set(r.trigger.allow_list or [])
        merged = old | new
        # build a fresh trigger with updated allow_list
        new_trigger = discord.AutoModTrigger(
            keyword_filter=r.trigger.keyword_filter,
            regex_patterns=r.trigger.regex_patterns,
            allow_list=list(merged),
            presets=getattr(r.trigger, "presets", None),
            mention_limit=getattr(r.trigger, "mention_total_limit", None)
        )
        await r.edit(trigger=new_trigger)
        added = new - old
        await ctx.send(
            f"✅ Added: {', '.join(added) or '— none —'}\n"
            f"Current allow list: {', '.join(sorted(merged)) or '— empty —'}"
        )

    @automod.command(name="remove")
    async def remove_words(self, ctx, rule: str, *, words: str):
        """Remove words from a keyword rule’s allow list."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")
        sm = await self.get_shortname_mapping(ctx.guild)
        rid = sm.get(rule) or int(rule)
        r = await ctx.guild.fetch_automod_rule(rid)
        if r.trigger.type is not discord.AutoModTriggerType.keyword:
            return await ctx.send("❌ Only keyword rules support an allow list.")
        to_rem = {w.strip() for w in words.replace("\n", ",").split(",") if w.strip()}
        old = set(r.trigger.allow_list or [])
        merged = old - to_rem
        new_trigger = discord.AutoModTrigger(
            keyword_filter=r.trigger.keyword_filter,
            regex_patterns=r.trigger.regex_patterns,
            allow_list=list(merged),
            presets=getattr(r.trigger, "presets", None),
            mention_limit=getattr(r.trigger, "mention_total_limit", None)
        )
        await r.edit(trigger=new_trigger)
        removed = old & to_rem
        await ctx.send(
            f"✅ Removed: {', '.join(removed) or '— none —'}\n"
            f"Current allow list: {', '.join(sorted(merged)) or '— empty —'}"
        )

async def setup(bot):
    await bot.add_cog(D1AutoMod(bot))
