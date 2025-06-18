import discord
from redbot.core import commands, Config
from typing import Set, Optional

class D1AutoMod(commands.Cog):
    """AutoMod: Manage Discord AutoMod rules via simple commands."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=2468101214)
        self.config.register_guild(allowed_roles=[], shortnames={})

    async def has_automod_permission(self, ctx: commands.Context):
        if ctx.author.guild_permissions.administrator:
            return True
        allowed = await self.config.guild(ctx.guild).allowed_roles()
        return any(r.id in allowed for r in ctx.author.roles)

    async def get_shortname_mapping(self, guild: discord.Guild):
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
    async def automod(self, ctx: commands.Context,
                     rule: Optional[str] = None,
                     action: Optional[str] = None,
                     *, rest: Optional[str] = None):
        """
        Manage AutoMod rules.

        Usage:
          • !automod list
          • !automod roles
          • !automod allowrole <role>
          • !automod removerole <role>
          • !automod <rule>                   # show info
          • !automod <rule> enable            # enable rule
          • !automod <rule> disable           # disable rule
          • !automod <rule> add    w1,w2,w3   # add allowed words
          • !automod <rule> remove w1,w2      # remove allowed words
        """
        # no args → help
        if not rule:
            return await ctx.send_help()

        lower = rule.lower()
        # global subcommands
        if lower == "list":
            return await self._list(ctx)
        if lower == "roles":
            return await self._list_roles(ctx)

        # allowrole/removerole inherit different decorator, so show help
        if lower in ("allowrole", "removerole"):
            return await ctx.send_help()

        # from here on we need permission
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")

        # find the rule ID
        sm = await self.get_shortname_mapping(ctx.guild)
        try:
            rid = int(rule) if rule.isdigit() else sm[rule]
        except Exception:
            return await ctx.send("❌ Rule not found. Use `!automod list`.")

        # fetch the rule
        try:
            r = await ctx.guild.fetch_automod_rule(rid)
        except Exception as e:
            return await ctx.send(f"❌ Could not fetch rule: {e}")

        # enable / disable
        if action and action.lower() == "enable":
            await r.edit(enabled=True)
            return await ctx.send(f"✅ **{r.name}** enabled.")

        if action and action.lower() == "disable":
            await r.edit(enabled=False)
            return await ctx.send(f"❌ **{r.name}** disabled.")

        # add to allow list
        if action and action.lower() == "add":
            return await self._add_words(ctx, r, rest or "")

        # remove from allow list
        if action and action.lower() == "remove":
            return await self._remove_words(ctx, r, rest or "")

        # otherwise, just show info
        return await self._show_info(ctx, r)

    async def _show_info(self, ctx, r: discord.AutoModRule):
        trig = r.trigger
        em = discord.Embed(
            title=f"AutoMod Rule: {r.name}",
            colour=await ctx.embed_colour()
        )
        em.description = (
            f"**Type:** {trig.type.name}\n"
            f"**Enabled:** {r.enabled}\n"
            f"**Rule ID:** {r.id}"
        )
        # metadata fields
        if getattr(trig, "keyword_filter", None):
            em.add_field("Keyword Filter", ", ".join(trig.keyword_filter), inline=False)
        if getattr(trig, "allow_list", None):
            em.add_field("Allowed List", ", ".join(trig.allow_list), inline=False)
        if getattr(trig, "regex_patterns", None):
            em.add_field("Regex Patterns", ", ".join(trig.regex_patterns), inline=False)
        if getattr(trig, "mention_total_limit", None) is not None:
            em.add_field("Mention Limit", str(trig.mention_total_limit), inline=False)
        # actions
        lines = []
        for a in r.actions:
            line = f"- {a.type.name}"
            if getattr(a, "channel_id", None):
                line += f" → <#{a.channel_id}>"
            if getattr(a, "custom_message", None):
                line += f"\n  • Msg: {a.custom_message}"
            if getattr(a, "duration", None):
                line += f"\n  • Timeout: {a.duration}"
            lines.append(line)
        if lines:
            em.add_field("Actions", "\n".join(lines), inline=False)
        # creator & exemptions
        creator = r.creator.mention if r.creator else str(r.creator_id)
        em.add_field("Created by", creator, inline=False)
        if r.exempt_roles:
            em.add_field("Exempt Roles", "\n".join(x.mention for x in r.exempt_roles), inline=False)
        if r.exempt_channels:
            em.add_field("Exempt Channels", "\n".join(c.mention for c in r.exempt_channels), inline=False)
        return await ctx.send(embed=em)

    async def _list(self, ctx):
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")
        try:
            rules = await ctx.guild.fetch_automod_rules()
        except Exception as e:
            return await ctx.send(f"❌ Failed to fetch rules: {e}")
        if not rules:
            return await ctx.send("No AutoMod rules.")
        sm = await self.get_shortname_mapping(ctx.guild)
        lines = []
        for short, rid in sm.items():
            rule = discord.utils.get(rules, id=rid)
            if rule:
                lines.append(f"**{short}** — {rule.name} (`{rid}`)")
        return await ctx.send("**AutoMod rules:**\n" + "\n".join(lines))

    async def _list_roles(self, ctx):
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")
        allowed = await self.config.guild(ctx.guild).allowed_roles()
        if not allowed:
            return await ctx.send("No roles currently allowed.")
        mentions = [f"<@&{rid}>" for rid in allowed if ctx.guild.get_role(rid)]
        return await ctx.send("Allowed roles: " + ", ".join(mentions))

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

    async def _add_words(self, ctx, r: discord.AutoModRule, words: str):
        if r.trigger.type is not discord.AutoModTriggerType.keyword:
            return await ctx.send("❌ Only keyword rules support an allow list.")
        new: Set[str] = {w.strip() for w in words.replace("\n", ",").split(",") if w.strip()}
        old = set(r.trigger.allow_list or [])
        merged = old | new
        new_trigger = discord.AutoModTrigger(
            keyword_filter=r.trigger.keyword_filter,
            regex_patterns=r.trigger.regex_patterns,
            allow_list=list(merged),
            presets=getattr(r.trigger, "presets", None),
            mention_limit=getattr(r.trigger, "mention_total_limit", None)
        )
        await r.edit(trigger=new_trigger)
        added = new - old
        return await ctx.send(
            f"✅ Added: {', '.join(added) or '— none —'}\n"
            f"Current allow list: {', '.join(sorted(merged)) or '— empty —'}"
        )

    async def _remove_words(self, ctx, r: discord.AutoModRule, words: str):
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
        return await ctx.send(
            f"✅ Removed: {', '.join(removed) or '— none —'}\n"
            f"Current allow list: {', '.join(sorted(merged)) or '— empty —'}"
        )

async def setup(bot):
    await bot.add_cog(D1AutoMod(bot))
