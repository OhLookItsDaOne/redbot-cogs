import discord
from redbot.core import commands, Config, app_commands
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

    async def resolve_rule(self, ctx, rule: str):
        """Resolve a shortname or rule ID to an AutoMod rule."""
        sm = await self.get_shortname_mapping(ctx.guild)
        rid = sm.get(rule) if not rule.isdigit() else int(rule)
        if not rid:
            return None, None
        try:
            return await ctx.guild.fetch_automod_rule(rid), None
        except Exception as e:
            return None, e

    @commands.hybrid_group(name="automod", extras={"red_force_enable": True})
    @commands.guild_only()
    @app_commands.default_permissions(administrator=True)
    async def automod(self, ctx):
        """Manage Discord AutoMod rules."""
        pass

    @automod.command(name="list")
    async def list_rules(self, ctx):
        """List all AutoMod rules."""
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
        return await ctx.send("**AutoMod rules:**\n" + "\n".join(lines))

    @automod.command(name="roles")
    async def list_roles(self, ctx):
        """List roles allowed to manage AutoMod rules."""
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
        """Grant a role access to automod commands."""
        lst = await self.config.guild(ctx.guild).allowed_roles()
        if role.id in lst:
            return await ctx.send("❌ That role is already allowed.")
        lst.append(role.id)
        await self.config.guild(ctx.guild).allowed_roles.set(lst)
        await ctx.send(f"✅ {role.mention} can now use automod commands.")

    @automod.command(name="removerole")
    @commands.has_guild_permissions(administrator=True)
    async def removerole(self, ctx, role: discord.Role):
        """Revoke a role's access to automod commands."""
        lst = await self.config.guild(ctx.guild).allowed_roles()
        if role.id not in lst:
            return await ctx.send("❌ That role wasn't allowed.")
        lst.remove(role.id)
        await self.config.guild(ctx.guild).allowed_roles.set(lst)
        await ctx.send(f"❌ {role.mention} can no longer use automod commands.")

    @automod.command(name="info")
    async def info(self, ctx, rule: str):
        """Show information about an AutoMod rule."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")
        r, err = await self.resolve_rule(ctx, rule)
        if err:
            return await ctx.send(f"❌ Could not fetch rule: {err}")
        if not r:
            return await ctx.send("❌ Rule not found. Use `/automod list`.")
        await self._show_info(ctx, r)

    @automod.command(name="enable")
    async def enable(self, ctx, rule: str):
        """Enable an AutoMod rule."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")
        r, err = await self.resolve_rule(ctx, rule)
        if err:
            return await ctx.send(f"❌ Could not fetch rule: {err}")
        if not r:
            return await ctx.send("❌ Rule not found. Use `/automod list`.")
        await r.edit(enabled=True)
        return await ctx.send(f"✅ **{r.name}** enabled.")

    @automod.command(name="disable")
    async def disable(self, ctx, rule: str):
        """Disable an AutoMod rule."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")
        r, err = await self.resolve_rule(ctx, rule)
        if err:
            return await ctx.send(f"❌ Could not fetch rule: {err}")
        if not r:
            return await ctx.send("❌ Rule not found. Use `/automod list`.")
        await r.edit(enabled=False)
        return await ctx.send(f"❌ **{r.name}** disabled.")

    @automod.command(name="add")
    async def add_words(self, ctx, rule: str, words: str):
        """Add words to an AutoMod rule's allow list (comma separated)."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")
        r, err = await self.resolve_rule(ctx, rule)
        if err:
            return await ctx.send(f"❌ Could not fetch rule: {err}")
        if not r:
            return await ctx.send("❌ Rule not found. Use `/automod list`.")
        await self._add_words(ctx, r, words)

    @automod.command(name="remove")
    async def remove_words(self, ctx, rule: str, words: str):
        """Remove words from an AutoMod rule's allow list (comma separated)."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")
        r, err = await self.resolve_rule(ctx, rule)
        if err:
            return await ctx.send(f"❌ Could not fetch rule: {err}")
        if not r:
            return await ctx.send("❌ Rule not found. Use `/automod list`.")
        await self._remove_words(ctx, r, words)

    async def _show_info(self, ctx, r):
        trig = r.trigger
        em = discord.Embed(
            title=f"AutoMod Rule: {r.name}",
            colour=await ctx.embed_colour()
        )
        em.description = (
            f"**Type:** `{getattr(trig, 'type', 'unknown')}`\n"
            f"**Enabled:** {r.enabled}\n"
            f"**Rule ID:** {r.id}"
        )
        # metadata
        if getattr(trig, "keyword_filter", None):
            em.add_field(name="Keyword Filter", value=", ".join(trig.keyword_filter), inline=False)
        if getattr(trig, "allow_list", None):
            em.add_field(name="Allowed List", value=", ".join(trig.allow_list), inline=False)
        if getattr(trig, "regex_patterns", None):
            em.add_field(name="Regex Patterns", value=", ".join(trig.regex_patterns), inline=False)
        if getattr(trig, "mention_total_limit", None) is not None:
            em.add_field(name="Mention Limit", value=str(trig.mention_total_limit), inline=False)
        # actions
        parts = []
        for a in r.actions:
            line = f"- {a.type.name}"
            if getattr(a, "channel_id", None):
                line += f" → <#{a.channel_id}>"
            if getattr(a, "custom_message", None):
                line += f"\n  • Msg: {a.custom_message}"
            if getattr(a, "duration", None):
                line += f"\n  • Timeout: {a.duration}"
            parts.append(line)
        if parts:
            em.add_field(name="Actions", value="\n".join(parts), inline=False)
        # creator & exemptions
        creator = r.creator.mention if r.creator else str(r.creator_id)
        em.add_field(name="Created by", value=creator, inline=False)
        if r.exempt_roles:
            em.add_field(
                name="Exempt Roles",
                value="\n".join(role.mention for role in r.exempt_roles),
                inline=False
            )
        if r.exempt_channels:
            em.add_field(
                name="Exempt Channels",
                value="\n".join(chan.mention for chan in r.exempt_channels),
                inline=False
            )
        return await ctx.send(embed=em)

    async def _add_words(self, ctx, r, words: str):
        old = set(getattr(r.trigger, "allow_list", []))
        if old is None:
            return await ctx.send("❌ Only keyword‑style rules support an allow list.")
        new = {w.strip() for w in words.replace("\n", ",").split(",") if w.strip()}
        merged = old | new

        # rebuild just the keyword fields
        kt = r.trigger.keyword_filter if hasattr(r.trigger, "keyword_filter") else None
        rp = r.trigger.regex_patterns if hasattr(r.trigger, "regex_patterns") else None
        await r.edit(
            trigger=discord.AutoModTrigger(
                keyword_filter=kt,
                allow_list=list(merged),
                regex_patterns=rp
            )
        )
        added = sorted(new - old)
        return await ctx.send(
            f"✅ Added: {', '.join(added) or '— none —'}\n"
            f"Current allow list: {', '.join(sorted(merged)) or '— empty —'}"
        )

    async def _remove_words(self, ctx, r, words: str):
        old = set(getattr(r.trigger, "allow_list", []))
        if old is None:
            return await ctx.send("❌ Only keyword‑style rules support an allow list.")
        rem = {w.strip() for w in words.replace("\n", ",").split(",") if w.strip()}
        kept = old - rem

        kt = r.trigger.keyword_filter if hasattr(r.trigger, "keyword_filter") else None
        rp = r.trigger.regex_patterns if hasattr(r.trigger, "regex_patterns") else None
        await r.edit(
            trigger=discord.AutoModTrigger(
                keyword_filter=kt,
                allow_list=list(kept),
                regex_patterns=rp
            )
        )
        gone = sorted(old & rem)
        return await ctx.send(
            f"✅ Removed: {', '.join(gone) or '— none —'}\n"
            f"Current allow list: {', '.join(sorted(kept)) or '— empty —'}"
        )

async def setup(bot):
    await bot.add_cog(D1AutoMod(bot))
