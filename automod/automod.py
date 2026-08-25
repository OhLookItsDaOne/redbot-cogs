import discord
import re
from redbot.core import commands, Config, app_commands
from typing import Set

class AutoMod(commands.Cog):
    """AutoMod: Manage Discord AutoMod rules via simple commands."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=2468101214)
        self.config.register_guild(allowed_roles=[], shortnames={}, default_rule=None)

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

    async def resolve_effective_rule(self, ctx, rule: str = None):
        """Resolve the given rule, or fall back to the configured default rule."""
        if rule:
            r, err = await self.resolve_rule(ctx, rule)
            return r, err, rule
        default = await self.config.guild(ctx.guild).default_rule()
        if not default:
            return None, None, None
        r, err = await self.resolve_rule(ctx, default)
        return r, err, default

    @staticmethod
    def _bare(entry: str) -> str:
        """Return a bare domain form (no scheme, www, port) for comparison.

        ``*https://www.dlink.com*`` and ``dlink.com`` both become ``dlink.com``.
        """
        e = entry.strip().lower()
        e = e.strip("*")
        e = re.sub(r"^https?://", "", e)
        e = re.sub(r"^www\.", "", e)
        e = e.rstrip("/")
        e = re.sub(r":\d+$", "", e)
        return e

    @staticmethod
    def _normalize_entry(entry: str) -> str:
        """Normalize a whitelist entry to the AutoMod wildcard format.

        ``https://www.bestbuy.ca`` becomes ``*bestbuy.ca*`` so it matches any
        message containing the domain. Scheme, port and ``www.`` are stripped,
        existing wildcards are preserved.
        """
        e = entry.strip().lower()
        if not e:
            return ""
        if "*" not in e:
            # strip scheme, www., trailing slash and port
            e = re.sub(r"^https?://", "", e)
            e = re.sub(r"^www\.", "", e)
            e = e.rstrip("/")
            e = re.sub(r":\d+$", "", e)
            if e:
                e = f"*{e}*"
        return e

    async def _get_allow_list(self, r) -> set:
        old = getattr(r.trigger, "allow_list", None)
        if old is None:
            return None
        return set(old)

    async def _set_allow_list(self, ctx, r, entries: set):
        kt = r.trigger.keyword_filter if hasattr(r.trigger, "keyword_filter") else None
        rp = r.trigger.regex_patterns if hasattr(r.trigger, "regex_patterns") else None
        await r.edit(
            trigger=discord.AutoModTrigger(
                keyword_filter=kt,
                allow_list=sorted(entries),
                regex_patterns=rp
            )
        )

    @commands.hybrid_group(name="siteallow", extras={"red_force_enable": True})
    @commands.guild_only()
    @app_commands.default_permissions(administrator=True)
    async def siteallow(self, ctx):
        """Manage which sites are allowed by a rule."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help(ctx.command)

    @siteallow.command(name="default")
    async def siteallow_default(self, ctx, rule: str = None):
        """Set the default rule used when none is given."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")
        if rule is None:
            default = await self.config.guild(ctx.guild).default_rule()
            if default:
                return await ctx.send(f"Default rule is currently: `{default}`")
            return await ctx.send("No default rule is set.")
        r, err = await self.resolve_rule(ctx, rule)
        if err:
            return await ctx.send(f"❌ Could not fetch rule: {err}")
        if not r:
            return await ctx.send("❌ Rule not found. Use `/siteallow list`.")
        await self.config.guild(ctx.guild).default_rule.set(rule)
        return await ctx.send(f"✅ Default rule set to `{rule}`.")

    @siteallow.command(name="add")
    async def siteallow_add(self, ctx, domains: str, rule: str = None):
        """Add domains to a rule's allow list (comma separated)."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")
        r, err, used = await self.resolve_effective_rule(ctx, rule)
        if err:
            return await ctx.send(f"❌ Could not fetch rule: {err}")
        if not r:
            return await ctx.send("❌ No rule given and no default rule set. Use `/siteallow default`.")
        old = await self._get_allow_list(r)
        if old is None:
            return await ctx.send("❌ Only keyword‑style rules support an allow list.")
        new = {self._normalize_entry(w) for w in domains.replace("\n", ",").split(",") if self._normalize_entry(w)}
        merged = old | new
        await self._set_allow_list(ctx, r, merged)
        added = sorted(new - old)
        return await ctx.send(
            f"✅ Rule `{used}` — Added: {', '.join(added) or '— none —'}\n"
            f"Current allow list: {', '.join(sorted(merged)) or '— empty —'}"
        )

    @siteallow.command(name="remove")
    async def siteallow_remove(self, ctx, domains: str, rule: str = None):
        """Remove domains from a rule's allow list (comma separated)."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")
        r, err, used = await self.resolve_effective_rule(ctx, rule)
        if err:
            return await ctx.send(f"❌ Could not fetch rule: {err}")
        if not r:
            return await ctx.send("❌ No rule given and no default rule set. Use `/siteallow default`.")
        old = await self._get_allow_list(r)
        if old is None:
            return await ctx.send("❌ Only keyword‑style rules support an allow list.")
        # Build bare forms of requested domains for robust matching
        wanted = {self._bare(w) for w in domains.replace("\n", ",").split(",") if self._bare(w)}
        # Remove any entry whose bare form matches a requested domain
        kept = set()
        gone = []
        for entry in old:
            if self._bare(entry) in wanted:
                gone.append(entry)
            else:
                kept.add(entry)
        await self._set_allow_list(ctx, r, kept)
        return await ctx.send(
            f"✅ Rule `{used}` — Removed: {', '.join(sorted(gone)) or '— none —'}\n"
            f"Current allow list: {', '.join(sorted(kept)) or '— empty —'}"
        )

    @siteallow.command(name="edit")
    async def siteallow_edit(self, ctx, old: str, new: str, rule: str = None):
        """Replace one domain with another in a rule's allow list."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")
        r, err, used = await self.resolve_effective_rule(ctx, rule)
        if err:
            return await ctx.send(f"❌ Could not fetch rule: {err}")
        if not r:
            return await ctx.send("❌ No rule given and no default rule set. Use `/siteallow default`.")
        current = await self._get_allow_list(r)
        if current is None:
            return await ctx.send("❌ Only keyword‑style rules support an allow list.")
        old_n = self._normalize_entry(old)
        new_n = self._normalize_entry(new)
        if old_n not in current:
            return await ctx.send(f"❌ `{old_n}` is not in the allow list.")
        current.remove(old_n)
        current.add(new_n)
        await self._set_allow_list(ctx, r, current)
        return await ctx.send(
            f"✅ Rule `{used}` — Replaced `{old_n}` with `{new_n}`.\n"
            f"Current allow list: {', '.join(sorted(current)) or '— empty —'}"
        )

    @siteallow.command(name="list")
    async def siteallow_list(self, ctx, rule: str = None):
        """Show allow lists (all rules, or a specific rule)."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("❌ You do not have permission.")
        if rule:
            r, err = await self.resolve_rule(ctx, rule)
            if err:
                return await ctx.send(f"❌ Could not fetch rule: {err}")
            if not r:
                return await ctx.send("❌ Rule not found. Use `/siteallow list`.")
            current = await self._get_allow_list(r)
            if current is None:
                return await ctx.send("❌ Only keyword‑style rules support an allow list.")
            return await ctx.send(
                f"**Allow list for `{r.name}`:**\n" +
                ("\n".join(f"• `{e}`" for e in sorted(current)) if current else "— empty —")
            )

        # No rule: show all rule names and their shortnames
        sm = await self.get_shortname_mapping(ctx.guild)
        try:
            rules = await ctx.guild.fetch_automod_rules()
        except Exception as e:
            return await ctx.send(f"❌ Failed to fetch rules: {e}")
        default = await self.config.guild(ctx.guild).default_rule()
        lines = []
        for short, rid in sm.items():
            rule_obj = discord.utils.get(rules, id=rid)
            if rule_obj:
                marker = "⭐" if short == default else ""
                lines.append(f"{marker} **{short}** — {rule_obj.name} (`{rid}`)")
        if default:
            lines.append(f"\nDefault rule: `{default}` (⭐)")
        else:
            lines.append("\nNo default rule set. Use `/siteallow default <rule>`.")
        return await ctx.send("**AutoMod rules:**\n" + "\n".join(lines))

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
    await bot.add_cog(AutoMod(bot))
