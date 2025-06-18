import discord
from redbot.core import commands, Config

class D1AutoMod(commands.Cog):
    """AutoMod: Fully permission-protected, full info, classic commands only."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=2468101214)
        default = {"allowed_roles": [], "shortnames": {}}
        self.config.register_guild(**default)

    async def has_automod_permission(self, ctx):
        # Admins or roles in our whitelist
        if ctx.author.guild_permissions.administrator:
            return True
        allowed = await self.config.guild(ctx.guild).allowed_roles()
        return any(r.id in allowed for r in ctx.author.roles)

    async def get_shortname_mapping(self, guild):
        # Build shortname→ID map
        mapping = await self.config.guild(guild).shortnames()
        try:
            rules = await guild.fetch_automod_rules()
        except:
            return {}
        used = set()
        newmap = {}
        for r in rules:
            parts = r.name.lower().split()
            if not parts:
                continue
            short = parts[0]
            if short in used:
                short = parts[0] + (parts[1] if len(parts) > 1 else str(r.id)[-3:])
            while short in used:
                short += str(r.id)[-1]
            used.add(short)
            newmap[short] = r.id
        await self.config.guild(guild).shortnames.set(newmap)
        return newmap

    def format_rule_embed(self, ctx, rule):
        """Build an embed with every piece of info (keywords, allow list, regex, etc)."""
        trig: discord.AutoModTrigger = rule.trigger
        ttype = str(trig.type)
        embed = discord.Embed(
            title=f"AutoMod Rule: {rule.name}",
            description=(
                f"Type: `{ttype}`\n"
                f"Enabled: {rule.enabled}\n"
                f"Rule ID: {rule.id}"
            ),
        )

        # keyword_filter
        if getattr(trig, "keyword_filter", None):
            embed.add_field(
                name="Keyword Filter",
                value=", ".join(trig.keyword_filter),
                inline=False
            )
        # allow_list (always show, even if empty)
        al = getattr(trig, "allow_list", [])
        embed.add_field(
            name="Allowed List",
            value=", ".join(al) if al else "*None*",
            inline=False
        )
        # regex_patterns
        if getattr(trig, "regex_patterns", None):
            embed.add_field(
                name="Regex Patterns",
                value=", ".join(trig.regex_patterns),
                inline=False
            )
        # mention limit
        mlim = getattr(trig, "mention_total_limit", None)
        if mlim is not None:
            embed.add_field(name="Mention Limit", value=str(mlim), inline=False)

        # Actions
        lines = []
        for act in rule.actions:
            ct = act.type
            chan = getattr(act, "channel_id", None) or getattr(act, "channel", None)
            # unify channel mention
            chs = f"<#{chan.id}>" if hasattr(chan, "id") else (f"<#{chan}>" if chan else "None")
            dur = getattr(act, "duration", None)
            # custom message may live in metadata
            cm = getattr(act, "custom_message", None)
            md = getattr(act, "metadata", {})
            cm = cm or md.get("custom_message")
            part = f"Type: {ct} Channel: {chs} Duration: {dur or 'None'}"
            if cm:
                part += f"\nCustom Message: {cm}"
            lines.append(part)
        if lines:
            embed.add_field(name="Actions", value="\n".join(lines), inline=False)

        # Creator
        cr = rule.creator
        embed.add_field(
            name="Created by",
            value=f"<@{cr.id}>" if cr else "Unknown",
            inline=False
        )

        # Exempt roles
        ers = getattr(rule, "exempt_roles", [])
        if not ers:
            # fallback to IDs
            ers = [f"<@&{rid}>" for rid in getattr(rule, "exempt_role_ids", [])]
        embed.add_field(
            name="Exempt Roles",
            value="\n".join(r.mention for r in ers) if ers else "*None*",
            inline=False
        )

        # Exempt channels
        ecs = getattr(rule, "exempt_channels", [])
        if not ecs:
            ecs = [f"<#{cid}>" for cid in getattr(rule, "exempt_channel_ids", [])]
        embed.add_field(
            name="Exempt Channels",
            value="\n".join(c.mention for c in ecs) if ecs else "*None*",
            inline=False
        )

        return embed

    def is_keyword_rule(self, rule):
        t = rule.trigger.type
        return str(t).lower().endswith(".keyword")

    @commands.group(invoke_without_command=True)
    @commands.guild_only()
    async def automod(self, ctx, short: str = None, action: str = None, *rest):
        """
        Show/edit AutoMod rules.
        • !automod <short>                → info
        • !automod <short> enable/disable
        • !automod <short> allow add/remove word1,word2
        • !automod list
        """
        if not await self.has_automod_permission(ctx):
            return await ctx.send("You do not have permission.")

        if not short:
            return await ctx.send_help()

        # find the rule
        sm = await self.get_shortname_mapping(ctx.guild)
        rid = sm.get(short) if not short.isdigit() else int(short)
        if not rid:
            return await ctx.send("Rule not found. Use `!automod list`.")

        try:
            rule = await ctx.guild.fetch_automod_rule(rid)
        except Exception as e:
            return await ctx.send(f"Could not fetch rule: {e}")

        # handle sub-actions
        if action:
            a = action.lower()

            # enable / disable
            if a in ("enable", "on"):
                await rule.edit(enabled=True, reason=f"Enabled by {ctx.author}")
                return await ctx.send(f"Rule **{rule.name}** is now **ENABLED**.")
            if a in ("disable", "off"):
                await rule.edit(enabled=False, reason=f"Disabled by {ctx.author}")
                return await ctx.send(f"Rule **{rule.name}** is now **DISABLED**.")

            # allow-list
            if a == "allow":
                if not self.is_keyword_rule(rule):
                    return await ctx.send("This rule does not support allowed words/phrases.")
                if len(rest) < 2:
                    return await ctx.send("Usage: `!automod <short> allow add/remove word1,word2`")
                mode = rest[0].lower()
                words = [w.strip() for w in " ".join(rest[1:]).replace("\n", ",").split(",") if w.strip()]

                trig = rule.trigger
                old = set(trig.allow_list or [])
                new = set(old)

                if mode == "add":
                    new.update(words)
                elif mode == "remove":
                    new.difference_update(words)
                else:
                    return await ctx.send("Usage: `allow add` or `allow remove`")

                # rebuild trigger
                new_trigger = discord.AutoModTrigger(
                    keyword_filter=trig.keyword_filter,
                    allow_list=list(new),
                    regex_patterns=getattr(trig, "regex_patterns", None),
                    mention_limit=getattr(trig, "mention_total_limit", None),
                    presets=getattr(trig, "presets", None),
                )
                try:
                    await rule.edit(trigger=new_trigger, reason=f"Edited by {ctx.author}")
                except Exception as e:
                    return await ctx.send(f"Failed to update allow list: {e}")

                added = new - old
                removed = old - new
                if mode == "add":
                    return await ctx.send(f"Added: {', '.join(added) or '*nothing new*'}")
                else:
                    return await ctx.send(f"Removed: {', '.join(removed) or '*nothing removed*'}")

            return await ctx.send("Unknown action. Try `enable`, `disable`, `allow add/remove`.")

        # no sub-action → just show info
        embed = self.format_rule_embed(ctx, rule)
        await ctx.send(embed=embed)

    @automod.command(name="list")
    async def list_rules(self, ctx):
        if not await self.has_automod_permission(ctx):
            return await ctx.send("You do not have permission.")
        try:
            rules = await ctx.guild.fetch_automod_rules()
        except Exception as e:
            return await ctx.send(f"Failed to fetch rules: {e}")
        if not rules:
            return await ctx.send("No rules set up.")
        sm = await self.get_shortname_mapping(ctx.guild)
        lines = []
        for sn, rid in sm.items():
            r = discord.utils.get(rules, id=rid)
            if r:
                lines.append(f"**{sn}** — {r.name} (ID: `{r.id}`)")
        await ctx.send("**AutoMod rules and short names:**\n" + "\n".join(lines))

    @automod.command(name="allowrole")
    @commands.has_guild_permissions(administrator=True)
    async def allowrole(self, ctx, role: discord.Role):
        # only admins can add/remove roles
        if not await self.has_automod_permission(ctx):
            return await ctx.send("You do not have permission.")
        allowed = await self.config.guild(ctx.guild).allowed_roles()
        if role.id in allowed:
            return await ctx.send(f"{role.mention} is already allowed.")
        allowed.append(role.id)
        await self.config.guild(ctx.guild).allowed_roles.set(allowed)
        await ctx.send(f"{role.mention} may now use automod commands.")

    @automod.command(name="removerole")
    @commands.has_guild_permissions(administrator=True)
    async def removerole(self, ctx, role: discord.Role):
        if not await self.has_automod_permission(ctx):
            return await ctx.send("You do not have permission.")
        allowed = await self.config.guild(ctx.guild).allowed_roles()
        if role.id not in allowed:
            return await ctx.send(f"{role.mention} is not in the allowed list.")
        allowed.remove(role.id)
        await self.config.guild(ctx.guild).allowed_roles.set(allowed)
        await ctx.send(f"{role.mention} may NO LONGER use automod commands.")

    @automod.command(name="roles")
    async def roles(self, ctx):
        if not await self.has_automod_permission(ctx):
            return await ctx.send("You do not have permission.")
        allowed = await self.config.guild(ctx.guild).allowed_roles()
        if not allowed:
            return await ctx.send("No roles are currently allowed.")
        mentions = [f"<@&{rid}>" for rid in allowed if ctx.guild.get_role(rid)]
        await ctx.send("Allowed roles:\n" + ("\n".join(mentions) if mentions else "*none valid*"))

async def setup(bot):
    await bot.add_cog(D1AutoMod(bot))
