import discord
from redbot.core import commands, Config

class D1AutoMod(commands.Cog):
    """Discord AutoMod management (keyword rules, actions, allow-lists etc)."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=2468101214)
        default_guild = {"allowed_roles": [], "shortnames": {}}
        self.config.register_guild(**default_guild)

    # ------- Permissions -------
    async def has_automod_permission(self, ctx):
        if ctx.author.guild_permissions.administrator:
            return True
        allowed_roles = await self.config.guild(ctx.guild).allowed_roles()
        return any(role.id in allowed_roles for role in ctx.author.roles)

    def _require_perm(self):
        async def predicate(ctx):
            return await self.has_automod_permission(ctx)
        return commands.check(predicate)

    # ------- Shortnames -------
    async def get_shortname_mapping(self, guild):
        mapping = await self.config.guild(guild).shortnames()
        try:
            rules = await guild.fetch_automod_rules()
        except Exception:
            return {}
        names_used, newmap = set(), {}
        for rule in rules:
            words = rule.name.lower().split()
            short = (words[0] if words else str(rule.id)[-3:])
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

    # ------- Helper: Rule Embed -------
    def format_action(self, action):
        t = action.type.name
        msg = ""
        if t == "block_message":
            msg += "- Block Message"
            if getattr(action, "custom_message", None):
                msg += f"\n  Custom: {action.custom_message}"
        elif t == "send_alert_message":
            msg += f"- Alert to <#{getattr(action, 'channel_id', None)}>"
        elif t == "timeout":
            d = getattr(action, "duration", None)
            if d:
                mins = d.total_seconds() // 60
                msg += f"- Timeout ({int(mins)} min)"
            else:
                msg += "- Timeout"
        else:
            msg += f"- {t}"
        return msg

    def format_rule_embed(self, rule):
        trigger = getattr(rule, "trigger", None)
        trigger_type = getattr(trigger, "type", None)
        readable_type = str(trigger_type)
        em = discord.Embed(
            title=f"AutoMod Rule: {rule.name}",
            description=f"Type: `{readable_type}`\nEnabled: `{rule.enabled}`\nRule ID: `{rule.id}`",
            color=discord.Color.blurple(),
        )
        # Add relevant metadata fields
        meta = getattr(rule, "trigger_metadata", None)
        if meta:
            if hasattr(meta, "keyword_filter") and meta.keyword_filter:
                em.add_field(name="Keyword Filter", value="\n".join(meta.keyword_filter), inline=False)
            if hasattr(meta, "allow_list") and meta.allow_list:
                em.add_field(name="Allowed List", value="\n".join(meta.allow_list), inline=False)
            if hasattr(meta, "regex_patterns") and meta.regex_patterns:
                em.add_field(name="Regex Patterns", value="\n".join(meta.regex_patterns), inline=False)
            if hasattr(meta, "mention_total_limit"):
                em.add_field(name="Mention Limit", value=str(meta.mention_total_limit), inline=False)
        # Actions
        actions = rule.actions if hasattr(rule, "actions") else []
        actions_str = "\n".join(self.format_action(a) for a in actions) or "*None*"
        em.add_field(name="Actions", value=actions_str, inline=False)
        # Creator
        creator = getattr(rule, "creator", None)
        creator_str = (getattr(creator, "mention", None) if creator else None) or f"`{getattr(rule, 'creator_id', 'unknown')}`"
        em.add_field(name="Creator", value=creator_str, inline=True)
        # Exempt roles/channels
        if hasattr(rule, "exempt_roles") and rule.exempt_roles:
            em.add_field(name="Exempt Roles", value="\n".join(r.mention for r in rule.exempt_roles), inline=True)
        if hasattr(rule, "exempt_channels") and rule.exempt_channels:
            em.add_field(name="Exempt Channels", value="\n".join(c.mention for c in rule.exempt_channels), inline=True)
        return em

    # ------- Main group -------
    @commands.group(invoke_without_command=True)
    @commands.guild_only()
    @_require_perm()
    async def automod(self, ctx, rule: str = None, action: str = None, *args):
        """
        Show info or edit AutoMod rules.
        Usage:
          !automod <shortname>         # Info about rule
          !automod <shortname> enable/disable
          !automod <shortname> allow add word1,word2
          !automod <shortname> allow remove word1,word2
          !automod list
        """
        if rule is None:
            return await ctx.send_help()
        # ---- Get Rule ----
        shortmap = await self.get_shortname_mapping(ctx.guild)
        rule_id = shortmap.get(rule) if not rule.isdigit() else int(rule)
        if not rule_id:
            return await ctx.send("Rule not found by short name or ID.\nUse `!automod list` to see available rules and their short names.")
        try:
            rule_obj = await ctx.guild.fetch_automod_rule(rule_id)
        except Exception as e:
            return await ctx.send(f"Could not fetch rule: {e}")

        # ---- Subcommands: enable/disable ----
        if action is not None:
            action = action.lower()
            if action in ("enable", "on"):
                await rule_obj.edit(enabled=True, reason=f"Enabled by {ctx.author}")
                await ctx.send(f"Rule **{rule_obj.name}** is now ENABLED.")
                return
            elif action in ("disable", "off"):
                await rule_obj.edit(enabled=False, reason=f"Disabled by {ctx.author}")
                await ctx.send(f"Rule **{rule_obj.name}** is now DISABLED.")
                return
            elif action == "allow":
                # Only for keyword rules!
                meta = getattr(rule_obj, "trigger_metadata", None)
                allow_supported = meta and hasattr(meta, "allow_list")
                if not allow_supported:
                    await ctx.send("This rule does **not** support allowed words/phrases.")
                    return
                if not args:
                    await ctx.send("Usage: `!automod <shortname> allow add/remove word1,word2`")
                    return
                allow_cmd, *allow_words = args
                allow_cmd = allow_cmd.lower()
                word_string = " ".join(allow_words)
                words = [w.strip() for w in word_string.replace("\n", ",").split(",") if w.strip()]
                if allow_cmd == "add":
                    # Add to allow_list
                    newlist = set(meta.allow_list or [])
                    before = set(newlist)
                    newlist.update(words)
                    try:
                        await rule_obj.edit(trigger_metadata=discord.AutoModRuleTriggerMetadata(
                            allow_list=list(newlist),
                            keyword_filter=meta.keyword_filter
                        ))
                        added = ", ".join(set(words) - before)
                        await ctx.send(f"Added: {added or '*Nothing new*'}")
                    except Exception as e:
                        await ctx.send(f"Failed to update allow list: {e}")
                elif allow_cmd == "remove":
                    newlist = set(meta.allow_list or [])
                    before = set(newlist)
                    newlist.difference_update(words)
                    try:
                        await rule_obj.edit(trigger_metadata=discord.AutoModRuleTriggerMetadata(
                            allow_list=list(newlist),
                            keyword_filter=meta.keyword_filter
                        ))
                        removed = ", ".join(before - set(newlist))
                        await ctx.send(f"Removed: {removed or '*Nothing removed*'}")
                    except Exception as e:
                        await ctx.send(f"Failed to update allow list: {e}")
                else:
                    await ctx.send("Usage: `!automod <shortname> allow add/remove word1,word2`")
                return
            else:
                await ctx.send("Unknown action. Try `enable`, `disable`, `allow add ...`, or `allow remove ...`.")
                return

        # ---- Default: Show info about rule ----
        em = self.format_rule_embed(rule_obj)
        await ctx.send(embed=em)

    # ------- List all rules -------
    @automod.command(name="list")
    @_require_perm()
    async def list_rules(self, ctx):
        """List all AutoMod rules with their short names."""
        rules = await ctx.guild.fetch_automod_rules()
        if not rules:
            return await ctx.send("No AutoMod rules found.")
        shortmap = await self.get_shortname_mapping(ctx.guild)
        msg = ""
        for shortname, ruleid in shortmap.items():
            rule = discord.utils.get(rules, id=ruleid)
            if rule:
                msg += f"**{shortname}** — {rule.name} (ID: `{rule.id}`)\n"
        await ctx.send(f"**AutoMod rules and short names:**\n{msg or 'None'}")

    # ------- Role management -------
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
    @_require_perm()
    async def roles(self, ctx):
        """List all roles allowed to use automod management commands."""
        allowed_roles = await self.config.guild(ctx.guild).allowed_roles()
        if not allowed_roles:
            return await ctx.send("No roles are currently allowed to use automod management commands.")
        mentions = [f"<@&{role_id}>" for role_id in allowed_roles if ctx.guild.get_role(role_id)]
        await ctx.send("Allowed roles: " + ", ".join(mentions) if mentions else "No valid roles found.")

async def setup(bot):
    await bot.add_cog(D1AutoMod(bot))
