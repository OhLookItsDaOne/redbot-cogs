import discord
from redbot.core import commands, Config

class D1AutoMod(commands.Cog):
    """AutoMod: Fully permission-protected, full info, and classic commands."""

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

    def format_rule_embed(self, ctx, rule_obj):
        """Format a full info embed for any rule."""
        guild = ctx.guild
        trigger = getattr(rule_obj, "trigger", None)
        trigger_type = getattr(trigger, "type", None)
        readable_type = str(trigger_type)
        enabled = getattr(rule_obj, "enabled", None)
        meta = getattr(rule_obj, "trigger_metadata", None)
        actions = getattr(rule_obj, "actions", [])
        creator = getattr(rule_obj, "creator", None)
        creator_str = f"<@{creator.id}>" if getattr(creator, "id", None) else "Unknown"

        embed = discord.Embed(
            title=f"AutoMod Rule: {rule_obj.name}",
            description=(
                f"Type: `{readable_type}`\n"
                f"Enabled: {enabled}\n"
                f"Rule ID: {rule_obj.id}"
            ),
        )

        # Keyword filter
        if meta and hasattr(meta, "keyword_filter") and meta.keyword_filter:
            embed.add_field(name="Keyword Filter", value=", ".join(meta.keyword_filter), inline=False)
        # Allowed list (support empty lists too)
        if meta and hasattr(meta, "allow_list"):
            alist = meta.allow_list or []
            embed.add_field(name="Allowed List", value=", ".join(alist) if alist else "*None*", inline=False)
        # Regex patterns
        if meta and hasattr(meta, "regex_patterns") and meta.regex_patterns:
            embed.add_field(name="Regex Patterns", value=", ".join(meta.regex_patterns), inline=False)
        # Mention Limit
        if meta and hasattr(meta, "mention_total_limit"):
            embed.add_field(name="Mention Limit", value=str(meta.mention_total_limit), inline=False)
        # Custom message(s) in actions
        action_strs = []
        for act in actions:
            act_type = getattr(act, "type", None)
            chan = getattr(act, "channel", None)
            duration = getattr(act, "duration", None)
            custom_msg = getattr(act, "custom_message", None)
            # Try .metadata for custom message (discord.py >=2.3)
            metadata = getattr(act, "metadata", {})
            if not custom_msg and isinstance(metadata, dict):
                custom_msg = metadata.get("custom_message")
            chan_mention = getattr(chan, "mention", None) if chan else None
            chan_id = getattr(chan, "id", None) if chan else None
            cstr = (
                f"Type: {act_type} "
                f"Channel: {chan_mention or (f'ID {chan_id}' if chan_id else 'None')} "
                f"Duration: {duration or 'None'}"
            )
            if custom_msg:
                cstr += f"\nCustom Message: {custom_msg}"
            action_strs.append(cstr)
        if action_strs:
            embed.add_field(name="Actions", value="\n".join(action_strs), inline=False)

        # Creator
        embed.add_field(name="Created by", value=creator_str, inline=False)
        # Exempt Roles
        exempt_roles = getattr(rule_obj, "exempt_roles", [])
        if exempt_roles:
            er_mentions = [r.mention for r in exempt_roles if hasattr(r, "mention")]
            if not er_mentions and getattr(rule_obj, "exempt_role_ids", None):
                er_mentions = [f"<@&{rid}>" for rid in getattr(rule_obj, "exempt_role_ids", [])]
            embed.add_field(name="Exempt Roles", value="\n".join(er_mentions) if er_mentions else "*None*", inline=False)
        # Exempt Channels
        exempt_channels = getattr(rule_obj, "exempt_channels", [])
        if exempt_channels:
            ch_mentions = [c.mention for c in exempt_channels if hasattr(c, "mention")]
            if not ch_mentions and getattr(rule_obj, "exempt_channel_ids", None):
                ch_mentions = [f"<#{cid}>" for cid in getattr(rule_obj, "exempt_channel_ids", [])]
            embed.add_field(name="Exempt Channels", value="\n".join(ch_mentions) if ch_mentions else "*None*", inline=False)
        return embed

    def rule_supports_allowlist(self, rule_obj):
        """Detect if this rule supports allowed words/phrases (keyword rules)."""
        # Most reliable: check for trigger type == keyword
        trigger = getattr(rule_obj, "trigger", None)
        trigger_type = getattr(trigger, "type", None)
        return str(trigger_type).lower() == "keyword" or str(trigger_type).endswith(".keyword")

    @commands.group(invoke_without_command=True)
    @commands.guild_only()
    async def automod(self, ctx, rule: str = None, action: str = None, *args):
        """
        Show info or edit AutoMod rules.
        Usage:
        !automod <shortname>         # Info about rule
        !automod <shortname> enable/disable
        !automod <shortname> allow add <word1,word2>
        !automod <shortname> allow remove <word1,word2>
        !automod list
        """
        if not await self.has_automod_permission(ctx):
            return await ctx.send("You do not have permission to use this command.")

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

        # Subcommands
        if action:
            action = action.lower()
            # ENABLE/DISABLE
            if action in ("enable", "on"):
                await rule_obj.edit(enabled=True, reason=f"Enabled by {ctx.author}")
                await ctx.send(f"Rule **{rule_obj.name}** is now ENABLED.")
                return
            elif action in ("disable", "off"):
                await rule_obj.edit(enabled=False, reason=f"Disabled by {ctx.author}")
                await ctx.send(f"Rule **{rule_obj.name}** is now DISABLED.")
                return
            # ALLOW LIST (for keyword rules)
            elif action == "allow":
                if not self.rule_supports_allowlist(rule_obj):
                    return await ctx.send("This rule does not support allowed words/phrases.")
                if not args or len(args) < 2:
                    return await ctx.send("Usage: `!automod <shortname> allow add/remove word1,word2`")
                allow_cmd = args[0].lower()
                word_string = " ".join(args[1:])
                words = [w.strip() for w in word_string.replace("\n", ",").split(",") if w.strip()]
                meta = getattr(rule_obj, "trigger_metadata", None)
                allow_list = set(getattr(meta, "allow_list", []) or [])
                if allow_cmd == "add":
                    before = set(allow_list)
                    allow_list.update(words)
                    try:
                        await rule_obj.edit(trigger_metadata=discord.AutoModRuleTriggerMetadata(
                            allow_list=list(allow_list),
                            keyword_filter=getattr(meta, "keyword_filter", [])
                        ))
                        added = ", ".join(set(words) - before)
                        await ctx.send(f"Added: {added or '*Nothing new*'}")
                    except Exception as e:
                        await ctx.send(f"Failed to update allow list: {e}")
                elif allow_cmd == "remove":
                    before = set(allow_list)
                    allow_list.difference_update(words)
                    try:
                        await rule_obj.edit(trigger_metadata=discord.AutoModRuleTriggerMetadata(
                            allow_list=list(allow_list),
                            keyword_filter=getattr(meta, "keyword_filter", [])
                        ))
                        removed = ", ".join(before - set(allow_list))
                        await ctx.send(f"Removed: {removed or '*Nothing removed*'}")
                    except Exception as e:
                        await ctx.send(f"Failed to update allow list: {e}")
                else:
                    await ctx.send("Usage: `!automod <shortname> allow add/remove word1,word2`")
                return
            else:
                await ctx.send("Unknown action. Try `enable`, `disable`, `allow add ...`, or `allow remove ...`.")
                return

        # Show info
        em = self.format_rule_embed(ctx, rule_obj)
        await ctx.send(embed=em)

    @automod.command(name="list")
    async def list_rules(self, ctx):
        if not await self.has_automod_permission(ctx):
            return await ctx.send("You do not have permission to use this command.")
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
        if not await self.has_automod_permission(ctx):
            return await ctx.send("You do not have permission to use this command.")
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
        if not await self.has_automod_permission(ctx):
            return await ctx.send("You do not have permission to use this command.")
        allowed_roles = await self.config.guild(ctx.guild).allowed_roles()
        if role.id in allowed_roles:
            allowed_roles.remove(role.id)
            await self.config.guild(ctx.guild).allowed_roles.set(allowed_roles)
            await ctx.send(f"{role.mention} can no longer use automod management commands.")
        else:
            await ctx.send(f"{role.mention} is not in the allowed roles.")

    @automod.command(name="roles")
    async def roles(self, ctx):
        if not await self.has_automod_permission(ctx):
            return await ctx.send("You do not have permission to use this command.")
        allowed_roles = await self.config.guild(ctx.guild).allowed_roles()
        if not allowed_roles:
            return await ctx.send("No roles are currently allowed to use automod management commands.")
        mentions = [f"<@&{role_id}>" for role_id in allowed_roles if ctx.guild.get_role(role_id)]
        await ctx.send("Allowed roles: " + ", ".join(mentions) if mentions else "No valid roles found.")

async def setup(bot):
    await bot.add_cog(D1AutoMod(bot))
