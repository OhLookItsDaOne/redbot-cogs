import discord
from redbot.core import commands, Config

class D1AutoMod(commands.Cog):
    """AutoMod: Manage Discord AutoMod rules (fully via commands, permission protected)."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=2468101214)
        default_guild = {"allowed_roles": [], "shortnames": {}}
        self.config.register_guild(**default_guild)

    async def has_automod_permission(self, ctx):
        """Admin or allowed role required."""
        if ctx.author.guild_permissions.administrator:
            return True
        allowed_roles = await self.config.guild(ctx.guild).allowed_roles()
        return any(role.id in allowed_roles for role in ctx.author.roles)

    async def get_shortname_mapping(self, guild):
        """Create/update shortname mapping for easier rule calls."""
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

    def format_rule_embed(self, rule_obj):
        """Format an info embed for a rule (all details, FUS RO BOT style)."""
        trigger = getattr(rule_obj, "trigger", None)
        trigger_type = getattr(trigger, "type", None)
        readable_type = str(trigger_type)
        enabled = getattr(rule_obj, "enabled", None)
        meta = getattr(rule_obj, "trigger_metadata", None)
        actions = getattr(rule_obj, "actions", [])
        action_strs = []
        for act in actions:
            # Example: block message, send alert etc
            act_type = getattr(act, "type", None)
            chan = getattr(act, "channel", None)
            duration = getattr(act, "duration", None)
            line = f"Type: {act_type} Channel: {getattr(chan, 'mention', None) or getattr(chan, 'id', None) or 'None'} Duration: {duration or 'None'}"
            action_strs.append(line)
        creator = getattr(rule_obj, "creator", None)
        creator_str = f"{getattr(creator, 'name', 'Unknown')} ({getattr(creator, 'id', 'N/A')})" if creator else "Unknown"

        embed = discord.Embed(
            title=f"AutoMod Rule: {rule_obj.name}",
            description=(
                f"Type: `{readable_type}`\n"
                f"Enabled: {enabled}\n"
                f"Rule ID: {rule_obj.id}"
            ),
        )
        # Metadata fields
        if meta:
            if hasattr(meta, "keyword_filter") and meta.keyword_filter:
                embed.add_field(name="Keyword Filter", value=", ".join(meta.keyword_filter), inline=False)
            if hasattr(meta, "allow_list") and meta.allow_list:
                embed.add_field(name="Allowed Words/Phrases", value=", ".join(meta.allow_list), inline=False)
            if hasattr(meta, "regex_patterns") and meta.regex_patterns:
                embed.add_field(name="Regex Patterns", value=", ".join(meta.regex_patterns), inline=False)
            if hasattr(meta, "mention_total_limit"):
                embed.add_field(name="Mention Limit", value=str(meta.mention_total_limit), inline=False)
        if action_strs:
            embed.add_field(name="Actions", value="\n".join(action_strs), inline=False)
        embed.add_field(name="Created by", value=creator_str, inline=False)
        return embed

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
            meta = getattr(rule_obj, "trigger_metadata", None)
            # ENABLE/DISABLE
            if action in ("enable", "on"):
                await rule_obj.edit(enabled=True, reason=f"Enabled by {ctx.author}")
                await ctx.send(f"Rule **{rule_obj.name}** is now ENABLED.")
                return
            elif action in ("disable", "off"):
                await rule_obj.edit(enabled=False, reason=f"Disabled by {ctx.author}")
                await ctx.send(f"Rule **{rule_obj.name}** is now DISABLED.")
                return
            # ALLOW LIST (keyword rules only)
            elif action == "allow":
                allow_supported = meta and hasattr(meta, "allow_list")
                if not allow_supported:
                    return await ctx.send("This rule does **not** support allowed words/phrases.")
                if not args or len(args) < 2:
                    return await ctx.send("Usage: `!automod <shortname> allow add/remove word1,word2`")
                allow_cmd = args[0].lower()
                word_string = " ".join(args[1:])
                words = [w.strip() for w in word_string.replace("\n", ",").split(",") if w.strip()]
                if allow_cmd == "add":
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

        # Show info
        em = self.format_rule_embed(rule_obj)
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
        """Allow a role to use automod management commands."""
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
        """Remove a role from automod management access."""
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
