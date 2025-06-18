import discord
from redbot.core import commands, Config
from redbot.core.utils.chat_formatting import inline

class D1AutoMod(commands.Cog):
    """AutoMod: Manage Discord AutoMod rules and allowed words/phrases interactively."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=2468101214)
        default_guild = {"allowed_roles": [], "shortnames": {}}
        self.config.register_guild(**default_guild)

    async def has_automod_permission(self, ctx):
        # Only Admin or allowed roles
        if ctx.author.guild_permissions.administrator:
            return True
        allowed_roles = await self.config.guild(ctx.guild).allowed_roles()
        return any(role.id in allowed_roles for role in ctx.author.roles)

    async def get_shortname_mapping(self, guild):
        # Build shortnames -> rule IDs (cache)
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
    async def automod(self, ctx, rule: str = None, action: str = None, subaction: str = None, *, values: str = None):
        """
        Manage Discord AutoMod rules.
        Only users with allowed roles or admin can use these commands.
        Usage:
        !automod <shortname>
        !automod <shortname> enable/disable
        !automod <shortname> allow add <word1,word2,...>
        !automod <shortname> allow remove <word1,word2,...>
        !automod list
        """
        if not await self.has_automod_permission(ctx):
            return await ctx.send("You do not have permission to use this command.")

        if rule is None:
            return await ctx.send_help()

        # Subcommand: list
        if rule.lower() == "list":
            return await self.list_rules(ctx)

        # Map shortname/ID
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

        # Enable/Disable
        if action and action.lower() in ["enable", "disable"]:
            try:
                await rule_obj.edit(enabled=(action.lower() == "enable"))
                return await ctx.send(f"Rule **{rule_obj.name}** is now {'ENABLED' if action.lower() == 'enable' else 'DISABLED'}.")
            except Exception as e:
                return await ctx.send(f"Failed to change rule state: {e}")

        # Allowlist management (only for keyword rules)
        if action and action.lower() == "allow":
            allow_supported = hasattr(trigger, "allow_list")
            if not allow_supported:
                return await ctx.send("This rule does not support allowed words/phrases.")

            allow_list = list(trigger.allow_list) if trigger.allow_list else []
            if subaction and subaction.lower() == "add" and values:
                words = [w.strip() for w in values.split(",") if w.strip()]
                before = set(allow_list)
                allow_list = list(set(allow_list) | set(words))
                try:
                    await rule_obj.edit(
                        trigger_metadata=discord.AutoModRuleTriggerMetadata(
                            allow_list=allow_list,
                            keyword_filter=getattr(trigger, "keyword_filter", None),
                            regex_patterns=getattr(trigger, "regex_patterns", None)
                        )
                    )
                    added = set(allow_list) - before
                    return await ctx.send(
                        f"Added to allowed list: {', '.join(added) if added else 'None'}"
                    )
                except Exception as e:
                    return await ctx.send(f"Failed to update allow list: {e}")
            elif subaction and subaction.lower() == "remove" and values:
                words = [w.strip() for w in values.split(",") if w.strip()]
                before = set(allow_list)
                allow_list = [w for w in allow_list if w not in words]
                try:
                    await rule_obj.edit(
                        trigger_metadata=discord.AutoModRuleTriggerMetadata(
                            allow_list=allow_list,
                            keyword_filter=getattr(trigger, "keyword_filter", None),
                            regex_patterns=getattr(trigger, "regex_patterns", None)
                        )
                    )
                    removed = before - set(allow_list)
                    return await ctx.send(
                        f"Removed from allowed list: {', '.join(removed) if removed else 'None'}"
                    )
                except Exception as e:
                    return await ctx.send(f"Failed to update allow list: {e}")
            else:
                # Show current allowlist
                return await ctx.send(
                    f"Current allowed list: {', '.join(allow_list) if allow_list else '*None*'}"
                )

        # Show info for rule (as nice embed)
        embed = discord.Embed(
            title=f"AutoMod Rule: {rule_obj.name}",
            description=(
                f"Type: `{readable_type}`\n"
                f"Enabled: {enabled}\n"
                f"Rule ID: {rule_obj.id}"
            ),
        )

        # Show triggers/filters/limits
        trigger = getattr(rule_obj, "trigger", None)
        meta = getattr(rule_obj, "trigger_metadata", None) or getattr(trigger, "trigger_metadata", None)
        if hasattr(trigger, "keyword_filter") and trigger.keyword_filter:
            embed.add_field(name="Keyword Filter", value=", ".join(trigger.keyword_filter), inline=False)
        if hasattr(trigger, "allow_list") and trigger.allow_list:
            embed.add_field(name="Allowed List", value=", ".join(trigger.allow_list), inline=False)
        if hasattr(trigger, "regex_patterns") and trigger.regex_patterns:
            embed.add_field(name="Regex Patterns", value=", ".join(trigger.regex_patterns), inline=False)
        if hasattr(trigger, "mention_limit"):
            embed.add_field(name="Mention Limit", value=str(trigger.mention_limit), inline=False)
        if meta and hasattr(meta, "mention_total_limit"):
            embed.add_field(name="Mention Total Limit", value=str(meta.mention_total_limit), inline=False)

        # Show actions
        action_str = ""
        for action in getattr(rule_obj, "actions", []):
            # These values might be discord enums; show as string for clarity
            act_type = getattr(action, "type", None)
            ch = getattr(action, "channel", None) or getattr(action, "channel_id", None)
            dur = getattr(action, "duration", None)
            custom_msg = getattr(action, "custom_message", None)
            action_str += f"Type: {act_type} Channel: {ch} Duration: {dur}\n"
            if custom_msg:
                action_str += f"Custom Message: {custom_msg}\n"
        if not action_str:
            action_str = "*None*"
        embed.add_field(name="Actions", value=action_str, inline=False)

        # Show creator, exemptions, etc.
        creator = getattr(rule_obj, "creator", None)
        creator_id = getattr(rule_obj, "creator_id", None)
        embed.add_field(
            name="Created by",
            value=f"{getattr(creator, 'mention', None) or inline(str(creator_id))}",
            inline=False,
        )
        # Exempt roles/channels
        exroles = getattr(rule_obj, "exempt_roles", [])
        if exroles:
            embed.add_field(
                name="Exempt Roles",
                value="\n".join(getattr(r, "mention", str(r)) for r in exroles),
                inline=False,
            )
        exchans = getattr(rule_obj, "exempt_channels", [])
        if exchans:
            embed.add_field(
                name="Exempt Channels",
                value="\n".join(getattr(c, "mention", str(c)) for c in exchans),
                inline=False,
            )

        await ctx.send(embed=embed)

    @automod.command(name="list")
    async def list_rules(self, ctx):
        """List all AutoMod rules with their short names."""
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
    async def roles(self, ctx):
        """List all roles allowed to use automod management commands."""
        if not await self.has_automod_permission(ctx):
            return await ctx.send("You do not have permission to use this command.")
        allowed_roles = await self.config.guild(ctx.guild).allowed_roles()
        if not allowed_roles:
            return await ctx.send("No roles are currently allowed to use automod management commands.")
        mentions = [f"<@&{role_id}>" for role_id in allowed_roles if ctx.guild.get_role(role_id)]
        await ctx.send("Allowed roles: " + ", ".join(mentions) if mentions else "No valid roles found.")

async def setup(bot):
    await bot.add_cog(D1AutoMod(bot))
