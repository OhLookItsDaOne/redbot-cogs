import discord
from redbot.core import commands, Config

def automod_manage_check():
    async def predicate(ctx):
        cog = ctx.cog
        return await cog.has_automod_permission(ctx)
    return commands.check(predicate)

class D1AutoMod(commands.Cog):
    """AutoMod: Manage Discord AutoMod keyword rules and allowed words/phrases interactively."""

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

    @commands.group(invoke_without_command=True)
    @commands.guild_only()
    @automod_manage_check()
    async def automod(self, ctx, rule: str = None, action: str = None, *args):
        """
        Manage and inspect Discord AutoMod rules.

        Usage:
        !automod <shortname>
        !automod <shortname> enable
        !automod <shortname> disable
        !automod <shortname> add word1,word2
        !automod <shortname> remove word1,word2
        !automod list
        """
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

        # --- Process action if present ---
        performed_action = False
        msg_action = ""
        trigger_meta = getattr(rule_obj, "trigger_metadata", None)
        # Always allow add/remove for keyword rules, even if allow_list is missing!
        is_keyword = getattr(getattr(rule_obj, "trigger", None), "type", None)
        is_keyword_rule = (str(is_keyword).lower() == "keyword" or str(is_keyword).endswith(".keyword") or str(is_keyword).endswith(": 1>"))

        if action is not None:
            lower_action = action.lower()
            if lower_action == "enable":
                await rule_obj.edit(enabled=True)
                msg_action = "Rule enabled."
                performed_action = True
            elif lower_action == "disable":
                await rule_obj.edit(enabled=False)
                msg_action = "Rule disabled."
                performed_action = True
            elif lower_action in ("add", "remove") and is_keyword_rule:
                if not args:
                    return await ctx.send(f"Please specify words to {lower_action}: `!automod <rule> {lower_action} word1,word2`")
                added_or_removed = [w.strip() for w in " ".join(args).replace("\n", ",").split(",") if w.strip()]
                # Always default to empty if missing!
                allow_list = set(getattr(trigger_meta, "allow_list", []) or [])
                before = set(allow_list)
                if lower_action == "add":
                    allow_list.update(added_or_removed)
                else:
                    allow_list.difference_update(added_or_removed)
                try:
                    await rule_obj.edit(
                        trigger_metadata=discord.AutoModRuleTriggerMetadata(
                            allow_list=list(allow_list),
                            keyword_filter=getattr(trigger_meta, "keyword_filter", [])
                        )
                    )
                    if lower_action == "add":
                        msg_action = f"Added: {', '.join(set(added_or_removed) - before)}" if set(added_or_removed) - before else "No new words added."
                    else:
                        msg_action = f"Removed: {', '.join(before - set(allow_list))}" if before - set(allow_list) else "No words were removed."
                except Exception as e:
                    msg_action = f"Failed to update rule: {e}"
                performed_action = True
            elif lower_action in ("add", "remove"):
                return await ctx.send("This rule does not support allowed words/phrases.")
            else:
                return await ctx.send("Unknown action or unsupported for this rule. Use `enable`, `disable`, `add`, or `remove`.")
            # Refresh for updated state!
            try:
                rule_obj = await ctx.guild.fetch_automod_rule(rule_id)
                trigger_meta = getattr(rule_obj, "trigger_metadata", None)
            except Exception:
                pass


        # Parse action for enable/disable/add/remove
        if action is not None:
            lower_action = action.lower()
            if lower_action == "enable":
                await rule_obj.edit(enabled=True)
                msg_action = "Rule enabled."
                performed_action = True
            elif lower_action == "disable":
                await rule_obj.edit(enabled=False)
                msg_action = "Rule disabled."
                performed_action = True
            elif lower_action == "add" and allow_list_supported:
                if not args:
                    return await ctx.send("Please specify words to add: `!automod <rule> add word1,word2`")
                added = [w.strip() for w in " ".join(args).replace("\n", ",").split(",") if w.strip()]
                allow_list = set(trigger_meta.allow_list or [])
                before = set(allow_list)
                allow_list.update(added)
                try:
                    await rule_obj.edit(
                        trigger_metadata=discord.AutoModRuleTriggerMetadata(
                            allow_list=list(allow_list),
                            keyword_filter=trigger_meta.keyword_filter
                        )
                    )
                    msg_action = f"Added: {', '.join(set(added) - before)}" if set(added) - before else "No new words added."
                except Exception as e:
                    msg_action = f"Failed to update rule: {e}"
                performed_action = True
            elif lower_action == "remove" and allow_list_supported:
                if not args:
                    return await ctx.send("Please specify words to remove: `!automod <rule> remove word1,word2`")
                to_remove = [w.strip() for w in " ".join(args).replace("\n", ",").split(",") if w.strip()]
                allow_list = set(trigger_meta.allow_list or [])
                before = set(allow_list)
                allow_list.difference_update(to_remove)
                try:
                    await rule_obj.edit(
                        trigger_metadata=discord.AutoModRuleTriggerMetadata(
                            allow_list=list(allow_list),
                            keyword_filter=trigger_meta.keyword_filter
                        )
                    )
                    msg_action = f"Removed: {', '.join(before - set(allow_list))}" if before - set(allow_list) else "No words were removed."
                except Exception as e:
                    msg_action = f"Failed to update rule: {e}"
                performed_action = True
            elif lower_action in ("add", "remove") and not allow_list_supported:
                return await ctx.send("This rule does not support allowed words/phrases.")
            else:
                return await ctx.send("Unknown action or unsupported for this rule. Use `enable`, `disable`, `add`, or `remove`.")

            # Refresh rule object to show latest state
            try:
                rule_obj = await ctx.guild.fetch_automod_rule(rule_id)
            except Exception:
                pass

        # --- Compose embed like FUS RO DA BOT ---
        embed = discord.Embed(
            title=f"AutoMod Rule: {rule_obj.name}",
            description=(
                f"**Type:** `{getattr(getattr(rule_obj, 'trigger', None), 'type', None)}`\n"
                f"**Enabled:** `{getattr(rule_obj, 'enabled', None)}`\n"
                f"**Rule ID:** `{rule_obj.id}`"
            ),
        )

        # List all info fields for known rule types
        meta = getattr(rule_obj, "trigger_metadata", None)
        trigger = getattr(rule_obj, "trigger", None)
        if meta:
            if hasattr(meta, "keyword_filter") and meta.keyword_filter:
                embed.add_field(name="Keyword Filter", value=", ".join(meta.keyword_filter), inline=False)
            if hasattr(meta, "allow_list") and meta.allow_list:
                embed.add_field(name="Allowed List", value=", ".join(meta.allow_list), inline=False)
            if hasattr(meta, "regex_patterns") and meta.regex_patterns:
                embed.add_field(name="Regex Patterns", value=", ".join(meta.regex_patterns), inline=False)
            if hasattr(meta, "mention_total_limit") and meta.mention_total_limit is not None:
                embed.add_field(name="Mention Limit", value=str(meta.mention_total_limit), inline=False)
            # Add other meta fields as you wish

        # Show actions
        if hasattr(rule_obj, "actions"):
            acts = []
            for act in rule_obj.actions:
                acts.append(f"Type: {getattr(act, 'type', None)} Channel: {getattr(act, 'channel', None)} Duration: {getattr(act, 'duration', None)}")
            if acts:
                embed.add_field(name="Actions", value="\n".join(acts), inline=False)

        # Show who created
        if hasattr(rule_obj, "creator"):
            embed.set_footer(text=f"Created by: {getattr(rule_obj.creator, 'name', '')} ({getattr(rule_obj, 'creator_id', '')})")

        # Add last action msg if any
        if performed_action:
            embed.insert_field_at(0, name="Update", value=msg_action, inline=False)

        await ctx.send(embed=embed)

    @automod.command(name="enable")
    @automod_manage_check()
    async def enable_rule(self, ctx, rule: str):
        """Enable an AutoMod rule."""
        shortmap = await self.get_shortname_mapping(ctx.guild)
        rule_id = shortmap.get(rule) if not rule.isdigit() else int(rule)
        if not rule_id:
            return await ctx.send("Rule not found by short name or ID.")
        rule_obj = await ctx.guild.fetch_automod_rule(rule_id)
        await rule_obj.edit(enabled=True)
        await ctx.send(f"Rule **{rule_obj.name}** is now **enabled**.")

    @automod.command(name="disable")
    @automod_manage_check()
    async def disable_rule(self, ctx, rule: str):
        """Disable an AutoMod rule."""
        shortmap = await self.get_shortname_mapping(ctx.guild)
        rule_id = shortmap.get(rule) if not rule.isdigit() else int(rule)
        if not rule_id:
            return await ctx.send("Rule not found by short name or ID.")
        rule_obj = await ctx.guild.fetch_automod_rule(rule_id)
        await rule_obj.edit(enabled=False)
        await ctx.send(f"Rule **{rule_obj.name}** is now **disabled**.")

    @automod.group(name="allow")
    @automod_manage_check()
    async def allow(self, ctx, rule: str = None):
        """Manage allowed words/phrases for a keyword rule."""
        if rule is None:
            return await ctx.send("You must specify a rule.")

    @allow.command(name="add")
    @automod_manage_check()
    async def allow_add(self, ctx, rule: str, *, words: str):
        """Add allowed words/phrases (comma or newline separated)."""
        shortmap = await self.get_shortname_mapping(ctx.guild)
        rule_id = shortmap.get(rule) if not rule.isdigit() else int(rule)
        if not rule_id:
            return await ctx.send("Rule not found by short name or ID.")
        rule_obj = await ctx.guild.fetch_automod_rule(rule_id)
        if not hasattr(rule_obj.trigger_metadata, "allow_list"):
            return await ctx.send("This rule does not support allowed words.")
        added = [w.strip() for w in words.replace("\n", ",").split(",") if w.strip()]
        allow_list = set(rule_obj.trigger_metadata.allow_list or [])
        before = set(allow_list)
        allow_list.update(added)
        try:
            await rule_obj.edit(
                trigger_metadata=discord.AutoModRuleTriggerMetadata(
                    allow_list=list(allow_list),
                    keyword_filter=rule_obj.trigger_metadata.keyword_filter
                )
            )
            msg = f"Added: {', '.join(set(added) - before)}" if set(added) - before else "No new words added."
        except Exception as e:
            msg = f"Failed to update rule: {e}"
        await ctx.send(msg)

    @allow.command(name="remove")
    @automod_manage_check()
    async def allow_remove(self, ctx, rule: str, *, words: str):
        """Remove allowed words/phrases (comma or newline separated)."""
        shortmap = await self.get_shortname_mapping(ctx.guild)
        rule_id = shortmap.get(rule) if not rule.isdigit() else int(rule)
        if not rule_id:
            return await ctx.send("Rule not found by short name or ID.")
        rule_obj = await ctx.guild.fetch_automod_rule(rule_id)
        if not hasattr(rule_obj.trigger_metadata, "allow_list"):
            return await ctx.send("This rule does not support allowed words.")
        to_remove = [w.strip() for w in words.replace("\n", ",").split(",") if w.strip()]
        allow_list = set(rule_obj.trigger_metadata.allow_list or [])
        before = set(allow_list)
        allow_list.difference_update(to_remove)
        try:
            await rule_obj.edit(
                trigger_metadata=discord.AutoModRuleTriggerMetadata(
                    allow_list=list(allow_list),
                    keyword_filter=rule_obj.trigger_metadata.keyword_filter
                )
            )
            msg = f"Removed: {', '.join(before - set(allow_list))}" if before - set(allow_list) else "No words were removed."
        except Exception as e:
            msg = f"Failed to update rule: {e}"
        await ctx.send(msg)

    @automod.command(name="list")
    @automod_manage_check()
    async def list_rules(self, ctx):
        """List all AutoMod rules with their short names."""
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
    @automod_manage_check()
    async def roles(self, ctx):
        """List all roles allowed to use automod management commands."""
        allowed_roles = await self.config.guild(ctx.guild).allowed_roles()
        if not allowed_roles:
            return await ctx.send("No roles are currently allowed to use automod management commands.")
        mentions = [f"<@&{role_id}>" for role_id in allowed_roles if ctx.guild.get_role(role_id)]
        await ctx.send("Allowed roles: " + ", ".join(mentions) if mentions else "No valid roles found.")

async def setup(bot):
    await bot.add_cog(D1AutoMod(bot))
