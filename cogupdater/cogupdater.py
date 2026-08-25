import asyncio
import logging
from pathlib import Path

import discord
from redbot.core import commands, app_commands

log = logging.getLogger("red.cogupdater")


class CogUpdater(commands.Cog):
    """Updates the git repos of all loaded cogs and reloads them.

    The command ``cogupdate`` pulls every git repository that contains at
    least one loaded cog, then reloads those cogs so the new code is used.
    The cog running this command is never reloaded by itself.
    """

    def __init__(self, bot):
        self.bot = bot

    @staticmethod
    def _git_root(start: Path) -> Path | None:
        """Walk upwards from *start* to find the directory containing .git."""
        current = start
        while True:
            if (current / ".git").exists():
                return current
            if current.parent == current:
                return None
            current = current.parent

    @staticmethod
    async def _git_pull(repo: Path):
        """Run ``git pull --ff-only`` in *repo* and return (returncode, output)."""
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(repo),
            "pull",
            "--ff-only",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return proc.returncode, (out + err).decode("utf-8", errors="replace")

    def _extension_groups(self):
        """Map git repo path -> list of loaded extensions in that repo (excluding self)."""
        self_ext = self.__class__.__module__.split(".")[0]
        groups = {}
        for name, ext in self.bot.extensions.items():
            if name == self_ext:
                continue
            f = getattr(ext, "__file__", None)
            if not f:
                continue
            root = self._git_root(Path(f).resolve())
            if root is not None:
                groups.setdefault(str(root), []).append(name)
        return groups

    @commands.hybrid_command(name="cogupdate")
    @commands.is_owner()
    @app_commands.default_permissions(administrator=True)
    async def cogupdate(self, ctx):
        """Pulls the git repos of all loaded cogs and reloads them (except this cog)."""
        groups = self._extension_groups()
        if not groups:
            await ctx.send("❌ No git repos found for the loaded cogs.")
            return

        await ctx.send("🔄 Updating cog repos...")

        repo_lines = []
        to_reload = []
        failed_pulls = []
        for repo, exts in groups.items():
            code, output = await self._git_pull(Path(repo))
            if code == 0:
                repo_lines.append(f"✅ `{repo}`\n```\n{output.strip()}\n```")
                to_reload.extend(exts)
            else:
                repo_lines.append(f"❌ `{repo}` (exit {code})\n```\n{output.strip()}\n```")
                failed_pulls.extend(exts)

        reloaded, failed = [], []
        for name in to_reload:
            try:
                await self.bot.reload_extension(name)
                reloaded.append(name)
            except Exception as e:
                failed.append((name, str(e)))
                log.exception("Failed to reload %s", name)

        embed = discord.Embed(
            title="🔄 Cog Update abgeschlossen",
            colour=discord.Colour.green() if not (failed_pulls or failed) else discord.Colour.red(),
            timestamp=ctx.message.created_at,
        )
        embed.add_field(name="Repos", value="\n".join(repo_lines) or "—", inline=False)
        embed.add_field(name="Reloaded", value=", ".join(f"`{n}`" for n in reloaded) or "—", inline=False)
        if failed_pulls:
            embed.add_field(
                name="Pull fehlgeschlagen",
                value=", ".join(f"`{n}`" for n in failed_pulls),
                inline=False,
            )
        if failed:
            embed.add_field(
                name="Reload fehlgeschlagen",
                value="\n".join(f"`{n}` — {e}" for n, e in failed),
                inline=False,
            )
        embed.set_footer(text=f"{len(reloaded)} cogs reloaded")
        await ctx.send(embed=embed)


async def setup(bot):
    await bot.add_cog(CogUpdater(bot))
