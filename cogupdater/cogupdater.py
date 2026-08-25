import asyncio
import logging
import shutil
from pathlib import Path

import discord
from redbot.core import commands, app_commands

log = logging.getLogger("red.cogupdater")


class CogUpdater(commands.Cog):
    """Updates the repos of all loaded cogs and reloads them.

    The command ``cogupdate``:
    1. Finds the git repositories that contain the loaded cogs
       (via the Downloader repos folder and/or git roots).
    2. Runs ``git pull`` in each repo.
    3. Copies the updated cog files from the repo into the installed
       cog locations, so the running cogs actually use the new code.
    4. Reloads those cogs (except this cog, which cannot reload itself).
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

    def _get_repos_folder(self) -> Path | None:
        """Return the Downloader repos folder if available."""
        downloader = self.bot.get_cog("Downloader")
        if downloader is None:
            return None
        try:
            return Path(downloader._repo_manager.repos_folder)
        except Exception:
            return None

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

    def _extension_groups(self) -> dict:
        """Map git repo path -> list of loaded extensions in that repo (excluding self).

        A loaded cog belongs to a repo if:
        - the cog's file is inside a git checkout, OR
        - the cog name matches a folder inside a Downloader repo.
        """
        self_ext = self.__class__.__module__.split(".")[0]
        groups = {}

        def add(name, repo_path):
            groups.setdefault(str(repo_path), [])
            if name not in groups[str(repo_path)]:
                groups[str(repo_path)].append(name)

        for name, ext in self.bot.extensions.items():
            if name == self_ext:
                continue
            f = getattr(ext, "__file__", None)
            if not f:
                continue
            root = self._git_root(Path(f).resolve())
            if root is not None:
                add(name, root)

        # Map remaining loaded cogs to Downloader repos by folder name
        repos_folder = self._get_repos_folder()
        if repos_folder is not None and repos_folder.exists():
            known = set()
            for exts in groups.values():
                known.update(exts)
            for name, ext in self.bot.extensions.items():
                if name == self_ext or name in known:
                    continue
                for repo_dir in repos_folder.iterdir():
                    if repo_dir.is_dir() and (repo_dir / name).is_dir():
                        add(name, repo_dir)
                        break

        return groups

    @staticmethod
    def _sync_cog_files(repo: Path, name: str, installed_dir: Path) -> Path | None:
        """Copy the cog files from the repo folder into the installed location."""
        src = repo / name
        if not src.is_dir() or not installed_dir.is_dir():
            return None
        shutil.copytree(src, installed_dir, dirs_exist_ok=True)
        return installed_dir

    @commands.hybrid_command(name="cogupdate", extras={"red_force_enable": True})
    @commands.is_owner()
    @app_commands.default_permissions(administrator=True)
    async def cogupdate(self, ctx):
        """Pulls the git repos of all loaded cogs, syncs and reloads them (except this cog)."""
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
                for name in exts:
                    ext = self.bot.extensions.get(name)
                    if ext is None or not getattr(ext, "__file__", None):
                        continue
                    installed_dir = Path(ext.__file__).resolve().parent
                    synced = self._sync_cog_files(Path(repo), name, installed_dir)
                    if synced is not None:
                        to_reload.append(name)
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
