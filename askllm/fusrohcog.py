import aiohttp
import aiofiles
import asyncio
import json
import os
from typing import Optional, List, Dict
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timedelta

import discord
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path

class DeepSeekCog(commands.Cog):
    """Ein Cog, das die DeepSeek API für spezifische Themenanfragen nutzt mit RAG-Lernfunktion."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1234567890)
        default_global = {
            "api_key": None,
            "base_url": "https://api.deepseek.com/v1",
            "context_data": "",
            "context_source": "internal",
            "github_url": None,
            "github_branch": "main",
            "github_path": "",
            "cache_duration": 300,
            "timeout": 30,
            "rag_context_messages": 15,
            "learning_enabled": True
        }
        
        # RAG-Lern-Datenbank
        default_guild = {
            "learned_solutions": {},
            "learning_role": None
        }
        
        self.config.register_global(**default_global)
        self.config.register_guild(**default_guild)
        
        self.data_path = cog_data_path(self) / "context_data.txt"
        self.learned_db_path = cog_data_path(self) / "learned_solutions.json"
        self.session = None
        self.cache = {"data": "", "timestamp": None}
        self.learned_data = {}

    async def cog_load(self):
        """Wird beim Laden des Cogs aufgerufen."""
        timeout = aiohttp.ClientTimeout(total=await self.config.timeout())
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # Gelernnte Daten laden
        await self.load_learned_data()

    async def cog_unload(self):
        """Wird beim Entladen des Cogs aufgerufen."""
        if self.session and not self.session.closed:
            await self.session.close()
        
        # Gelernnte Daten speichern
        await self.save_learned_data()

    async def load_learned_data(self):
        """Lädt die gelernten Lösungen."""
        try:
            if self.learned_db_path.exists():
                async with aiofiles.open(self.learned_db_path, 'r', encoding='utf-8') as f:
                    self.learned_data = json.loads(await f.read())
        except Exception as e:
            print(f"Fehler beim Laden der gelernten Daten: {e}")
            self.learned_data = {}

    async def save_learned_data(self):
        """Speichert die gelernten Lösungen."""
        try:
            async with aiofiles.open(self.learned_db_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.learned_data, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Fehler beim Speichern der gelernten Daten: {e}")

    async def get_message_context(self, message: discord.Message, context_messages: int = 15) -> List[Dict]:
        """Holt den Kontext von Nachrichten um eine bestimmte Nachricht herum."""
        try:
            # Nachrichten vor der Zielnachricht holen
            messages_before = []
            async for msg in message.channel.history(limit=context_messages//2, before=message, oldest_first=False):
                messages_before.append({
                    "author": msg.author.display_name,
                    "content": msg.clean_content,
                    "timestamp": msg.created_at.isoformat()
                })
            
            # Nachrichten nach der Zielnachricht holen
            messages_after = []
            async for msg in message.channel.history(limit=context_messages//2, after=message, oldest_first=True):
                messages_after.append({
                    "author": msg.author.display_name,
                    "content": msg.clean_content,
                    "timestamp": msg.created_at.isoformat()
                })
            
            # Nachrichten zusammenführen und sortieren
            all_messages = messages_before[::-1] + [{
                "author": message.author.display_name,
                "content": message.clean_content,
                "timestamp": message.created_at.isoformat(),
                "is_target": True
            }] + messages_after
            
            return all_messages
            
        except Exception as e:
            print(f"Fehler beim Holen des Nachrichtenkontexts: {e}")
            return []

    async def can_learn(self, user: discord.Member) -> bool:
        """Prüft, ob ein User lernen darf."""
        # Bot-Owner darf immer
        app_info = await self.bot.application_info()
        if user.id == app_info.owner.id:
            return True
        
        # Admin-Rechte prüfen
        if user.guild_permissions.administrator:
            return True
        
        # Spezielle Rolle prüfen
        learning_role_id = await self.config.guild(user.guild).learning_role()
        if learning_role_id:
            learning_role = user.guild.get_role(learning_role_id)
            if learning_role and learning_role in user.roles:
                return True
        
        return False

    async def learn_solution(self, problem_message: discord.Message, solution: str, learner: discord.Member) -> Dict:
        """Speichert eine gelernte Lösung."""
        try:
            # Kontextnachrichten holen
            context_messages = await self.config.rag_context_messages()
            message_context = await self.get_message_context(problem_message, context_messages)
            
            # Problem-Key erstellen (erste 100 Zeichen der Problemnachricht)
            problem_key = problem_message.clean_content[:100].lower().strip()
            
            # Lösung speichern
            learned_entry = {
                "problem": problem_message.clean_content,
                "solution": solution,
                "context": message_context,
                "learned_by": learner.display_name,
                "learned_at": datetime.now().isoformat(),
                "message_id": problem_message.id,
                "channel_id": problem_message.channel.id,
                "guild_id": problem_message.guild.id if problem_message.guild else None
            }
            
            # In Memory speichern
            self.learned_data[problem_key] = learned_entry
            
            # In Datei speichern
            await self.save_learned_data()
            
            return learned_entry
            
        except Exception as e:
            print(f"Fehler beim Speichern der Lösung: {e}")
            return None

    async def find_solution(self, question: str) -> Optional[Dict]:
        """Findet eine passende Lösung für eine Frage."""
        question_lower = question.lower().strip()
        
        # Einfache String-Matching für den Anfang
        for problem_key, solution_data in self.learned_data.items():
            if problem_key in question_lower or any(keyword in question_lower for keyword in problem_key.split()[:3]):
                return solution_data
        
        # Ähnlichkeitsprüfung könnte hier erweitert werden
        return None

    # [Bestehende Methoden: get_github_raw_content, get_github_repo_content, get_context_data, split_message]

    @commands.group()
    @commands.is_owner()
    async def deepseek(self, ctx):
        """Einstellungen für das DeepSeek Cog."""
        pass

    @deepseek.command()
    async def contextmessages(self, ctx, count: int):
        """Setzt die Anzahl der Kontextnachrichten für RAG."""
        if 5 <= count <= 50:
            await self.config.rag_context_messages.set(count)
            await ctx.send(f"Kontextnachrichten wurden auf {count} gesetzt.")
        else:
            await ctx.send("Bitte eine Zahl zwischen 5 und 50 eingeben.")

    @deepseek.command()
    @commands.admin_or_permissions(administrator=True)
    async def learnrole(self, ctx, role: discord.Role = None):
        """Setzt eine Rolle, die lernen darf."""
        if role:
            await self.config.guild(ctx.guild).learning_role.set(role.id)
            await ctx.send(f"Lern-Rolle wurde auf {role.name} gesetzt.")
        else:
            await self.config.guild(ctx.guild).learning_role.set(None)
            await ctx.send("Lern-Rolle wurde entfernt.")

    @deepseek.command()
    async def learning(self, ctx, enabled: bool):
        """Aktiviert/deaktiviert die Lernfunktion."""
        await self.config.learning_enabled.set(enabled)
        status = "aktiviert" if enabled else "deaktiviert"
        await ctx.send(f"Lernfunktion wurde {status}.")

    @commands.command()
    @commands.guild_only()
    async def learn(self, ctx, *, solution: str):
        """Speichert eine Lösung für das vorherige Problem."""
        if not await self.config.learning_enabled():
            await ctx.send("Die Lernfunktion ist deaktiviert.")
            return
        
        if not await self.can_learn(ctx.author):
            await ctx.send("Du hast keine Berechtigung zum Lernen.")
            return
        
        # Referenznachricht finden (reply oder letzte Nachricht)
        target_message = None
        if ctx.message.reference and ctx.message.reference.message_id:
            try:
                target_message = await ctx.channel.fetch_message(ctx.message.reference.message_id)
            except discord.NotFound:
                await ctx.send("Die Referenznachricht wurde nicht gefunden.")
                return
        else:
            # Letzte Nachricht im Channel suchen
            async for message in ctx.channel.history(limit=10, before=ctx.message):
                if message.author != ctx.author and not message.author.bot:
                    target_message = message
                    break
        
        if not target_message:
            await ctx.send("Keine passende Nachricht zum Lernen gefunden.")
            return
        
        # Lösung speichern
        learned_entry = await self.learn_solution(target_message, solution, ctx.author)
        
        if learned_entry:
            # Kontext zur Überprüfung anzeigen
            context_preview = "\n".join([
                f"{msg['author']}: {msg['content'][:50]}..."
                for msg in learned_entry['context'][-3:]
            ])
            
            embed = discord.Embed(
                title="✅ Lösung gespeichert",
                description=f"**Problem:** {learned_entry['problem'][:100]}...",
                color=discord.Color.green()
            )
            embed.add_field(name="Lösung", value=learned_entry['solution'][:500] + "..." if len(learned_entry['solution']) > 500 else learned_entry['solution'], inline=False)
            embed.add_field(name="Kontext", value=context_preview or "Kein Kontext", inline=False)
            embed.add_field(name="Gelernt von", value=learned_entry['learned_by'], inline=True)
            
            await ctx.send(embed=embed)
        else:
            await ctx.send("Fehler beim Speichern der Lösung.")

    @commands.command()
    async def ask(self, ctx, *, question: str):
        """Stellt eine Frage zum spezifischen Thema."""
        # Zuerst nach gelernten Lösungen suchen
        if await self.config.learning_enabled():
            solution = await self.find_solution(question)
            if solution:
                embed = discord.Embed(
                    title="🎓 Gelernte Lösung",
                    description=solution['solution'],
                    color=discord.Color.blue()
                )
                embed.set_footer(text=f"Gelernt von {solution['learned_by']}")
                await ctx.send(embed=embed)
                return
        
        # Falls keine Lösung gefunden, DeepSeek API nutzen
        api_key = await self.config.api_key()
        if not api_key:
            await ctx.send("API-Schlüssel ist nicht konfiguriert.")
            return

        # [Rest der ask-Methode wie zuvor...]

    @commands.command()
    async def forget(self, ctx, *, problem: str):
        """Vergisst eine gelernte Lösung."""
        if not await self.can_learn(ctx.author):
            await ctx.send("Du hast keine Berechtigung zum Vergessen.")
            return
        
        problem_key = problem[:100].lower().strip()
        if problem_key in self.learned_data:
            del self.learned_data[problem_key]
            await self.save_learned_data()
            await ctx.send("✅ Lösung vergessen.")
        else:
            await ctx.send("❌ Keine passende Lösung gefunden.")

    @commands.command()
    async def learned(self, ctx):
        """Zeigt alle gelernten Lösungen an."""
        if not self.learned_data:
            await ctx.send("Noch keine Lösungen gelernt.")
            return
        
        embed = discord.Embed(title="📚 Gelernte Lösungen", color=discord.Color.gold())
        
        for i, (problem_key, solution_data) in enumerate(list(self.learned_data.items())[:5]):
            embed.add_field(
                name=f"Lösung {i+1}",
                value=f"**Problem:** {solution_data['problem'][:50]}...\n**Lösung:** {solution_data['solution'][:100]}...",
                inline=False
            )
        
        if len(self.learned_data) > 5:
            embed.set_footer(text=f"Und {len(self.learned_data) - 5} weitere Lösungen...")
        
        await ctx.send(embed=embed)

async def setup(bot):
    await bot.add_cog(DeepSeekCog(bot))
