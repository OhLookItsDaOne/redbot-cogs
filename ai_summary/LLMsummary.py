import discord
from redbot.core import commands, Config
import sys
import subprocess

# Sicherstellen, dass ollama installiert ist
try:
    import ollama
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "ollama"], check=True)
    import ollama

class LLMSummary(commands.Cog):
    """Cog, der einen Thread oder Channel zusammenfassen und die Zusammenfassung per DM an den Command-Invoker (Admin) senden kann."""
    
    def __init__(self, bot):
        self.bot = bot
        # Konfigurationswerte für die LLM-Parameter:
        self.config = Config.get_conf(self, identifier=9876543211)
        default_global = {
            "model": "default-llm",
            "context_length": 4864,
            "api_url": "http://localhost:11434",
            "summary_message_limit": 50  # Anzahl der Nachrichten, die standardmäßig zusammengefasst werden
        }
        self.config.register_global(**default_global)
    
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setsummarylimit(self, ctx, limit: int):
        """Legt die Anzahl der Nachrichten fest, die für die Zusammenfassung abgerufen werden.
        Beispiel: !setsummarylimit 50
        """
        await self.config.summary_message_limit.set(limit)
        await ctx.send(f"Nachrichtenlimit für die Zusammenfassung wurde auf {limit} gesetzt.")
    
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def summary(self, ctx, message_limit: int = None):
        """Fasst die letzten Nachrichten eines Channels (oder Threads) zusammen und sendet die Zusammenfassung per DM an dich.
        
        Optional kannst du die Anzahl der zu berücksichtigenden Nachrichten angeben (Standardwert ist in der Config gesetzt).
        Beispiel: !summary oder !summary 100
        """
        # Bestimme den Nachrichten-Limit-Wert
        default_limit = await self.config.summary_message_limit()
        if message_limit is None:
            message_limit = default_limit
        
        # Hole die letzten Nachrichten aus dem aktuellen Channel oder Thread
        messages = []
        async for msg in ctx.channel.history(limit=message_limit, oldest_first=True):
            # Für die Zusammenfassung reicht meist der Textinhalt und der Autorname.
            # Du kannst den Formatierungsstil anpassen, falls erforderlich.
            messages.append(f"{msg.author.display_name}: {msg.content}")
        
        if not messages:
            await ctx.send("Keine Nachrichten gefunden, die zusammengefasst werden könnten.")
            return

        # Erstelle den Prompt für die LLM: eine Aufforderung zur Zusammenfassung
        conversation = "\n".join(messages)
        prompt = (
            "Fasse bitte folgende Konversation prägnant zusammen. "
            "Nutze als Grundlage nur die dargestellten Inhalte und fasse die Kernaussagen zusammen.\n\n"
            f"{conversation}\n\nZusammenfassung:"
        )
        
        # Lese die LLM-Konfiguration aus der globalen Config
        model = await self.config.model()
        context_length = await self.config.context_length()
        api_url = await self.config.api_url()
        
        try:
            # Sende den Prompt an Ollama (als Chat-Anfrage)
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"context": context_length},
                base_url=api_url
            )
            summary_text = response["message"]["content"]
        except Exception as e:
            await ctx.send(f"Fehler bei der Kommunikation mit der LLM: {e}")
            return
        
        # Sende die Zusammenfassung per DM an den Admin (den Command-Invoker)
        admin = ctx.author
        try:
            await admin.send(
                f"Hier ist die Zusammenfassung der letzten {message_limit} Nachrichten aus {ctx.channel.mention}:\n\n{summary_text}"
            )
            await ctx.send("Die Zusammenfassung wurde dir per DM gesendet.")
        except Exception as e:
            await ctx.send(f"Fehler beim Senden der DM: {e}")
