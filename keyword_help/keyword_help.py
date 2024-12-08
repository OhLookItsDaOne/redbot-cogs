import discord
from redbot.core import commands, Config
from datetime import datetime, timedelta

class KeywordHelp(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1234567890, force_registration=True)
        self.config.register_global(keywords={})  # Speichert Keywords und Antworten
        self.config.register_global(channel_ids=[])  # Speichert die Liste der Channel-IDs
        self.user_help_log = {}  # Temporäres Log für Cooldown-Checks
        self.cooldown = timedelta(minutes=10)  # Cooldown-Zeit für Hilfe pro Benutzer

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def add_channel(self, ctx, channel_id: int):
        """Add a channel (by ID) where the bot will monitor messages for keywords."""
        async with self.config.channel_ids() as channel_ids:
            if channel_id not in channel_ids:
                channel_ids.append(channel_id)
                await ctx.send(f"Channel with ID `{channel_id}` has been added to the monitoring list.")
            else:
                await ctx.send(f"Channel with ID `{channel_id}` is already being monitored.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def remove_channel(self, ctx, channel_id: int):
        """Remove a channel (by ID) from the monitoring list."""
        async with self.config.channel_ids() as channel_ids:
            if channel_id in channel_ids:
                channel_ids.remove(channel_id)
                await ctx.send(f"Channel with ID `{channel_id}` has been removed from the monitoring list.")
            else:
                await ctx.send(f"Channel with ID `{channel_id}` was not found in the monitoring list.")

    @commands.command()
    async def list_channels(self, ctx):
        """List all active channels being monitored."""
        channel_ids = await self.config.channel_ids()
        if not channel_ids:
            await ctx.send("No channels are currently being monitored.")
        else:
            channel_list = "\n".join([f"- <#{channel_id}> (ID: {channel_id})" for channel_id in channel_ids])
            await ctx.send(f"Currently monitored channels:\n{channel_list}")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def add_keyword(self, ctx, keyword: str, *, response: str):
        """Add a keyword and its associated response."""
        async with self.config.keywords() as keywords:
            keywords[keyword.lower()] = response
        await ctx.send(f"Keyword `{keyword}` with response `{response}` has been added.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def remove_keyword(self, ctx, *, keyword: str):
        """Remove a keyword and its associated response."""
        async with self.config.keywords() as keywords:
            if keyword.lower() in keywords:
                del keywords[keyword.lower()]
                await ctx.send(f"Keyword `{keyword}` has been removed.")
            else:
                await ctx.send(f"Keyword `{keyword}` was not found.")

    @commands.command()
    async def list_keywords(self, ctx):
        """List all keywords and their associated responses."""
        keywords = await self.config.keywords()
        if not keywords:
            await ctx.send("No keywords have been added yet.")
        else:
            keywords_list = "\n".join([f"- {k}: {v}" for k, v in keywords.items()])
            await ctx.send(f"Here are the current keywords and responses:\n{keywords_list}")

    @commands.Cog.listener()
    async def on_message(self, message):
        """Listen for messages in monitored channels and respond to keywords."""
        if message.author.bot:
            return

        channel_ids = await self.config.channel_ids()
        if message.channel.id not in channel_ids:
            return

        content = message.content.lower()
        keywords = await self.config.keywords()

        # Check for keywords in the message
        for keyword, response in keywords.items():
            if keyword in content:
                user_id = message.author.id

                # Check if the user was recently helped
                if self.can_help_user(user_id, keyword):
                    await message.channel.send(f"{response}")
                    self.log_help(user_id, keyword)
                else:
                    await message.channel.send(
                        f"It seems I already helped you with `{keyword}` recently, {message.author.mention}."
                    )
                break  # Avoid responding multiple times for multiple keywords

    def can_help_user(self, user_id, keyword):
        """Checks if the bot can help the user with the given keyword."""
        now = datetime.utcnow()
        if user_id in self.user_help_log:
            user_log = self.user_help_log[user_id]
            if keyword in user_log and now - user_log[keyword] < self.cooldown:
                return False  # Still under cooldown for this keyword
        return True

    def log_help(self, user_id, keyword):
        """Log that a user has been helped with a specific keyword."""
        now = datetime.utcnow()
        if user_id not in self.user_help_log:
            self.user_help_log[user_id] = {}
        self.user_help_log[user_id][keyword] = now

def setup(bot):
    bot.add_cog(KeywordHelp(bot))
