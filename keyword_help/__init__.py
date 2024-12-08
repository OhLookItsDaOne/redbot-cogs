from redbot.core.bot import Red
from .keyword_help import KeywordHelp

async def setup(bot: Red):
    cog = KeywordHelp(bot)
    await bot.add_cog(cog)
