from .ai_autosup import AIHelp

async def setup(bot):
    await bot.add_cog(AIHelp(bot))
