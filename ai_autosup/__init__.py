from .AI_Autosup import AIHelp

async def setup(bot):
    await bot.add_cog(AIHelp(bot))
