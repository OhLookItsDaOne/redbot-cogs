from .aihelp import AIHelp

async def setup(bot):
    await bot.add_cog(AIHelp(bot))

