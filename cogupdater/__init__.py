from .cogupdater import CogUpdater

async def setup(bot):
    await bot.add_cog(CogUpdater(bot))
