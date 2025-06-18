from .automod import AutoMod

async def setup(bot):
    await bot.add_cog(AutoMod(bot))

