from .d1automod import D1AutoMod

async def setup(bot):
    await bot.add_cog(AutoMod(bot))

