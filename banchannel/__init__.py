from .banchannel import BanChannel

async def setup(bot):
    await bot.add_cog(BanChannel(bot))
