from .spam_out import ChannelGuard

async def setup(bot):
    await bot.add_cog(ChannelGuard(bot))
