from .autoreply import AutoReply

async def setup(bot):
    await bot.add_cog(AutoReply(bot))
