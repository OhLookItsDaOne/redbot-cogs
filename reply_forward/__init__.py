from .forward_msg import UnsupportedMessageForwarder

async def setup(bot):
    await bot.add_cog(UnsupportedMessageForwarder(bot))
