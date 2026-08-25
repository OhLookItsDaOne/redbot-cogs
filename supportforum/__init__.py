from .supportforum import SupportForum

async def setup(bot):
    await bot.add_cog(SupportForum(bot))
