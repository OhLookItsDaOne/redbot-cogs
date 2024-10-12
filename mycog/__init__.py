from .mycog import ForumPostNotifier

async def setup(bot):
    await bot.add_cog(ForumPostNotifier(bot))
