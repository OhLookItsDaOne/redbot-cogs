from .imagespam import ImageSpam

async def setup(bot):
    await bot.add_cog(ImageSpam(bot))
