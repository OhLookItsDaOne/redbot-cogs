from .askllmc import LLMManager

async def setup(bot):
    await bot.add_cog(LLMManager(bot))
