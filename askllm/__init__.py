from .askllmc import LLMManager

def setup(bot):
    bot.add_cog(LLMManager(bot))
