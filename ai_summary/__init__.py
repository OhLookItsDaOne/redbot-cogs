from .LLMsummary import LLMSummary

async def setup(bot):
    bot.add_cog(LLMSummary(bot))
