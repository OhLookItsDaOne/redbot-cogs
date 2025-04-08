from .LLMsummary import LLMSummary

def setup(bot):
    bot.add_cog(LLMSummary(bot))
