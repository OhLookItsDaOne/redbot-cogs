from .mycog import ForumPostNotifier

def setup(bot):
    bot.add_cog(ForumPostNotifier(bot))