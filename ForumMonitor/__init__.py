from .ForumMessage import ForumMessage

def setup(bot):
    bot.add_cog(ForumMessage(bot))
