from .forward_msg import UnsupportedMessageForwarder

def setup(bot):
    bot.add_cog(UnsupportedMessageForwarder(bot))
