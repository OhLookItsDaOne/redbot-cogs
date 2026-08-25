import discord
from .messageforward import MessageForwarder, forward_to_support

async def setup(bot):
    bot.tree.add_command(forward_to_support)
    await bot.add_cog(MessageForwarder(bot))

async def teardown(bot):
    bot.tree.remove_command("Forward to Support", type=discord.AppCommandType.message)
