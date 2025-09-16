from .fusrohcog import DeepSeekCog

async def setup(bot):
    await bot.add_cog(DeepSeekCog(bot))

# ──────────────────────────────────────────────────────────────
# GDPR / data‑collection disclosure (required by Red 3.5+)
# ──────────────────────────────────────────────────────────────
__red_end_user_data_statement__ = (
    "This cog stores user‑provided support snippets and chat history "
    "embeddings in your self‑hosted Qdrant instance. "
    "No personal data is sent to third‑party services."
)

