from __future__ import annotations
import asyncio
import json
import logging
import textwrap
import time
from typing import Any, Dict, List

import aiohttp
from redbot.core import Config, commands
from redbot.core.bot import Red
from redbot.core.commands import BadArgument

logger = logging.getLogger("red.fusrohcog")
DEFAULT_COLLECTION = "fusroh_support"


class QdrantClient:
    """Very small async wrapper around Qdrant’s REST API."""

    def __init__(self, url: str, collection: str = DEFAULT_COLLECTION):
        self.base = url.rstrip("/")
        self.collection = collection

    # ---------- helpers ----------
    async def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base}{path}"
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, **kwargs) as resp:
                if resp.status >= 400:
                    txt = await resp.text()
                    raise RuntimeError(f"Qdrant {method} {path} failed: {resp.status} {txt}")
                return await resp.json()

    # ---------- collection mgmt ----------
    async def _ensure_collection(self):
        try:
            await self._request("GET", f"/collections/{self.collection}")
        except RuntimeError:
            schema = {
                "vectors": {
                    "size": 768,  # Gemma embeddings size – adjust if you use another model
                    "distance": "Cosine",
                }
            }
            await self._request("PUT", f"/collections/{self.collection}", json=schema)
            logger.info("Created Qdrant collection %s", self.collection)

    # ---------- CRUD ops ----------
    async def upsert(self, point_id: int, vector: List[float], payload: Dict[str, Any]):
        await self._ensure_collection()
        data = {
            "points": [
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": payload,
                }
            ]
        }
        await self._request("PUT", f"/collections/{self.collection}/points", json=data)

    async def delete(self, point_id: int):
        data = {"points": [point_id]}
        await self._request("DELETE", f"/collections/{self.collection}/points", json=data)

    async def search(self, vector: List[float], limit: int = 5):
        body = {
            "vector": vector,
            "limit": limit,
            "with_payload": True,
            "score_threshold": 0.25,  # tune this – lower → more results, higher → stricter
        }
        resp = await self._request("POST", f"/collections/{self.collection}/points/search", json=body)
        return resp.get("result", [])

    async def scroll(self, limit: int = 10, offset: int = 0):
        params = {"limit": limit, "offset": offset, "with_payload": True}
        resp = await self._request("GET", f"/collections/{self.collection}/points", params=params)
        return resp.get("result", [])


class FusRohCog(commands.Cog):
    """Cog entry‑point – add with ``[p]load fusrohcog``"""

    def __init__(self, bot: Red):
        self.bot = bot
        # noinspection PyTypeChecker
        self.config = Config.get_conf(self, identifier=0xABCD1234)
        self.config.register_global(
            model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333",
            autotype_channels=[],  # List[int]
        )
        # ---------- runtime cache ----------
        self._qdrant: QdrantClient | None = None

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    async def _get_qdrant(self) -> QdrantClient:
        if self._qdrant is None:
            url = await self.config.qdrant_url()
            self._qdrant = QdrantClient(url)
        return self._qdrant

    async def _create_embedding(self, text: str) -> List[float]:
        url = await self.config.api_url()
        payload = {
            "model": await self.config.model(),
            "prompt": text,
            "raw": False,
            "stream": False,
            "format": "json",
            "options": {"type": "embedding"},
        }
        endpoint = f"{url.rstrip('/')}/api/embeddings"
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"Ollama embeddings error {resp.status}: {await resp.text()}")
                data = await resp.json()
        return data["embedding"]  # type: ignore[index]

    async def _generate_reply(self, context_messages: List[Dict[str, str]]) -> str:
        url = await self.config.api_url()
        system_prompt = textwrap.dedent(
            """
            You are a helpful support assistant for the SkyrimVR mod‑list community.
            * If you do not find a factual answer in the provided `Knowledge` section, say "I’m not sure.".
            * Do *not* invent features or steps – be concise and precise.
            * If the user asks about mod installation troubleshooting, list clear steps.
            """
        ).strip()
        payload = {
            "model": await self.config.model(),
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                *context_messages,
            ],
        }
        endpoint = f"{url.rstrip('/')}/api/chat"
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"Ollama chat error {resp.status}: {await resp.text()}")
                data = await resp.json()
        return data["message"]["content"]  # type: ignore[index]

    async def _chunk_send(self, ctx: commands.Context, text: str):
        """Split *text* into <=2000‑char chunks for Discord."""
        for i in range(0, len(text), 1990):
            await ctx.send(f"```{text[i:i + 1990]}```")

    @staticmethod
    def _ts_id() -> int:
        return int(time.time() * 1000)

    # -------------------------------------------------------------------
    # Commands
    # -------------------------------------------------------------------

    @commands.command(name="fusknow")
    @commands.cooldown(1, 10, commands.BucketType.user)
    async def fusknow(self, ctx: commands.Context, *, text: str):
        """Add *text* to the knowledge base."""
        vector = await self._create_embedding(text)
        point_id = self._ts_id()
        payload = {"text": text, "author": str(ctx.author), "ts": ctx.message.created_at.isoformat()}
        qd = await self._get_qdrant()
        await qd.upsert(point_id, vector, payload)
        await ctx.tick()
        await ctx.send(f"Knowledge saved with id `{point_id}`.")

    @commands.command(name="fusshow")
    async def fusshow(self, ctx: commands.Context, count: int = 10, offset: int = 0):
        """Show *count* entries starting at *offset*."""
        qd = await self._get_qdrant()
        entries = await qd.scroll(limit=count, offset=offset)
        if not entries:
            await ctx.send("No entries found.")
            return
        parts = []
        for p in entries:
            tid = p["id"]
            txt = p["payload"].get("text", "<no text>")
            parts.append(f"• **{tid}** – {txt[:150]}…")
        await self._chunk_send(ctx, "\n".join(parts))

    @commands.command(name="fusknowdel")
    async def fusknowdel(self, ctx: commands.Context, point_id: int):
        """Delete an entry by its *point_id*."""
        qd = await self._get_qdrant()
        try:
            await qd.delete(point_id)
        except RuntimeError as exc:
            raise BadArgument(str(exc))
        await ctx.tick()
        await ctx.send(f"Deleted knowledge id `{point_id}`.")

    @commands.command(name="learn")
    async def learn(self, ctx: commands.Context, count: int = 5):
        """Ingest the last *count* messages from this channel."""
        if count < 1 or count > 20:
            raise BadArgument("Count must be between 1 and 20.")
        messages = [m async for m in ctx.channel.history(limit=count + 1) if m.id != ctx.message.id]
        messages.reverse()  # oldest‑first
        joined = "\n".join(f"{m.author.display_name}: {m.clean_content}" for m in messages)
        vector = await self._create_embedding(joined)
        qd = await self._get_qdrant()
        pid = self._ts_id()
        await qd.upsert(pid, vector, {"text": joined, "learned": True})
        await ctx.send(f"Learned from {len(messages)} messages (id `{pid}`).")

    @commands.command(name="autotype")
    async def autotype(self, ctx: commands.Context, mode: str | None = None):
        """Without args – show current status.  Use *on* or *off* to toggle for this channel."""
        channel_id = ctx.channel.id
        autos = await self.config.autotype_channels()
        if mode is None:
            await ctx.send("Auto‑typing is currently **{}** for this channel.".format(
                "enabled" if channel_id in autos else "disabled"))
            return
        mode = mode.lower()
        if mode not in {"on", "off"}:
            raise BadArgument("Mode must be 'on' or 'off'.")
        if mode == "on":
            if channel_id not in autos:
                autos.append(channel_id)
                await self.config.autotype_channels.set(autos)
            await ctx.send("Enabled auto‑typing in this channel.")
        else:
            if channel_id in autos:
                autos.remove(channel_id)
                await self.config.autotype_channels.set(autos)
            await ctx.send("Disabled auto‑typing in this channel.")

    # -------------------------------------------------------------------
    # Listeners
    # -------------------------------------------------------------------

    @commands.Cog.listener()
    async def on_message_without_command(self, message):
        """Auto‑reply when appropriate."""
        # Ignore DMs, bots, and our own messages
        if not message.guild or message.author.bot or message.author == self.bot.user:
            return

        ctx = await self.bot.get_context(message)
        # If it *is* a command we already handled above.
        if ctx.valid:
            return

        channel_id = message.channel.id
        autos = await self.config.autotype_channels()
        should_reply = channel_id in autos or self.bot.user in message.mentions
        if not should_reply:
            return

        # ---- Build context for LLM
        # Pull latest 5 msgs including user’s message
        history = [m async for m in message.channel.history(limit=5)]
        history.reverse()
        chat_context = [
            {"role": "user" if m.author == message.author else "assistant", "content": m.clean_content}
            for m in history
        ]
        # Embed user message to search KB
        vector = await self._create_embedding(message.clean_content)
        qd = await self._get_qdrant()
        hits = await qd.search(vector, limit=5)
        knowledge_blurbs = [h["payload"]["text"] for h in hits]
        if knowledge_blurbs:
            kb_section = "\n\n".join(f"* {txt}" for txt in knowledge_blurbs)
            chat_context.append({"role": "system", "content": f"Knowledge:\n{kb_section}"})

        try:
            reply = await self._generate_reply(chat_context)
        except Exception as exc:  # broad – log and bail silently
            logger.exception("Failed to generate reply: %s", exc)
            return
        # Avoid spamming – skip if LLM decides it’s unsure.
        if "i’m not sure" in reply.lower():
            return
        # Finally send answer
        await message.channel.typing()
        await message.channel.send(reply)

    # -------------------------------------------------------------------
    # Cog setup helper (for Red’s load mechanism)
    # -------------------------------------------------------------------


async def setup(bot: Red):
    await bot.add_cog(FusRohCog(bot))
