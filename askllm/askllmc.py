# askllmc.py  –  hybrid Qdrant + local-Ollama   (Red-DiscordBot cog)

import asyncio
import glob
import os
import re
import subprocess
import uuid
import json
from typing import List

import discord
import requests
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, http
from rank_bm25 import BM25Okapi

def _ensure_pkg(mod: str, pip_name: str | None = None):
    """Install missing PyPI packages on-the-fly (safe in Docker)."""
    try:
        __import__(mod)
    except ModuleNotFoundError:
        subprocess.check_call(
            ["python", "-m", "pip", "install", pip_name or mod],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        __import__(mod)


# ensure bs4, markdown, rank_bm25, etc.
_ensure_pkg("markdown")
_ensure_pkg("bs4", "beautifulsoup4")
_ensure_pkg("rank_bm25")


class LLMManager(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9876543210123)
        self.config.register_global(
            model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333",
            auto_channels=[],
            threshold=0.6,  # default token‐overlap threshold
            vector_threshold=0.2,
        )

        self.collection = "fus_wiki"
        self.embedder = SentenceTransformer("all-mpnet-base-v2")
        self.vec_dim = self.embedder.get_sentence_embedding_dimension()
        self.overlap_weight = 0.4  # Anteil, den Wort‑Überlappung am Gesamtscore hat
        self.q_client: QdrantClient | None = None
        self.bm25: BM25Okapi | None = None
        self._bm25_pts: List = []
        self._last_ranked_hits: List = []
        self._last_manual_id: int | None = None

    async def ensure_qdrant(self):
        if self.q_client is None:
            url = await self.config.qdrant_url()
            self.q_client = QdrantClient(url=url)

    def _vec(self, txt: str) -> List[float]:
        return self.embedder.encode(txt).tolist()

    async def _get_recent_context(
        self,
        channel: discord.TextChannel,
        *,
        before: discord.Message | None = None,
        n: int = 3,
    ) -> str:
        """Gibt bis zu `n` letzte *menschliche* Nachrichten als String zurück."""
        lines: list[str] = []
        async for m in channel.history(limit=n * 4, before=before):
            if m.author.bot:
                continue
            lines.append(m.content.strip())
            if len(lines) >= n:
                break
        lines.reverse()                     # chronologische Reihenfolge
        return "\n".join(lines)


    def _upsert_sync(self, tag: str, content: str, source: str) -> int:
        """Insert a new knowledge entry, extracting any inline images."""
        self._ensure_collection()
        # extract only true image URLs from markdown ![...](url.png|jpg|gif|webp)
        image_urls = re.findall(
            r'!\[.*?\]\((https?://[^\s)]+\.(?:png|jpe?g|gif|webp)(?:\?[^\s)]*)?)\)',
            content,
            flags=re.IGNORECASE,
        )
        # strip out image markup, keep normal links inline as "text: URL"
        txt = re.sub(r'!\[.*?\]\(https?://[^\s)]+\)', '', content)
        txt = re.sub(r'\[([^\]]+)\]\((https?://[^\s)]+)\)', r'\1: \2', txt).strip()

        pid = uuid.uuid4().int & ((1 << 64) - 1)
        vec = self._vec(f"{tag}. {tag}. {txt}")
        payload = {"tag": tag, "content": txt, "source": source}
        if image_urls:
            payload["images"] = image_urls

        self.q_client.upsert(
            self.collection,
            [{"id": pid, "vector": vec, "payload": payload}],
        )
        return pid

    def _ensure_collection(self, force: bool = False):
        """Create or recreate the Qdrant collection to match our embedding dim."""
        try:
            info = self.q_client.get_collection(self.collection)
            size = info.config.params.vectors.size  # type: ignore
            if size != self.vec_dim or force:
                raise ValueError("Dimension mismatch")
        except Exception:
            self.q_client.recreate_collection(
                collection_name=self.collection,
                vectors_config={
                    "size": self.vec_dim,
                    "distance": "Cosine",
                    "hnsw_config": {"m": 16, "ef_construct": 200},
                },
                optimizers_config={
                    "default_segment_number": 4,
                    "indexing_threshold": 256,
                },
                wal_config={"wal_capacity_mb": 1024},
                payload_indexing_config={
                    "enable": True,
                    "field_schema": {
                        "tag": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "content": {"type": "text"},
                    },
                },
                compression_config={
                    "type": "ProductQuantization",
                    "params": {"segments": 8, "subvector_size": 2},
                },
            )

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def initcollection(self, ctx: commands.Context):
        """Recreate the Qdrant collection from scratch."""
        await self.ensure_qdrant()
        await ctx.send(f"Recreating collection **{self.collection}** …")
        await asyncio.get_running_loop().run_in_executor(None, lambda: self._ensure_collection(force=True))
        await ctx.send("✅ Collection recreated.")

    @commands.command(name="llmknowaddimg")
    @commands.has_permissions(administrator=True)
    async def llmknowaddimg(self, ctx: commands.Context, doc_id: int, url: str):
        """Add an image URL to an existing entry (must be a real image file)."""
        if not re.search(r'\.(?:png|jpe?g|gif|webp)(?:\?.*)?$', url, flags=re.IGNORECASE):
            return await ctx.send("⚠️ That URL doesn't look like an image.")
        await self.ensure_qdrant()
        pts = self.q_client.retrieve(self.collection, [doc_id], with_payload=True)
        if not pts:
            return await ctx.send(f"No entry with ID {doc_id}.")
        payload = pts[0].payload or {}
        images = payload.get("images", [])
        if url in images:
            return await ctx.send("⚠️ Image already stored.")
        images.append(url)
        self.q_client.set_payload(
            collection_name=self.collection,
            payload={"images": images},
            points=[doc_id],
        )
        await ctx.send("✅ Image added.")

    @commands.command(name="llmknowrmimg")
    @commands.has_permissions(administrator=True)
    async def llmknowrmimg(self, ctx: commands.Context, doc_id: int, url: str):
        """Remove a stored image URL from an entry."""
        await self.ensure_qdrant()
        pts = self.q_client.retrieve(self.collection, [doc_id], with_payload=True)
        if not pts:
            return await ctx.send(f"No entry with ID {doc_id}.")
        payload = pts[0].payload or {}
        images = payload.get("images", [])
        if url not in images:
            return await ctx.send("⚠️ URL not found.")
        images.remove(url)
        self.q_client.set_payload(
            collection_name=self.collection,
            payload={"images": images},
            points=[doc_id],
        )
        await ctx.send("✅ Image removed.")

    @commands.command(name="llmknowmvimg")
    @commands.has_permissions(administrator=True)
    async def llmknowmvimg(self, ctx: commands.Context, doc_id: int, from_pos: int, to_pos: int):
        """Reorder an image URL in an entry's image list."""
        await self.ensure_qdrant()
        pts = self.q_client.retrieve(self.collection, [doc_id], with_payload=True)
        if not pts:
            return await ctx.send(f"No entry with ID {doc_id}.")
        imgs = pts[0].payload.get("images", [])
        if not (1 <= from_pos <= len(imgs)):
            return await ctx.send("⚠️ 'from' position out of range.")
        url = imgs.pop(from_pos - 1)
        to_pos = max(1, min(to_pos, len(imgs) + 1))
        imgs.insert(to_pos - 1, url)
        self.q_client.set_payload(
            collection_name=self.collection,
            payload={"images": imgs},
            points=[doc_id],
        )
        await ctx.send("✅ Image reordered.")

    @commands.command(name="learn")
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx: commands.Context, num: int):
        """Build a new knowledge entry from the last `num` messages."""
        await self.ensure_qdrant()
        loop = asyncio.get_running_loop()

        # fetch last messages (skip bot messages)
        msgs = [m async for m in ctx.channel.history(limit=num + 20)]
        text_msgs = [m.content for m in msgs if not m.author.bot]
        excerpt = "\n".join(reversed(text_msgs[-num:]))

        # initial draft
        api, model = await self.config.api_url(), await self.config.model()
        draft_prompt = (
            "Create a concise knowledge entry under 1500 chars from these messages:\n"
            f"{excerpt}\n\nEntry:"
        )
        draft = await loop.run_in_executor(None, self._ollama_chat_sync, api, model, draft_prompt)

        # interactive yes/no/edit loop
        while True:
            preview = draft if len(draft) <= 1500 else draft[:1500] + "…"
            await ctx.send(f"**Draft:**\n```{preview}```\nReply with `yes`, `no`, or `edit`.")
            try:
                reply = await self.bot.wait_for(
                    "message",
                    check=lambda m: m.author == ctx.author and m.channel == ctx.channel and m.content.lower() in ["yes", "no", "edit"],
                    timeout=300,
                )
            except asyncio.TimeoutError:
                return await ctx.send("⏱️ Timeout—aborting.")
            cmd = reply.content.lower()
            if cmd == "no":
                return await ctx.send("❌ Discarded.")
            if cmd == "edit":
                await ctx.send("✏️ Please provide feedback:")
                try:
                    fb = await self.bot.wait_for(
                        "message",
                        check=lambda m: m.author == ctx.author and m.channel == ctx.channel,
                        timeout=300,
                    )
                except asyncio.TimeoutError:
                    return await ctx.send("⏱️ Timeout—aborting.")
                edit_prompt = (
                    "Revise this entry under 1500 chars given the feedback:\n"
                    f"Entry: {draft}\nFeedback: {fb.content}\n\nRevised entry:"
                )
                draft = await loop.run_in_executor(None, self._ollama_chat_sync, api, model, edit_prompt)
                continue
            break  # yes

        # generate tags
        tag_prompt = f"Generate comma-separated tags for this entry:\n{draft}\n\nTags:"
        tags = await loop.run_in_executor(None, self._ollama_chat_sync, api, model, tag_prompt)

        # confirm tags
        while True:
            await ctx.send(f"**Proposed tags:** {tags}\nReply `yes`, `no`, or `edit`.")
            try:
                reply = await self.bot.wait_for(
                    "message",
                    check=lambda m: m.author == ctx.author and m.channel == ctx.channel and m.content.lower() in ["yes", "no", "edit"],
                    timeout=300,
                )
            except asyncio.TimeoutError:
                return await ctx.send("⏱️ Timeout—aborting.")
            cmd = reply.content.lower()
            if cmd == "no":
                return await ctx.send("❌ Cancelled.")
            if cmd == "edit":
                await ctx.send("📝 Enter new tags (comma-separated):")
                try:
                    tr = await self.bot.wait_for(
                        "message",
                        check=lambda m: m.author == ctx.author and m.channel == ctx.channel,
                        timeout=300,
                    )
                except asyncio.TimeoutError:
                    return await ctx.send("⏱️ Timeout—aborting.")
                tags = tr.content
                continue
            break

        # save entry
        await loop.run_in_executor(None, self._upsert_sync, tags, draft, "manual")
        await ctx.send(f"✅ Saved with tags: {tags}")

    @commands.command(name="setvectorthreshold")
    @commands.has_permissions(administrator=True)
    async def setvectorthreshold(self, ctx, value: float):
        if not 0.0 <= value <= 1.0:
            return await ctx.send("⚠️ Threshold must be between 0.0 and 1.0")
        await self.config.vector_threshold.set(value)
        await ctx.send(f"✅ Vector threshold set to {value:.0%}")

    async def _answer(self, question: str, context: str | None = None) -> str:
        await self.ensure_qdrant()
        loop = asyncio.get_running_loop()

        query_text = f"{context}\n{question}" if context else question
        q_vec = await loop.run_in_executor(None, self._vec, query_text)

        raw_hits = await loop.run_in_executor(
            None,
            lambda: self.q_client.search(
                collection_name=self.collection,
                query_vector=q_vec,
                limit=40,                # etwas breiter suchen
                with_payload=True,
            ),
        )

        def _rank(hit):
            base = hit.score or 0.0
            overlap = self._token_overlap(query_text, hit.payload.get("content", ""))
            bonus = 0.1 if hit.payload.get("source") == "manual" else 0.0
            return base + overlap * self.overlap_weight + bonus

        chosen = sorted(raw_hits, key=_rank, reverse=True)[:5]
        self._last_ranked_hits = chosen
        if not chosen:
            return "No relevant information found."

        facts = "\n\n".join(f"#{i}\n{h.payload['content']}" for i, h in enumerate(chosen))
        prompt = (
            "Below are numbered facts. Answer the question using ONLY these facts.\n"
            "At the end, list which fact indices you used, e.g. Used: [#0,#2].\n\n"
            f"### Facts ###\n{facts}\n\n"
            f"### Question ###\n{question}\n\n"
            "### Answer ###"
        )
        api, model = await self.config.api_url(), await self.config.model()
        return await loop.run_in_executor(
            None, self._ollama_chat_sync, api, model, prompt
        )

    @commands.command()
    async def llmknow(self, ctx: commands.Context, tag: str, *, content: str):
        """Add a manual knowledge entry."""
        await self.ensure_qdrant()
        new_id = await asyncio.get_running_loop().run_in_executor(None, self._upsert_sync, tag.lower(), content, "manual")
        await ctx.send(f"Added entry under '{tag}' (ID {new_id}).")
        self._last_manual_id = new_id

    @commands.command(name="llmknowshow")
    async def llmknowshow(self, ctx: commands.Context):
        """List all entries, showing full content and image URLs."""
        await self.ensure_qdrant()
        pts, _ = self.q_client.scroll(self.collection, with_payload=True, limit=1000)
        if not pts:
            return await ctx.send("No entries stored.")
        out = []
        for p in pts:
            pl = p.payload or {}
            line = f"[{p.id}] ({pl.get('tag')}, {pl.get('source')}): {pl.get('content')}"
            imgs = pl.get("images", [])
            if imgs:
                line += "\n  → Images:\n" + "\n".join(f"    • {u}" for u in imgs)
            out.append(line)
        for chunk in ("\n\n".join(out)[i:i+1900] for i in range(0, len("\n\n".join(out)), 1900)):
            await ctx.send(f"```{chunk}```")

    @commands.command(name="llmknowclearimgs")
    @commands.has_permissions(administrator=True)
    async def llmknowclearimgs(self, ctx: commands.Context, doc_id: int | None = None):
        """Clear image URLs from one entry or all entries."""
        await self.ensure_qdrant()
        if doc_id:
            pts = self.q_client.retrieve(self.collection, [doc_id], with_payload=True)
            if not pts:
                return await ctx.send(f"No entry {doc_id}.")
            self.q_client.set_payload(self.collection, {"images": []}, [doc_id])
            return await ctx.send(f"Cleared images from {doc_id}.")
        pts, _ = self.q_client.scroll(self.collection, with_payload=True, limit=1000)
        to_clear = [p.id for p in pts if p.payload.get("images")]
        for pid in to_clear:
            self.q_client.set_payload(self.collection, {"images": []}, [pid])
        await ctx.send(f"Cleared images from {len(to_clear)} entries.")

    @commands.command()
    async def llmknowdelete(self, ctx: commands.Context, doc_id: int):
        """Delete a single entry by ID."""
        await self.ensure_qdrant()
        self.q_client.delete(self.collection, [doc_id])
        await ctx.send(f"Deleted entry {doc_id}.")

    @commands.command()
    async def llmknowdeletetag(self, ctx: commands.Context, tag: str):
        """Delete all entries with a given tag."""
        await self.ensure_qdrant()
        filt = {"must": [{"key": "tag", "match": {"value": tag.lower()}}]}
        pts, _ = self.q_client.scroll(self.collection, with_payload=False, limit=1000, scroll_filter=filt)
        ids = [p.id for p in pts]
        if ids:
            self.q_client.delete(self.collection, ids)
        await ctx.send(f"Deleted {len(ids)} entries tagged '{tag}'.")

    @commands.command()
    async def llmknowdeletelast(self, ctx: commands.Context):
        """Delete the last added manual entry."""
        if self._last_manual_id is None:
            return await ctx.send("No recent manual entry.")
        await self.ensure_qdrant()
        self.q_client.delete(self.collection, [self._last_manual_id])
        await ctx.send(f"Deleted last entry {self._last_manual_id}.")
        self._last_manual_id = None

    @commands.command(name="addautochannel")
    @commands.has_permissions(administrator=True)
    async def add_auto_channel(self, ctx: commands.Context, channel: discord.TextChannel):
        """Enable auto-reply in this channel."""
        chans = await self.config.auto_channels()
        if channel.id in chans:
            return await ctx.send("Already enabled there.")
        chans.append(channel.id)
        await self.config.auto_channels.set(chans)
        await ctx.send(f"Auto-reply enabled in {channel.mention}.")

    @commands.command(name="removeautochannel")
    @commands.has_permissions(administrator=True)
    async def remove_auto_channel(self, ctx: commands.Context, channel: discord.TextChannel):
        """Disable auto-reply in this channel."""
        chans = await self.config.auto_channels()
        if channel.id not in chans:
            return await ctx.send("Not enabled there.")
        chans.remove(channel.id)
        await self.config.auto_channels.set(chans)
        await ctx.send(f"Auto-reply disabled in {channel.mention}.")

    @commands.command()
    @commands.has_permissions(administrator=True)
    async def setthreshold(self, ctx, value: float):
        """
        Set the image‐overlap threshold. 0.0–1.0 (e.g. 0.6 for 60%).
        """
        if not 0.0 <= value <= 1.0:
            return await ctx.send("⚠️ Threshold must be between 0.0 and 1.0")
        await self.config.threshold.set(value)
        await ctx.send(f"✅ Image‐overlap threshold set to {value:.0%}")

    @commands.command(name="listautochannels")
    async def list_auto_channels(self, ctx: commands.Context):
        """List all channels with auto-reply enabled."""
        chans = await self.config.auto_channels()
        if not chans:
            return await ctx.send("No auto-reply channels set.")
        mentions = ", ".join(f"<#{cid}>" for cid in chans)
        await ctx.send(f"Auto-reply active in: {mentions}")

    def _ollama_chat_sync(self, api: str, model: str, prompt: str) -> str:
        """Sync call to Ollama's chat endpoint, with robust JSON parsing."""
        r = requests.post(
            f"{api.rstrip('/')}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,  # ensure single‐object JSON
            },
            timeout=120,
        )
        r.raise_for_status()
        try:
            return r.json().get("message", {}).get("content", "")
        except (ValueError, json.JSONDecodeError):
            # fallback: try parsing the last valid JSON object in the response
            for line in r.text.strip().splitlines()[::-1]:
                try:
                    obj = json.loads(line)
                    return obj.get("message", {}).get("content", "")
                except json.JSONDecodeError:
                    continue
            # if all else fails, return raw text
            return r.text
        
    def _safe_search(self, **kwargs):
        """Perform a Qdrant search, auto-creating collection on dimension errors."""
        kwargs.setdefault("with_payload", True)
        try:
            return self.q_client.search(**kwargs)
        except http.exceptions.UnexpectedResponse as e:
            if "Vector dimension error" in str(e):
                self._ensure_collection(force=True)
                return self.q_client.search(**kwargs)
            raise

    def _token_overlap(self, a: str, b: str) -> float:
        """Prozentuale Token‑Überschneidung von *b* in *a* (0.0 – 1.0)."""
        import re
        ta = set(re.findall(r"\w+", a.lower()))
        tb = set(re.findall(r"\w+", b.lower()))
        if not tb:
            return 0.0
        return len(ta & tb) / len(tb)
    
    @commands.command(name="askllm")
    async def askllm_cmd(self, ctx: commands.Context, *, question: str):
        ctx_txt = await self._get_recent_context(ctx.channel, before=ctx.message)
        async with ctx.typing():
            ans = await self._answer(question, ctx_txt)
        await ctx.send(ans)

        # ----- Bilder anhängen, falls relevant ---------------------------------
        used = re.findall(r"#\d+", ans.split("Used:", 1)[-1])
        thr = await self.config.vector_threshold()
        for idx_str in used:
            idx = int(idx_str.lstrip("#"))
            if idx >= len(self._last_ranked_hits):
                continue
            hit = self._last_ranked_hits[idx]
            if hit.payload.get("source") == "manual" and (hit.score or 0.0) >= thr:
                for url in hit.payload.get("images", []):
                    await ctx.send(embed=discord.Embed().set_image(url=url))

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Auto‑Antwort, falls erwähnt oder Kanal auf Auto steht – mit Verlauf."""
        if message.author.bot or not message.guild:
            return

        autolist = await self.config.auto_channels()
        if self.bot.user.mentioned_in(message):
            q = message.clean_content.replace(
                f"@{self.bot.user.display_name}", ""
            ).strip()
        elif message.channel.id in autolist:
            q = message.content.strip()
        else:
            return

        if not q:
            return

        ctx_txt = await self._get_recent_context(message.channel, before=message)

        async with message.channel.typing():
            ans = await self._answer(q, ctx_txt)
        await message.channel.send(ans)

        thr = await self.config.threshold()
        for hit in self._last_ranked_hits:
            if hit.payload.get("source") != "manual":
                continue
            entry = hit.payload.get("content", "")
            if self._token_overlap(ans, entry) >= thr:
                for url in hit.payload.get("images", []):
                    await message.channel.send(embed=discord.Embed().set_image(url=url))

    @commands.command()
    async def setmodel(self, ctx: commands.Context, model: str):
        """Set the Ollama model name."""
        await self.config.model.set(model)
        await ctx.send(f"Model set to {model}")

    @commands.command()
    async def setapi(self, ctx: commands.Context, url: str):
        """Set the Ollama API URL."""
        await self.config.api_url.set(url.rstrip("/"))
        await ctx.send("API URL updated")

    @commands.command()
    async def setqdrant(self, ctx: commands.Context, url: str):
        """Set the Qdrant URL."""
        await self.config.qdrant_url.set(url.rstrip("/"))
        self.q_client = None
        await ctx.send("Qdrant URL updated")

    @commands.Cog.listener()
    async def on_ready(self):
        """Confirm the cog has loaded."""
        print("LLMManager cog loaded.")


async def setup(bot: commands.Bot):
    await bot.add_cog(LLMManager(bot))
