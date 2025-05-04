# askllmc.py  –  hybrid Qdrant + local‑Ollama   (Red‑DiscordBot cog)

import asyncio, glob, os, re, shutil, subprocess, uuid, json
from typing import List
import re
import discord, requests
from redbot.core import commands, Config
from redbot.core.data_manager import cog_data_path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, http

# ---------------------------------------------------------------------------
# helper – install missing PyPI packages on‑the‑fly (safe in Docker)
# ---------------------------------------------------------------------------
def _ensure_pkg(mod: str, pip_name: str | None = None):
    try:
        __import__(mod)
    except ModuleNotFoundError:
        subprocess.check_call(
            ["python", "-m", "pip", "install", pip_name or mod],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        __import__(mod)

_ensure_pkg("markdown")
_ensure_pkg("bs4", "beautifulsoup4")
_ensure_pkg("rank_bm25")            # <- hier
from markdown import markdown       # noqa: E402
from bs4 import BeautifulSoup       # noqa: E402
from rank_bm25 import BM25Okapi     # <- und hier

# ----------------------------------------------------------------------------
# LLMManager
# ----------------------------------------------------------------------------
class LLMManager(commands.Cog):
    """Interact with a local Ollama LLM through a Qdrant knowledge base."""

    # --------------------------------------------------------------------
    # init
    # --------------------------------------------------------------------
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=9876543210123)
        self.config.register_global(
            model="gemma3:12b",
            api_url="http://192.168.10.5:11434",
            qdrant_url="http://192.168.10.5:6333",
            auto_channels=[]
        )

        self.collection = "fus_wiki"

        # stronger model (768‑Dim)
        self.embedder = SentenceTransformer("all-mpnet-base-v2")
        self.vec_dim = self.embedder.get_sentence_embedding_dimension()

        self.q_client: QdrantClient | None = None
        self._last_manual_id: int | None = None  # last !llmknow id (session)
        # BM25-Retriever (auf Bedarf initialisiert)
        self.bm25: BM25Okapi | None = None
        self._bm25_pts: List = []
    # --------------------------------------------------------------------
    # basic helpers
    # --------------------------------------------------------------------
    async def ensure_qdrant(self):
        if self.q_client is None:
            self.q_client = QdrantClient(url=await self.config.qdrant_url())

    def _vec(self, txt: str) -> List[float]:
        return self.embedder.encode(txt).tolist()

     # --------------------------------------------------------------------
    # embedded images for qdrant entry (ersetzt deine zweite _upsert_sync)
    # --------------------------------------------------------------------
    def _upsert_sync(self, tag: str, content: str, source: str) -> int:
        self._ensure_collection()
        # 1) Alle Markdown-Links extrahieren
        image_urls = re.findall(r'\[.*?\]\((https?://[^\s\)]+)\)', content)
        # 2) Text ohne Link-Markup
        text_only = re.sub(r'\[.*?\]\((https?://[^\s\)]+)\)', '', content).strip()
        # 3) Vektor & Payload zusammenbauen
        pid = uuid.uuid4().int & ((1 << 64) - 1)
        vec = self._vec(f"{tag}. {tag}. {text_only}")
        payload = {"tag": tag, "content": text_only, "source": source}
        if image_urls:
            payload["images"] = image_urls
        # 4) Upsert
        self.q_client.upsert(
            self.collection,
            [{"id": pid, "vector": vec, "payload": payload}],
        )
        return pid


    # --------------------------------------------------------------------
    # low‑level Qdrant helpers (sync → executor)
    # --------------------------------------------------------------------
    def _ensure_collection(self, force: bool = False):
        """Create / recreate collection so its vector-dim matches the embedder."""
        try:
            info = self.q_client.get_collection(self.collection)
            size = info.config.params.vectors.size  # type: ignore
            if size != self.vec_dim or force:
                raise ValueError("dim mismatch → recreate")
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
                wal_config={
                    "wal_capacity_mb": 1024,
                },
                payload_indexing_config={
                    "enable": True,
                    "field_schema": {
                        "tag": {"type": "keyword"},
                        "source": {"type": "keyword"},
                    },
                },
                compression_config={
                    "type": "ProductQuantization",
                    "params": {"segments": 8, "subvector_size": 2},
                },
            )
    # --------------------------------------------------------------------
    # collection reset
    # --------------------------------------------------------------------
    @commands.command()
    @commands.has_permissions(administrator=True)
    async def initcollection(self, ctx):
        await self.ensure_qdrant()
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self.q_client.recreate_collection(
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
                wal_config={
                    "wal_capacity_mb": 1024,
                },
                payload_indexing_config={
                    "enable": True,
                    "field_schema": {
                        "tag": {"type": "keyword"},
                        "source": {"type": "keyword"},
                    },
                },
                compression_config={
                    "type": "ProductQuantization",
                    "params": {"segments": 8, "subvector_size": 2},
                },
            ),
        )
        await ctx.send(f"Collection **{self.collection}** recreated.")

    def _collect_ids_sync(self, filt: dict) -> List[int]:
        ids, offset = [], None
        while True:
            pts, offset = self.q_client.scroll(
                self.collection,
                limit=1000,
                with_payload=False,
                scroll_filter=filt,
                offset=offset,
            )
            ids.extend(p.id for p in pts)
            if offset is None:
                break
        return ids
    # --------------------------------------------------------------------
    # Commands zum Hinzufügen/Entfernen/Bewegen von Bild-URLs
    # --------------------------------------------------------------------
    @commands.command(name="llmknowaddimg")
    @commands.has_permissions(administrator=True)
    async def llmknowaddimg(self, ctx, doc_id: int, url: str):
        """Fügt eine Bild-URL zum Payload eines Eintrags hinzu."""
        await self.ensure_qdrant()
        pts = self.q_client.retrieve(self.collection, [doc_id], with_payload=True)
        if not pts:
            return await ctx.send(f"Eintrag {doc_id} nicht gefunden.")
        payload = pts[0].payload or {}
        images = payload.get("images", [])
        if url in images:
            return await ctx.send("URL ist bereits hinterlegt.")
        images.append(url)
        # Nur das images-Feld aktualisieren
        self.q_client.set_payload(
            collection_name=self.collection,
            payload={"images": images},
            points=[doc_id],
        )
        await ctx.send(f"Bild-URL hinzugefügt zu Eintrag {doc_id}.")

    @commands.command(name="llmknowrmimg")
    @commands.has_permissions(administrator=True)
    async def llmknowrmimg(self, ctx, doc_id: int, url: str):
        """Entfernt eine Bild-URL aus einem bestehenden Eintrag."""
        await self.ensure_qdrant()
        pts = self.q_client.retrieve(self.collection, [doc_id], with_payload=True)
        if not pts:
            return await ctx.send(f"Eintrag {doc_id} nicht gefunden.")
        payload = pts[0].payload or {}
        images = payload.get("images", [])
        if url not in images:
            return await ctx.send("URL nicht vorhanden.")
        images.remove(url)
        self.q_client.set_payload(
            collection_name=self.collection,
            payload={"images": images},
            points=[doc_id],
        )
        await ctx.send(f"Bild-URL entfernt von Eintrag {doc_id}.")

    @commands.command(name="llmknowmvimg")
    @commands.has_permissions(administrator=True)
    async def llmknowmvimg(self, ctx, doc_id: int, from_pos: int, to_pos: int):
        """Verschiebt eine Bild-URL innerhalb des Payloads eines Eintrags."""
        await self.ensure_qdrant()
        pts = self.q_client.retrieve(self.collection, [doc_id], with_payload=True)
        if not pts:
            return await ctx.send(f"Eintrag {doc_id} nicht gefunden.")
        payload = pts[0].payload or {}
        images = payload.get("images", [])
        if not (1 <= from_pos <= len(images)):
            return await ctx.send(f"Ungültige from_pos: {from_pos}. Es gibt nur {len(images)} Bilder.")
        url = images.pop(from_pos - 1)
        to_pos = max(1, min(to_pos, len(images) + 1))
        images.insert(to_pos - 1, url)
        self.q_client.set_payload(
            collection_name=self.collection,
            payload={"images": images},
            points=[doc_id],
        )
        await ctx.send(f"Bild von Position {from_pos} nach {to_pos} verschoben.")

    # --------------------------------------------------------------------
    # learn command
    # --------------------------------------------------------------------
    @commands.command(name="learn")
    @commands.has_permissions(administrator=True)
    async def learn(self, ctx, num: int):
        """Create a knowledge entry from the last `num` chat messages."""
        await self.ensure_qdrant()
        loop = asyncio.get_running_loop()

        # 1) Fetch last messages (excluding bot messages)
        msgs = [m async for m in ctx.channel.history(limit=num+20)]
        user_msgs = [m.content for m in msgs if not m.author.bot]
        content = "\n".join(reversed(user_msgs[-num:]))

        # 2) Generate initial draft via LLM
        api, model = await self.config.api_url(), await self.config.model()
        draft_prompt = (
            "Please create a concise knowledge entry from the following chat messages:\n"
            f"{content}\n\nEntry:"
        )
        draft = await loop.run_in_executor(None, self._ollama_chat_sync, api, model, draft_prompt)

        # 3) Interactive loop: yes / no / edit
        while True:
            await ctx.send(f"**Draft:**\n```{draft}```\nReply with `yes`, `no`, or `edit`.")
            try:
                reply = await self.bot.wait_for(
                    "message",
                    check=lambda m: m.author == ctx.author and m.channel == ctx.channel and m.content.lower() in ["yes", "no", "edit"],
                    timeout=300
                )
            except asyncio.TimeoutError:
                return await ctx.send("⏱️ Timeout – learn process aborted.")
            cmd = reply.content.lower()
            if cmd == "no":
                return await ctx.send("❌ Entry discarded.")
            if cmd == "edit":
                await ctx.send("✏️ Please provide your feedback:")
                try:
                    feedback = await self.bot.wait_for(
                        "message",
                        check=lambda m: m.author == ctx.author and m.channel == ctx.channel,
                        timeout=300
                    )
                except asyncio.TimeoutError:
                    return await ctx.send("⏱️ Timeout – learn process aborted.")
                edit_prompt = (
                    "Please revise the following entry based on this feedback:\n"
                    f"Entry: {draft}\nFeedback: {feedback.content}\n\nRevised entry:"
                )
                draft = await loop.run_in_executor(None, self._ollama_chat_sync, api, model, edit_prompt)
                continue
            # cmd == "yes"
            break

        # 4) Generate tags
        tag_prompt = (
            "Generate relevant tags (comma-separated) for this entry:\n"
            f"{draft}\n\nTags:"
        )
        tags = await loop.run_in_executor(None, self._ollama_chat_sync, api, model, tag_prompt)

        # 5) Tags confirmation loop
        while True:
            await ctx.send(f"**Proposed tags:** {tags}\nReply with `yes`, `no`, or `edit`.")
            try:
                reply = await self.bot.wait_for(
                    "message",
                    check=lambda m: m.author == ctx.author and m.channel == ctx.channel and m.content.lower() in ["yes", "no", "edit"],
                    timeout=300
                )
            except asyncio.TimeoutError:
                return await ctx.send("⏱️ Timeout – learn process aborted.")
            cmd = reply.content.lower()
            if cmd == "no":
                return await ctx.send("❌ Learn process cancelled.")
            if cmd == "edit":
                await ctx.send("📝 Please enter new tags (comma-separated):")
                try:
                    tag_reply = await self.bot.wait_for(
                        "message",
                        check=lambda m: m.author == ctx.author and m.channel == ctx.channel,
                        timeout=300
                    )
                except asyncio.TimeoutError:
                    return await ctx.send("⏱️ Timeout – learn process aborted.")
                tags = tag_reply.content
                continue
            # cmd == "yes"
            break

        # 6) Save entry
        await loop.run_in_executor(None, self._upsert_sync, tags, draft, "manual")
        await ctx.send(f"✅ Entry saved with tags: {tags}")


    # --------------------------------------------------------------------
    # knowledge‑management commands
    # --------------------------------------------------------------------
    @commands.command()
    async def llmknow(self, ctx, tag: str, *, content: str):
        """Add manual knowledge under **tag**."""
        await self.ensure_qdrant()
        loop = asyncio.get_running_loop()
        new_id = await loop.run_in_executor(None, self._upsert_sync, tag.lower(), content, "manual")
        self._last_manual_id = new_id
        await ctx.send(f"Added manual info under '{tag.lower()}' (ID {new_id}).")

    @commands.command()
    async def llmknowshow(self, ctx):
        """Show stored entries (chunked ≤2000 chars), including any image URLs."""
        await self.ensure_qdrant()
        try:
            pts, _ = self.q_client.scroll(
                self.collection,
                with_payload=True,
                limit=1000
            )
        except http.exceptions.ResponseHandlingException as e:
            await ctx.send(f"⚠️ Error: Connection to database failed!: {e}")
            return
        if not pts:
            return await ctx.send("No knowledge entries stored.")
        hdr, ftr, maxlen = "```\n", "```", 2000
        cur, chunks = hdr, []
        for p in pts:
            pl = p.payload or {}
            snip = pl.get("content", "")[:200].replace("\n", " ")
            line = f"[{p.id}] ({pl.get('tag','NoTag')},{pl.get('source','?')}): {snip}"
            images = pl.get("images", [])
            if images:
                # make images clearly visible
                line += "\n  → Images: " + ", ".join(images)
            line += "\n"
            if len(cur) + len(line) > maxlen - len(ftr):
                chunks.append(cur + ftr)
                cur = hdr + line
            else:
                cur += line
        chunks.append(cur + ftr)
        for c in chunks:
            await ctx.send(c)

    @commands.command(name="llmknowclearimgs")
    @commands.has_permissions(administrator=True)
    async def llmknowclearimgs(self, ctx, doc_id: int = None):
        """
        Clears all image URLs from a specific entry or from all entries if no ID given.
        Usage: !llmknowclearimgs [doc_id]
        """
        await self.ensure_qdrant()
        if doc_id:
            pts = self.q_client.retrieve(self.collection, [doc_id], with_payload=True)
            if not pts:
                return await ctx.send(f"Entry {doc_id} not found.")
            # remove images field
            self.q_client.set_payload(
                collection_name=self.collection,
                payload={"images": []},
                points=[doc_id],
            )
            await ctx.send(f"🔄 removed all pictures from {doc_id}.")
        else:
            # find all points with images
            all_pts, _ = self.q_client.scroll(self.collection, with_payload=True, limit=1000)
            ids_to_clear = [p.id for p in all_pts if p.payload and p.payload.get("images")]
            if not ids_to_clear:
                return await ctx.send("No entries with images found.")
            for pid in ids_to_clear:
                self.q_client.set_payload(
                    collection_name=self.collection,
                    payload={"images": []},
                    points=[pid],
                )
            await ctx.send(f"🔄 Images from {len(ids_to_clear)} removed.")


    @commands.command()
    async def llmknowdelete(self, ctx, doc_id: int):
        await self.ensure_qdrant()
        await asyncio.get_running_loop().run_in_executor(None, lambda: self.q_client.delete(self.collection, [doc_id]))
        await ctx.send(f"Deleted entry {doc_id}.")

    @commands.command()
    async def llmknowdeletetag(self, ctx, tag: str):
        await self.ensure_qdrant()
        filt = {"must": [{"key": "tag", "match": {"value": tag.lower()}}]}
        ids = await asyncio.get_running_loop().run_in_executor(None, self._collect_ids_sync, filt)
        if ids:
            await asyncio.get_running_loop().run_in_executor(None, lambda: self.q_client.delete(self.collection, ids))
        await ctx.send(f"Deleted entries with tag '{tag.lower()}'.")

    @commands.command()
    async def llmknowdeletelast(self, ctx):
        if self._last_manual_id is None:
            return await ctx.send("No manual entry recorded this session.")
        await self.ensure_qdrant()
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.q_client.delete(self.collection, [self._last_manual_id])
        )
        await ctx.send(f"Deleted last manual entry (ID {self._last_manual_id}).")
        self._last_manual_id = None

    # --------------------------------------------------------------------
    # Autochannel
    # --------------------------------------------------------------------
    @commands.command(name="addautochannel")
    @commands.has_permissions(administrator=True)
    async def add_auto_channel(self, ctx, channel: discord.TextChannel):
        """Fügt einen Channel zur Liste hinzu, in dem der Bot automatisch antwortet."""
        chans = await self.config.auto_channels()
        if channel.id in chans:
            return await ctx.send(f"{channel.mention} ist bereits registriert.")
        chans.append(channel.id)
        await self.config.auto_channels.set(chans)
        await ctx.send(f"{channel.mention} wurde zur Auto-Reply-Liste hinzugefügt.")

    @commands.command(name="removeautochannel")
    @commands.has_permissions(administrator=True)
    async def remove_auto_channel(self, ctx, channel: discord.TextChannel):
        """Entfernt einen Channel aus der Auto-Reply-Liste."""
        chans = await self.config.auto_channels()
        if channel.id not in chans:
            return await ctx.send(f"{channel.mention} war nicht registriert.")
        chans.remove(channel.id)
        await self.config.auto_channels.set(chans)
        await ctx.send(f"{channel.mention} wurde aus der Liste entfernt.")

    @commands.command(name="listautochannels")
    async def list_auto_channels(self, ctx):
        """Zeigt alle Channels, in denen der Bot automatisch antwortet."""
        chans = await self.config.auto_channels()
        if not chans:
            return await ctx.send("Keine Auto-Reply-Channels konfiguriert.")
        mentions = [f"<#{cid}>" for cid in chans]
        await ctx.send("Auto-Reply aktiv in: " + ", ".join(mentions))

    # --------------------------------------------------------------------
    # GitHub‑Wiki import  (unchanged)
    # --------------------------------------------------------------------
    @commands.command()
    async def importwiki(self, ctx, repo: str = "https://github.com/Kvitekvist/FUS.wiki.git"):
        await self.ensure_qdrant()
        data_dir = str(cog_data_path(self)); os.makedirs(data_dir, exist_ok=True)
        clone_dir = os.path.join(data_dir, "wiki")

        # … (Repo klonen/aktualisieren wie gehabt) …

        md_files = glob.glob(os.path.join(clone_dir, "*.md"))
        if not md_files:
            return await ctx.send("No markdown pages found – aborting.")

        def _import(fp: str):
            txt = open(fp, encoding="utf-8").read()
            # Markdown → HTML mit TOC-Erweiterung
            html = markdown(txt, extensions=["extra", "toc"])
            soup = BeautifulSoup(html, "html.parser")

            # Entferne „Back to Readme“-Links und ähnliche Navi-Leichen
            for a in soup.find_all("a"):
                if "Back to" in a.get_text():
                    a.decompose()

            # Splitte in Sektionen nach Überschriften
            for header in soup.find_all(re.compile(r"^h[1-6]$")):
                tag = header.get_text(strip=True)
                content_parts = []
                for sib in header.next_siblings:
                    if sib.name and re.match(r"^h[1-6]$", sib.name):
                        break
                    content_parts.append(str(sib))
                section_html = "".join(content_parts)
                section_text = BeautifulSoup(section_html, "html.parser").get_text(" ", strip=True)
                if section_text:
                    self._upsert_sync(tag, section_text, "wiki")

        loop = asyncio.get_running_loop()
        for fp in md_files:
            await loop.run_in_executor(None, _import, fp)
        await ctx.send(f"Wiki import done ({len(md_files)} pages).")

    # --------------------------------------------------------------------
    # Ollama helper (blocking → executor)
    # --------------------------------------------------------------------
    def _ollama_chat_sync(self, api: str, model: str, prompt: str) -> str:
        r = requests.post(
            f"{api}/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "")

    # --------------------------------------------------------------------
    # safe search (handles dim‑errors)
    # --------------------------------------------------------------------
    def _safe_search(self, *args, **kwargs):
        try:
            return self.q_client.search(*args, **kwargs)
        except http.exceptions.UnexpectedResponse as e:
            if "Vector dimension error" in str(e):
                # recreate collection & try again
                self._ensure_collection(force=True)
                return self.q_client.search(*args, **kwargs)
            raise
        
    # --------------------------------------------------------------------
    # main Q&A
    # --------------------------------------------------------------------
    async def _answer(self, question: str) -> str:
        await self.ensure_qdrant()
        loop = asyncio.get_running_loop()

        # --- BM25-Retrieval aufbauen (einmal) und ausführen ---
        if self.bm25 is None:
            pts, _ = self.q_client.scroll(self.collection, with_payload=True, limit=10000)
            if not pts:
                bm25_hits = []
            else:
                docs = [p.payload.get("content", "") for p in pts]
                tokenized = [doc.lower().split() for doc in docs]
                try:
                    self.bm25 = BM25Okapi(tokenized)
                    self._bm25_pts = pts
                except ZeroDivisionError:
                    self.bm25 = None
                    bm25_hits = []
                else:
                    tokenized_q = question.lower().split()
                    bm25_scores = self.bm25.get_scores(tokenized_q)
                    top_bm25 = sorted(
                        zip(self._bm25_pts, bm25_scores),
                        key=lambda x: x[1],
                        reverse=True
                    )[:20]
                    bm25_hits = [p for p, _ in top_bm25]
        else:
            tokenized_q = question.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_q)
            top_bm25 = sorted(
                zip(self._bm25_pts, bm25_scores),
                key=lambda x: x[1],
                reverse=True
            )[:20]
            bm25_hits = [p for p, _ in top_bm25]

        # ---------- (A) Heuristische Query-Erweiterung -----------------
        aug_q = question
        ql = question.lower()
        if "virtual" in ql and "desktop" in ql and "resolution" not in ql:
            aug_q += " resolution"
        # --------------------------------------------------------------

        # ---------- (B) Keyword- / Tag-Filter -------------------------
        clean_q = re.sub(r"[^\w\s]", " ", aug_q.lower())
        kws = [w for w in clean_q.split() if len(w) > 2]
        tag_filter = (
            {"should": (
                [{"key": "tag",     "match": {"value": k}} for k in kws] +
                [{"key": "content", "match": {"value": k}} for k in kws]
            )}
            if kws else None
        )
        # --------------------------------------------------------------

        # ---------- (C) Manual-Vektor-Suche ---------------------------
        manual_filter = {"must": [{"key": "source", "match": {"value": "manual"}}]}
        manual_vec_hits = self._safe_search(
            self.collection,
            query_vector=self._vec(aug_q),
            limit=20,
            query_filter=manual_filter,
        )
        if manual_vec_hits:
            hits = manual_vec_hits
        else:
            # Wiki-Fallback: Keyword-Filter
            tag_hits = []
            if tag_filter:
                tag_hits, _ = self.q_client.scroll(
                    self.collection,
                    with_payload=True,
                    limit=20,
                    scroll_filter=tag_filter,
                )
            # Wiki-Fallback: Vector-Suche
            vec_hits = self._safe_search(
                self.collection,
                query_vector=self._vec(aug_q),
                limit=20,
            )
            # Merge Wiki-Treffer
            hit_dict = {h.id: h for h in vec_hits}
            for h in tag_hits + bm25_hits:
                hit_dict.setdefault(h.id, h)
            hits = list(hit_dict.values())

        if not hits:
            return "No relevant information found."

        # ---------- (D) Kombinieren & Re-Rank -------------------------
        numbered = "\n\n".join(f"#{i}\n{h.payload.get('content','')}" for i, h in enumerate(hits))
        rank_prompt = (
            f"User question: {question}\n\n"
            "Below are context snippets (#0 …). Rate relevance 0-10.\n"
            "Return EXACT JSON: {\"scores\":[…]}\n\n"
            f"{numbered}"
        )
        api, model = await self.config.api_url(), await self.config.model()
        rank = await loop.run_in_executor(None, self._ollama_chat_sync, api, model, rank_prompt)
        try:
            scores = json.loads(rank)["scores"]
        except Exception:
            scores = [10] * len(hits)
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:5]
        # Treffer merken, damit wir später die images holen können
        self._last_ranked_hits = [h for h, _ in ranked]


        # ---------- (E) Token-Budget schonen: nur Top 5 & max. 200 Wörter ----------
        ctx = "\n\n".join(
            " ".join(h.payload.get("content", "").split()[:200])
            for h, _ in ranked
        )

        final_prompt = (
            "Use **only** the facts below to answer. If the facts are insufficient, say so.\n\n"
            f"### Facts ###\n{ctx}\n\n"
            f"### Question ###\n{question}\n\n"
            "### Answer ###"
        )
        return await loop.run_in_executor(None, self._ollama_chat_sync, api, model, final_prompt)


    # --------------------------------------------------------------------
    # public command + mention hook
    # --------------------------------------------------------------------
    @commands.command(name="askllm")
    async def askllm_cmd(self, ctx, *, question: str):
        async with ctx.typing():
            ans = await self._answer(question)
        await ctx.send(ans)
        # Bilder aus den zuletzt gerankten Treffern
        for h in getattr(self, "_last_ranked_hits", []):
            for url in h.payload.get("images", []):
                embed = discord.Embed()
                embed.set_image(url=url)
                await ctx.send(embed=embed)


    @commands.Cog.listener()
    async def on_message(self, m: discord.Message):
        if m.author.bot or not m.guild:
            return
        autolist = await self.config.auto_channels()
        if self.bot.user.mentioned_in(m) or m.content.startswith("!askllm"):
            q = m.clean_content.replace(f"@{self.bot.user.display_name}", "").strip()
        elif m.channel.id in autolist:
            q = m.content.strip()
        else:
            return
        if not q:
            return
        try:
            async with m.channel.typing():
                ans = await self._answer(q)
        except http.exceptions.ResponseHandlingException as e:
            return await m.channel.send(f"⚠️ Could not connect: {e}")
        await m.channel.send(ans)
        # Bilder senden
        for h in getattr(self, "_last_ranked_hits", []):
            for url in h.payload.get("images", []):
                embed = discord.Embed()
                embed.set_image(url=url)
                await m.channel.send(embed=embed)

    # --------------------------------------------------------------------
    # simple setters
    # --------------------------------------------------------------------
    @commands.command()
    async def setmodel(self, ctx, model):
        await self.config.model.set(model); await ctx.send(f"Model set to {model}")

    @commands.command()
    async def setapi(self, ctx, url):
        await self.config.api_url.set(url.rstrip("/")); await ctx.send("API URL updated")

    @commands.command()
    async def setqdrant(self, ctx, url):
        await self.config.qdrant_url.set(url.rstrip("/"))
        self.q_client = None; await ctx.send("Qdrant URL updated")  # reconnect next time

    # --------------------------------------------------------------------
    # on_ready
    # --------------------------------------------------------------------
    @commands.Cog.listener()
    async def on_ready(self):
        print("LLMManager cog loaded.")


async def setup(bot):
    await bot.add_cog(LLMManager(bot))
