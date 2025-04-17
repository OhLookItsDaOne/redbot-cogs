# askllmc.py  –  hybrid Qdrant + local‑Ollama   (Red‑DiscordBot cog)

import asyncio, glob, os, re, shutil, subprocess, uuid, json
from typing import List

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
from markdown import markdown              # noqa:  E402 (imported after _ensure_pkg)
from bs4 import BeautifulSoup              # noqa:  E402

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
        )

        self.collection = "fus_wiki"

        # stronger model (768‑Dim)
        self.embedder = SentenceTransformer("all-mpnet-base-v2")
        self.vec_dim = self.embedder.get_sentence_embedding_dimension()

        self.q_client: QdrantClient | None = None
        self._last_manual_id: int | None = None  # last !llmknow id (session)

    # --------------------------------------------------------------------
    # basic helpers
    # --------------------------------------------------------------------
    async def ensure_qdrant(self):
        if self.q_client is None:
            self.q_client = QdrantClient(url=await self.config.qdrant_url())

    def _vec(self, txt: str) -> List[float]:
        return self.embedder.encode(txt).tolist()

    # --------------------------------------------------------------------
    # low‑level Qdrant helpers (sync → executor)
    # --------------------------------------------------------------------
    def _ensure_collection(self, force: bool = False):
        """Create / recreate collection so its vector‑dim matches the embedder."""
        try:
            info = self.q_client.get_collection(self.collection)
            size = info.config.params.vectors.size  # type: ignore
            if size != self.vec_dim or force:
                raise ValueError("dim mismatch → recreate")
        except Exception:
            self.q_client.recreate_collection(
                collection_name=self.collection,
                vectors_config={"size": self.vec_dim, "distance": "Cosine"},
            )

    def _upsert_sync(self, tag: str, content: str, source: str) -> int:
        self._ensure_collection()
        pid = uuid.uuid4().int & ((1 << 64) - 1)
        vec = self._vec(f"{tag}. {tag}. {content}")  # tag doppelt → mehr Gewicht
        self.q_client.upsert(
            self.collection,
            [{"id": pid,
              "vector": vec,
              "payload": {"tag": tag, "content": content, "source": source}}],
        )
        return pid

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
        """Show stored entries (chunked ≤ 2000 chars)."""
        await self.ensure_qdrant()
        pts, _ = self.q_client.scroll(self.collection, with_payload=True, limit=1000)
        if not pts:
            return await ctx.send("No knowledge entries stored.")
        hdr, ftr, maxlen = "```\n", "```", 2000
        cur, chunks = hdr, []
        for p in pts:
            pl = p.payload or {}
            snip = pl.get("content", "")[:280].replace("\n", " ")
            line = f"[{p.id}] ({pl.get('tag','NoTag')},{pl.get('source','?')}): {snip}\n"
            if len(cur) + len(line) > maxlen - len(ftr):
                chunks.append(cur + ftr); cur = hdr + line
            else:
                cur += line
        chunks.append(cur + ftr)
        for c in chunks:
            await ctx.send(c)

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
    # GitHub‑Wiki import  (unchanged)
    # --------------------------------------------------------------------
    @commands.command()
    async def importwiki(self, ctx, repo: str = "https://github.com/Kvitekvist/FUS.wiki.git"):
        await self.ensure_qdrant()
        data_dir = str(cog_data_path(self)); os.makedirs(data_dir, exist_ok=True)
        clone_dir = os.path.join(data_dir, "wiki")

        # purge old wiki docs
        filt = {"must": [{"key": "source", "match": {"value": "wiki"}}]}
        ids = await asyncio.get_running_loop().run_in_executor(None, self._collect_ids_sync, filt)
        if ids:
            await asyncio.get_running_loop().run_in_executor(None, lambda: self.q_client.delete(self.collection, ids))

        # git clone / pull
        if os.path.isdir(os.path.join(clone_dir, ".git")):
            subprocess.run(["git", "-C", clone_dir, "pull"], check=False)
        else:
            shutil.rmtree(clone_dir, ignore_errors=True)
            subprocess.run(["git", "clone", repo, clone_dir], check=True)
        await ctx.send("Wiki repo updated – importing …")

        md_files = glob.glob(os.path.join(clone_dir, "*.md"))
        if not md_files:
            return await ctx.send("No markdown pages found – aborting.")

        def _import(fp: str):
            txt = open(fp, encoding="utf-8").read()
            soup = BeautifulSoup(markdown(txt), "html.parser")
            tags = ", ".join({h.get_text(strip=True) for h in soup.find_all(re.compile(r"^h[1-3]$"))})
            plain = soup.get_text(" ", strip=True)
            self._upsert_sync(tags or os.path.basename(fp), plain, "wiki")

        loop = asyncio.get_running_loop()
        for fp in md_files:
            await loop.run_in_executor(None, _import, fp)
        await ctx.send(f"Wiki import done ({len(md_files)} pages).")

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
                vectors_config={"size": self.vec_dim, "distance": "Cosine"},
            ),
        )
        await ctx.send(f"Collection **{self.collection}** recreated.")

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

        # ---------- (A) Heuristische Query‑Erweiterung -----------------
        aug_q = question
        ql = question.lower()

        # Wenn der User Virtual‑Desktop erwähnt, aber nicht ausdrücklich
        # nach „resolution“ fragt, erweitern wir die Such‑Query.
        if "virtual" in ql and "desktop" in ql and "resolution" not in ql:
            aug_q += " resolution"
        # --------------------------------------------------------------

        # ---------- (B) Keyword‑ / Tag‑Filter -------------------------
        clean_q = re.sub(r"[^\w\s]", " ", aug_q.lower())          # z. B. „Virtual‑Desktop“ → "virtual desktop"
        kws = [w for w in clean_q.split() if len(w) > 2]

        tag_filter = (
            {
                "should": (
                    [{"key": "tag",     "match": {"value": k}} for k in kws] +
                    [{"key": "content", "match": {"value": k}} for k in kws]
                )
            }
            if kws else None
        )

        # Suche ausschließlich über das Filter (kein Vektor nötig)
        tag_hits = []
        if tag_filter:
            tag_hits, _ = self.q_client.scroll(
                self.collection,
                with_payload=True,
                limit=20,
                scroll_filter=tag_filter,
            )
        # --------------------------------------------------------------

        # ---------- (C) Vektor‑Suche ----------------------------------
        vec_hits = self._safe_search(
            self.collection,
            query_vector=self._vec(aug_q),         #  <-  aug_q statt question
            limit=20,
        )
        # --------------------------------------------------------------

        # ---------- (D) Kombinieren & Re‑Rank -------------------------
        hit_dict = {h.id: h for h in vec_hits}
        for h in tag_hits:
            hit_dict.setdefault(h.id, h)
        hits = list(hit_dict.values())
        if not hits:
            return "No relevant information found."

        numbered = "\n\n".join(f"#{i}\n{h.payload.get('content','')[:600]}" for i, h in enumerate(hits))
        rank_prompt = (
            f"User question: {question}\n\n"          # ORIGINAL QUESTION
            "Below are context snippets (#0 …). Rate relevance 0‑10.\n"
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
        ctx = "\n\n".join(f"[{h.id}] {h.payload.get('content','')[:700]}" for h, _ in ranked)
        # --------------------------------------------------------------

        # ---------- (E) Finale Antwort‑Generierung --------------------
        final_prompt = (
            "Use **only** the facts below to answer. "
            "If the facts are insufficient, say so.\n\n"
            f"### Facts ###\n{ctx}\n\n"
            f"### Question ###\n{question}\n\n"      # ORIGINAL QUESTION
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

    @commands.Cog.listener()
    async def on_message(self, m: discord.Message):
        if m.author.bot or not m.guild:
            return
        if self.bot.user.mentioned_in(m):
            q = m.clean_content.replace(f"@{self.bot.user.display_name}", "").strip()
            if q:
                async with m.channel.typing():
                    ans = await self._answer(q)
                await m.channel.send(ans)

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
