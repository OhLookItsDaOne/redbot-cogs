```
██████╗  ██╗   ██╗ ███████╗ ██████╗  ███████╗ ███████╗ ███████╗ ██████╗      ██████╗ ██████╗   ██████╗  ███████╗
██╔══██╗ ██║   ██║ ██╔════╝ ██╔══██╗ ██╔════╝ ██╔════╝ ██╔════╝ ██╔══██╗    ██╔════╝ ██╔══██╗ ██╔════╝  ██╔════╝
██║  ██║ ██║   ██║ █████╗   ██████╔╝ ███████╗ █████╗   █████╗   ██████╔╝    ██║      ██║  ██║ ██║  ███╗ ███████╗
██║  ██║ ╚██╗ ██╔╝ ██╔══╝   ██╔══██╗ ╚════██║ ██╔══╝   ██╔══╝   ██╔══██╗    ██║      ██║  ██║ ██║   ██║ ╚════██║
╚██████╔╝  ╚████╔╝  ███████╗ ██║  ██║ ███████║ ███████╗ ███████╗ ██║  ██║    ╚██████╗ ╚██████╔╝ ╚██████╔╝ ███████║
 ╚═════╝   ╚═══╝   ╚══════╝ ╚═╝  ╚═╝ ╚══════╝ ╚══════╝ ╚══════╝ ╚═╝  ╚═╝     ╚═════╝  ╚═════╝  ╚═════╝  ╚══════╝
```

Custom [Red-DiscordBot](https://github.com/Cog-Creators/Red-DiscordBot) cogs for **Community Overseer#6370**, a help and moderation bot for the **FUS SkyrimVR community server**.

All commands are **hybrid** commands – available both as Discord slash commands (`/...`) and classic prefix commands (`!...`).

🐉 ⚔️ 🛡️ 🤴 👸 🛡️ ⚔️ 🐉

## 🐉 The Cogs

| Cog | Slash group | What it does |
|---|---|---|
| `automod` | `/automod`, `/siteallow` | Manage Discord AutoMod rules and maintain the allowed-site allow list (max. 100 entries) |
| `autoreply` | `/autoreply` | Keyword-based automatic help replies with per-user cooldowns, ignored roles and forum thread scanning |
| `banchannel` | `/banchannel` | Instant-ban channel with automatic unban after a configurable duration |
| `gallery` | `/gallery` | Image-only channels; every image post opens a thread for comments |
| `imagespam` | `/imagespam`, `/imagespaminfo` | Advanced image-spam protection with configurable limits and optional timeouts |
| `messageforward` | `/forward` | Forward messages to a support channel via command or message context menu |
| `supportforum` | `/forumhelp` | Automatic troubleshooting message in new forum threads, plus `/privacy` and `/setprivacypolicy` |
| `cogupdater` | `/cogupdate` | Owner-only: pull all loaded cog repos and reload them |

⚔️ 🛡️ ⚔️

## ⚔️ Installation

Add the repo to your bot's Downloader and install the cogs you need:

```
[p]repo add d1 https://github.com/OhLookItsDaOne/redbot-cogs
[p]cog install d1 <cog>
[p]cog load <cog>
```

> `[p]` is your bot's prefix (`!` for Community Overseer).

After installing cogs that add new slash commands, run:

```
!slash sync
```

Each cog shows its configuration options right after installation via its install message. Run the bare group command to see the available options, e.g. `[p]automod`, `[p]gallery`, `[p]forward`. Note that the cog folder name is used for installation: `/forward` lives in the `messageforward` cog, so install it with `[p]cog install d1 messageforward`.

🛡️ ⚔️ 🐉

## 🛡️ Privacy

See [PRIVACY.md](PRIVACY.md) for the full privacy policy. It is also available in Discord via `/privacy`.
