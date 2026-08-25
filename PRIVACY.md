# Privacy Policy

**Effective date:** 2026-08-25

This Privacy Policy explains what data the bot ("Community Overseer#6370", operated for the FUS SkyrimVR community server) collects, stores, and processes, and how it is handled.

## 1. Data collected and stored

The bot runs on the open-source [Red-DiscordBot](https://github.com/Cog-Creators/Red-DiscordBot) framework. The Red core, as well as the installed cogs, store configuration and operational data. This bot stores:

- **Guild/server configuration** – channel IDs, role IDs, configured keywords, response messages, allowed-site lists, limits and other server settings that server administrators configure through the bot.
- **Cooldown timestamps** – the timestamp of the last automated help reply sent to a user (used only to prevent spam/duplicate replies). This is the only end-user data stored by this bot.
- **Operational records** – for example, scheduled automatic unban timestamps used by the moderation cog.

The bot does **not** store message content persistently. Message content is processed in memory only for moderation and help features and is discarded immediately.

## 2. Message content handling

The bot inspects message content and attachments in real time to provide:

- **Image-spam protection** – counting images/attachments and deleting messages that exceed a configured limit.
- **Image gallery channels** – enforcing image-only channels and creating threads for image posts.
- **Keyword auto-reply** – scanning for configured keywords and posting relevant help information.

This processing happens **in memory only**. Message content is not stored, logged, sold, or shared with third parties.

## 3. Off-platform processing and AI

- No message content is transmitted off-platform (outside of Discord).
- No message content is used to train machine learning or AI models.
- No user data is sold or shared with third parties.

## 4. Moderation actions

The bot may perform moderation actions, including deleting messages, applying timeouts, or temporarily banning users, according to the configuration set by server administrators. These actions are necessary for the operation of the features described above and cannot be opted out of, as doing so would defeat the purpose of automated moderation.

## 5. How to access, correct, or delete your data

Because the bot only stores the cooldown timestamps described above, there is very little data associated with individual users.

- Server administrators can clear cooldown data using the bot's built-in commands.
- Users can use the Red core command `[p]mydata forgetme` to request deletion of non-operational data about them, subject to Red's standard data deletion flow.

## 6. Data retention

Cooldown timestamps are overwritten each time a new auto-reply is sent and do not accumulate indefinitely. Guild configuration is retained as long as the bot is used on the server and can be reset by administrators.

## 7. Security

The bot runs on a private, password-protected server. Access to the hosting machine is restricted to the bot owner.

## 8. Changes to this policy

This policy may be updated from time to time. The date at the top indicates when it was last updated.

## 9. Contact

For questions about this policy, please contact the server administrators of the FUS SkyrimVR community server.
