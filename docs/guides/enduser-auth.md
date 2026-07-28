# End-User Authentication — Running Agents as Platform Participants

How to run a mindroid agent with `auth.type = "enduser"`: the agent holds a
short-lived, participant-scoped JWT instead of an account credential. **This is
the recommended identity for any agent connected to a MagickMind-style
platform** — hosted by the platform, by you, or by a third party. The
`apikey` flow (email/password + API key) remains the right tool for the
development loop; see [When to use which](#when-to-use-which).

## Two roles, two credentials

Every agent deployment involves two distinct actors. Keep their credentials
separate:

| Role | Holds | Used for |
|------|-------|----------|
| **Control plane** (your backend, script, or supervisor) | Your platform account credential | Creating the agent identity, binding a persona, **minting the agent's end-user token** |
| **Agent process** (the mindroid runtime) | Only the minted end-user JWT | Everything at runtime: persona prepare, episodic memory, transport |

The control plane acts *before and around* the agent — the agent process never
holds the account credential. Mint the token with your service credential, hand
it to the process, done.

## Why run the agent as an end user

- **Memory ownership is enforced, not self-reported.** Under an end-user
  token, the server forces every episodic-memory write and read to the token's
  own subject. Under an account credential the agent *names* which agent's
  memory it touches — one bug and agent A's experiences land in agent B's
  mind. If you run more than one agent, this is the difference between
  divergent memory working and silently corrupting.
- **Least privilege.** An end-user token can read its own persona and touch
  its own memory. It cannot re-bind its persona, mint tokens, create or delete
  agents, or read any other participant's data. A compromised agent process is
  one participant for one token-lifetime — not your whole account.
- **Revocability.** Tokens are short-lived and individually revocable. Kill
  one agent without rotating your account credential.

## When to use which

| Situation | Use |
|-----------|-----|
| Local development, single test agent, quick iteration | `apikey` (dev loop — fewer moving parts) |
| Any agent running unattended, or more than one agent | `enduser` |
| Multiple agents sharing one process or host | `enduser`, one token each — never one shared account credential |

## The flow

### 1. Control plane: create the agent and mint its token

Using your service credential against the platform's control-plane API
(endpoint names vary by platform; MagickMind shown):

```
POST /v1/end-users                     {"participant_type": "AGENT", ...}   → agent identity
POST /v1/end-users/{id}/persona        {"persona_id": ...}                  → persona bound
POST /v1/end-users/tokens              {"subject_id": "<agent_id>"}         → { "token": "eyJ...", "expires_at": ... }
```

### 2. Agent config

```toml
[agent]
agent_id = "your-agent-id"        # kept for the mention gate; the token carries identity

[auth]
type = "enduser"                  # selects the id-less participant routes
token = "eyJ...the-minted-jwt..."
base_url = "https://api.example.com"

[transport]
type = "centrifugo"
url = "wss://realtime.example.com/connection/websocket"

[memory]
type = "magickmind"

[persona]
type = "magickmind-prepared-agent"

[episodes]
enabled = true
scope = "addressed"               # see the ingest-scope notes in the example config
```

A complete runnable config is at
`examples/persona_agent/magickmind-prepared-enduser.toml`.

### 3. What changes at runtime

`auth.type = "enduser"` routes the SDK to the **id-less participant
endpoints** — `POST /v1/end-user/persona/prepare`, the
`/v1/end-user/episodes` family — where the server derives identity from the
token subject. There is no agent id in any path or body: the agent *cannot*
ask for another agent's prompt or memory, by construction.

On the WebSocket, the end-user JWT is not a platform session token — it is
validated by the platform's **connect proxy**. The token must travel in the
connect command's `data` payload (where the proxy reads it), not in the
protocol-level `token` field (which the realtime server validates against its
own IdP and would reject).

## Token lifetime — plan for it

End-user tokens are deliberately short-lived (commonly 1 hour, capped at 24).
Two things expire together when the token does:

1. **REST calls** start returning 401.
2. **The WebSocket connection is dropped** — the platform pins the
   connection's lifetime to the JWT's expiry. Refreshing the connection
   requires presenting a *fresh* token, not re-sending the old one.

The SDK currently reads `auth.token` once at startup, so your **control plane
owns keeping the credential fresh**. Two working patterns:

- **Supervised (recommended when you run the process):** re-mint on a timer at
  ~half the token's lifetime and deliver the fresh token to the process —
  e.g. write it to a file the process re-reads, or restart the process with
  the new token. This is the same pattern Kubernetes uses for service-account
  tokens.
- **Self-refresh:** if your platform exposes a participant token-refresh
  endpoint (exchange a still-valid token for a fresh one), call it before
  expiry. Check your platform's API for availability and its chain-lifetime
  limits.

Either way: treat the token as ephemeral. Never commit it, never bake it into
an image, prefer injecting it at start (file or environment) over writing it
into a long-lived config file.

## Troubleshooting

- **`401 ... unexpected signing method` on a REST call** — you hit a
  control-plane (account-credential) route with the end-user token. End-user
  tokens only open the `/v1/end-user/...` surface; lifecycle operations
  (create/bind/mint) stay with your control plane. If a *participant*
  operation returns this, your platform may not expose that route on the
  end-user surface yet — check its API reference.
- **WebSocket connects, then drops after a while** — the JWT expired and the
  connection expired with it. See [Token lifetime](#token-lifetime--plan-for-it).
- **WebSocket rejected at connect** — the token was likely sent in the
  protocol `token` field instead of the connect `data` payload, so it was
  validated against the wrong issuer.
- **Empty recall for a brand-new agent** — not an auth problem: a new
  end-user has no episodes yet. Verify by writing one
  (`/v1/end-user/episodes/process`) and searching for it.

## Cross-references

- [Magick Mind Integration](magickmind-integration.md) — the full pipeline walkthrough (uses the `apikey` dev-loop flow)
- `examples/persona_agent/magickmind-prepared-enduser.toml` — runnable end-user config
- `examples/persona_agent/magickmind-prepared-agent.toml` — the same pipeline on the service-credential dev flow
