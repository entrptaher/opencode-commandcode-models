# opencode-commandcode-models

An [OpenCode](https://opencode.ai) plugin that auto-discovers CommandCode models and merges
any that are missing from your `commandcode` provider.

## What it does

- Fetches the CommandCode OpenAI-compatible `/models` endpoint at startup.
- Registers a **model transform** so newly discovered model IDs are added automatically.
  Your models already declared in `opencode.json` always win, so curated metadata
  (cost, variants, modalities, limits) is preserved.
- Re-checks upstream every 30 minutes and calls `ctx.model.reload()` to pick up
  models added later. No restart required.

Because the upstream `/models` response only includes `id`, `name`, and
`context_length`, any newly discovered model is filled in with family-based
defaults (reasoning variant sets and input modalities are inferred from the
model name — e.g. `claude-*`, `gpt-*`, `deepseek/*`, `qwen*`, …).

## Install

Add the plugin to your global `~/.config/opencode/opencode.json(c)`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "plugins": ["entrptaher/opencode-commandcode-models"]
}
```

Or point directly at a git ref:

```jsonc
{
  "plugins": ["github:entrptaher/opencode-commandcode-models"]
}
```

Then restart OpenCode (or let the config watcher pick it up). Confirm it loaded:

```sh
opencode plugin list
```

## Configuration

The plugin reads connection details from your global `opencode.json(c)`:

- **baseURL** — `provider.commandcode.options.baseURL` (or `settings.baseURL`),
  falling back to `https://api.commandcode.ai/provider/v1`.
- **apiKey** — `provider.commandcode.options.apiKey` (or `settings.apiKey`).

Environment overrides are also honored:

- `COMMANDCODE_BASE_URL`
- `COMMANDCODE_API_KEY`

## Diagnostics

A small log is written to `/tmp/opencode/commandcode-models.log` with discovery
and merge counts.

## Notes

- The plugin has **no runtime dependencies**; `@opencode/plugin` is a type-only
  import, so the published artifact is dependency-free.
- It is safe if the `commandcode` provider is not configured — the transform
  simply finds no models to merge.

## License

MIT