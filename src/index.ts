// opencode-commandcode-models
//
// Auto-discovers CommandCode models from its OpenAI-compatible `/models`
// endpoint and merges any that are missing from the `commandcode` provider.
//
// Design notes:
// - This is a plain `{ id, setup }` plugin with **no runtime dependencies**.
//   `@opencode/plugin` is imported with `import type` only, so the compiled
//   plugin carries zero imports. That keeps it loadable both as an installed
//   package plugin and as a local `.ts` file, and avoids depending on the
//   server's module resolution.
// - It registers a *model* transform, which operates on the materialized
//   models of available providers (including models declared in opencode.json).
//   Existing models always win, so curated metadata (cost, variants,
//   modalities, limits) is preserved. Only model IDs that are not already
//   present are added, using family-based defaults.
// - Connection details (baseURL / apiKey) are read from the global
//   opencode.json(c), with COMMANDCODE_BASE_URL / COMMANDCODE_API_KEY
//   environment overrides.

import type { Context } from "@opencode/plugin"
import { appendFileSync, existsSync, readFileSync } from "node:fs"
import { homedir } from "node:os"
import { join } from "node:path"

/** Provider ID this plugin augments. */
const PROVIDER_ID = "commandcode"
const DEFAULT_BASE_URL = "https://api.commandcode.ai/provider/v1"
/** How often to re-check upstream for newly added models. */
const REFRESH_MS = 30 * 60 * 1000 // 30 minutes
const LOG = "/tmp/opencode/commandcode-models.log"

export interface UpstreamModel {
  id: string
  name?: string
  context_length?: number
}

function log(message: string) {
  try {
    appendFileSync(LOG, `${new Date().toISOString()} ${message}\n`)
  } catch {
    /* best effort */
  }
}

function readConfig(): any {
  const dirs = [
    process.env.XDG_CONFIG_HOME ? join(process.env.XDG_CONFIG_HOME, "opencode") : undefined,
    join(homedir(), ".config", "opencode"),
  ].filter((d): d is string => Boolean(d))

  for (const dir of dirs) {
    for (const file of ["opencode.json", "opencode.jsonc"]) {
      const path = join(dir, file)
      if (!existsSync(path)) continue
      let text = readFileSync(path, "utf8")
      try {
        return JSON.parse(text)
      } catch {
        /* tolerate comments / trailing commas in .jsonc */
      }
      try {
        text = text
          .replace(/\/\*[\s\S]*?\*\//g, "")
          .replace(/(^|[^:])\/\/.*$/gm, "$1")
          .replace(/,(\s*[}\]])/g, "$1")
        return JSON.parse(text)
      } catch {
        return undefined
      }
    }
  }
  return undefined
}

function resolveConnection(): { baseURL: string; apiKey?: string } {
  const cfg = readConfig()
  const cc = cfg?.provider?.[PROVIDER_ID] ?? cfg?.providers?.[PROVIDER_ID]
  const baseURL: string =
    cc?.options?.baseURL ??
    cc?.settings?.baseURL ??
    process.env.COMMANDCODE_BASE_URL ??
    DEFAULT_BASE_URL
  const apiKey: string | undefined =
    cc?.options?.apiKey ?? cc?.settings?.apiKey ?? process.env.COMMANDCODE_API_KEY
  return { baseURL, apiKey }
}

async function fetchUpstream(baseURL: string, apiKey?: string): Promise<UpstreamModel[]> {
  const res = await fetch(`${baseURL.replace(/\/+$/, "")}/models`, {
    headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
  })
  if (!res.ok) throw new Error(`/models responded ${res.status} ${res.statusText}`)
  const json: any = await res.json()
  const list = Array.isArray(json) ? json : json?.data
  if (!Array.isArray(list)) throw new Error("/models returned an unexpected payload")
  return list.filter((m: any) => typeof m?.id === "string" && m.id.length > 0)
}

// --- family heuristics (upstream only returns id / name / context_length) ---
const VARIANT_SETS: Record<string, string[]> = {
  claude: ["low", "medium", "high", "xhigh", "max"],
  gpt: ["none", "low", "medium", "high", "xhigh", "max"],
  gemini: ["none", "minimal", "low", "medium", "high"],
  deepseek: ["none", "low", "high", "max"],
  qwen: ["none", "low", "medium", "high"],
  default: ["none", "low", "medium", "high"],
}

function classify(id: string): string {
  const s = id.toLowerCase()
  if (s.includes("claude")) return "claude"
  if (s.includes("gemini")) return "gemini"
  if (s.includes("gpt")) return "gpt"
  if (s.includes("deepseek")) return "deepseek"
  if (/qwen|glm|kimi|grok|minimax|mimo|muse|inkling|nemotron/.test(s)) return "qwen"
  return "default"
}

function inputModalities(id: string): string[] {
  const s = id.toLowerCase()
  if (s.includes("deepseek") && !s.includes("vision")) return ["text"]
  if (s.includes("ling") || s.includes("nemotron")) return ["text"]
  if (/claude|gpt|gemini/.test(s)) return ["text", "image", "pdf"]
  return ["text", "image"]
}

/**
 * Build a Model.Info-shaped object for a discovered model.
 * Mirrors Model.Info.default() from @opencode/schema with family heuristics.
 */
function buildModel(u: UpstreamModel) {
  const family = classify(u.id)
  const context = u.context_length ?? 200_000
  return {
    id: u.id,
    modelID: u.id,
    providerID: PROVIDER_ID,
    name: u.name ?? u.id,
    family,
    capabilities: { tools: true, input: inputModalities(u.id), output: ["text"] },
    variants: VARIANT_SETS[family].map((id) => ({ id, settings: { reasoningEffort: id } })),
    time: { released: 0 },
    cost: [],
    status: "active" as const,
    enabled: true,
    limit: { context, output: Math.min(context, 131_072) },
  }
}

let upstream: UpstreamModel[] = []

export default {
  id: "commandcode.models",

  async setup(ctx: Context) {
    const { baseURL, apiKey } = resolveConnection()
    if (!apiKey) {
      log("no API key found in global config or COMMANDCODE_API_KEY; discovery disabled")
    }

    try {
      upstream = await fetchUpstream(baseURL, apiKey)
      log(`discovered ${upstream.length} models from ${baseURL}`)
    } catch (error) {
      log(`initial discovery failed: ${(error as Error).message}`)
    }

    await ctx.model.transform((editor) => {
      const existing = editor.list(PROVIDER_ID)
      const have = new Set<string>(existing.map((m) => m.id))
      let added = 0
      let failed = 0
      for (const u of upstream) {
        if (have.has(u.id)) continue
        try {
          editor.update(PROVIDER_ID, u.id, (draft) => {
            Object.assign(draft, buildModel(u))
          })
          added += 1
        } catch (error) {
          failed += 1
          log(`could not add "${u.id}": ${(error as Error).message}`)
        }
      }
      log(
        `model transform: existing=${existing.length} upstream=${upstream.length} added=${added} failed=${failed}`,
      )
    })

    const timer = setInterval(() => {
      void (async () => {
        try {
          upstream = await fetchUpstream(baseURL, apiKey)
          await ctx.model.reload()
          log(`refresh: ${upstream.length} upstream models`)
        } catch (error) {
          log(`refresh failed: ${(error as Error).message}`)
        }
      })()
    }, REFRESH_MS)

    return () => clearInterval(timer)
  },
}