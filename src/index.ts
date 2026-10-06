// opencode-commandcode-models
//
// Auto-discovers CommandCode models from its OpenAI-compatible `/models`
// endpoint and merges any that are missing from the `commandcode` provider.
//
// Metadata strategy (no hardcoded model tables):
//
// The CommandCode `/models` response only carries `id`, `name`, and
// `context_length`, which is far too thin to register a model correctly. So
// for each discovered model this plugin looks the model up in the models.dev
// catalog that OpenCode already has in memory (`editor.provider`, populated by
// the built-in `opencode.models.dev` plugin) and copies the authoritative
// metadata from there: display name, family, modalities, context/output
// limits, real reasoning-effort variant list, and release date.
//
// Lookup is attempted against several plausible catalog IDs, from most to
// least specific (exact, `:free`-suffix stripped, vendor-prefix stripped,
// case-folded), across every provider in the catalog. Because OpenCode
// normalizes `reasoning_options` into per-provider `variants`, the variant
// *names* come from the catalog too, rather than a hand-written table.
//
// The one thing that cannot be derived is the reason the config provider exists
// in the first place: CommandCode is an OpenAI-compatible gateway, so request
// options must use the OpenAI protocol spelling (`reasoningEffort`) rather than
// a native vendor's spelling (Anthropic's `thinking`, for example). That is
// read from the provider's own catalog entry when available and otherwise
// falls back to the OpenAI-compatible default.
//
// A model that matches nothing in the catalog still needs *some* entry to be
// selectable, so a minimal entry is registered using only what the endpoint
// reported, plus OpenCode's documented fallback assumptions.
//
// Design notes:
// - Plain `{ id, setup }` plugin with no runtime dependencies.
//   `@opencode/plugin` is a type-only import, so the published artifact carries
//   zero imports and loads as both an installed package and a local `.ts` file.
// - Registers a *model* transform, which operates on the materialized models
//   of available providers (including models declared in opencode.json).
//   Existing models always win, so curated metadata is preserved.
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

// ---------------------------------------------------------------------------
// Catalog lookup
// ---------------------------------------------------------------------------

type CatalogEntry = {
  providerID: string
  modelID: string
  name: string
  family?: string
  capabilities?: { tools: boolean; input: string[]; output: string[] }
  variants?: { id: string }[]
  cost?: { input: number; output: number; cache: { read: number; write: number } }[]
  status?: string
  time?: { released?: number }
  limit?: { context?: number; input?: number; output?: number }
  settings?: Record<string, any>
}

/**
 * Candidate catalog IDs for a CommandCode model, most specific first.
 * CommandCode IDs are vendor-qualified (`deepseek/deepseek-v4-pro`) and may carry
 * a routing suffix (`:free`), while catalog IDs are usually bare.
 */
function candidateIDs(id: string): string[] {
  const bare = id.includes("/") ? id.slice(id.lastIndexOf("/") + 1) : id
  const withoutSuffix = (s: string) => s.replace(/:free$/, "")
  const out = [id, withoutSuffix(id), bare, withoutSuffix(bare)]
  const folded = out.map((s) => s.toLowerCase())
  return [...new Set([...out, ...folded])]
}

/**
 * Find the best catalog entry for a discovered model.
 *
 * Providers that serve the model through an OpenAI-compatible API are
 * preferred, because their recorded variant settings already use the protocol
 * spelling this provider needs. Ties break toward the entry with the richest
 * metadata so the most complete record wins.
 */
function findCatalogEntry(
  index: Map<string, CatalogEntry[]>,
  id: string,
  compatibleProviders: Set<string>,
): CatalogEntry | undefined {
  for (const candidate of candidateIDs(id)) {
    const entries = index.get(candidate)
    if (!entries?.length) continue
    const best = [...entries].sort((a, b) => {
      const compat = Number(compatibleProviders.has(b.providerID)) - Number(compatibleProviders.has(a.providerID))
      if (compat !== 0) return compat
      const meta =
        Number(b.capabilities?.input.length ?? 0) +
        Number(b.limit?.output ? 1 : 0) +
        Number(b.cost?.length ?? 0) +
        Number(b.time?.released ? 1 : 0)
      const metaA =
        Number(a.capabilities?.input.length ?? 0) +
        Number(a.limit?.output ? 1 : 0) +
        Number(a.cost?.length ?? 0) +
        Number(a.time?.released ? 1 : 0)
      return meta - metaA
    })[0]
    return best
  }
  return undefined
}

function buildIndex(editor: any, compatibleProviders: Set<string>): Map<string, CatalogEntry[]> {
  const index = new Map<string, CatalogEntry[]>()
  for (const record of editor.provider.list()) {
    for (const [modelID, info] of record.models) {
      const entry = info as CatalogEntry
      const key = modelID.toLowerCase()
      const list = index.get(key) ?? []
      list.push({ ...entry, providerID: record.provider.id, modelID })
      index.set(key, list)
    }
  }
  return index
}

/**
 * Reason for the provider's own package, read from the catalog.
 *
 * OpenCode resolves each configured provider to a runtime package
 * (`package`). Knowing the package tells us which protocol spelling request
 * options must use. A package whose name contains `openai` and not `anthropic`
 * means OpenAI-style option names such as `reasoningEffort`.
 */
function packageIsOpenAIStyle(pkg?: string): boolean {
  if (!pkg) return false
  const p = pkg.toLowerCase()
  if (!p.includes("openai")) return false
  return !p.includes("anthropic")
}

// ---------------------------------------------------------------------------
// Model construction
// ---------------------------------------------------------------------------

function buildModel(u: UpstreamModel, entry: CatalogEntry | undefined, compatible: boolean) {
  const family = entry?.family
  const input = entry?.capabilities?.input
  const output = entry?.capabilities?.output
  const tools = entry?.capabilities?.tools

  const context = u.context_length ?? entry?.limit?.context
  const outputLimit = entry?.limit?.output

  const limit: { context: number; output: number; input?: number } = {
    context: context ?? 200_000,
    output: outputLimit ?? 128_000,
  }
  const inputLimit = entry?.limit?.input
  if (inputLimit !== undefined) limit.input = inputLimit

  // Variant names come from the catalog; the request-option shape follows the
  // provider's protocol rather than the native vendor's.
  const variantIDs = (entry?.variants ?? []).map((v) => v.id)
  const variants = variantIDs.map((id) => ({
    id,
    settings: compatible ? { reasoningEffort: id } : (entry?.variants ?? []).find((v) => v.id === id)?.settings ?? {},
  }))

  const model: Record<string, any> = {
    id: u.id,
    modelID: u.id,
    providerID: PROVIDER_ID,
    name: entry?.name ?? u.name ?? u.id,
  }
  if (family) model.family = family

  model.capabilities = {
    tools: tools ?? true,
    input: input ?? ["text", "image"],
    output: output ?? ["text"],
  }
  model.variants = variants
  model.time = { released: entry?.time?.released ?? 0 }
  model.cost = entry?.cost ?? []
  model.status = entry?.status ?? "active"
  model.enabled = true
  model.limit = limit
  return model
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

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

      // Determine this provider's runtime package so variant options use the
      // right protocol spelling.
      const providerRecord = editor.provider.get(PROVIDER_ID)
      const compatible = packageIsOpenAIStyle((providerRecord?.provider as any)?.package)

      const index = buildIndex(editor, new Set())
      let added = 0
      let unmatched = 0
      const misses: string[] = []
      for (const u of upstream) {
        if (have.has(u.id)) continue
        const entry = findCatalogEntry(index, u.id, new Set())
        if (!entry) {
          unmatched += 1
          misses.push(u.id)
        }
        try {
          editor.update(PROVIDER_ID, u.id, (draft) => {
            Object.assign(draft, buildModel(u, entry, compatible))
          })
          added += 1
        } catch (error) {
          log(`could not add "${u.id}": ${(error as Error).message}`)
        }
      }
      log(
        `merge: existing=${existing.length} upstream=${upstream.length} added=${added} ` +
          `catalog-matched=${added - unmatched} catalog-missed=${unmatched} compatible=${compatible}`,
      )
      if (misses.length) log(`no catalog entry for: ${misses.join(", ")}`)
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