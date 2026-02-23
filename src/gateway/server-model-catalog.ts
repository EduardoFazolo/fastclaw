import {
  loadModelCatalog,
  type ModelCatalogEntry,
  resetModelCatalogCacheForTest,
} from "../agents/model-catalog.js";
import {
  getCustomProviderApiKey,
  resolveAwsSdkEnvVarName,
  resolveEnvApiKey,
} from "../agents/model-auth.js";
import { ensureAuthProfileStore, listProfilesForProvider } from "../agents/auth-profiles.js";
import { normalizeProviderId } from "../agents/model-selection.js";
import { loadConfig } from "../config/config.js";

export type GatewayModelChoice = ModelCatalogEntry;

const NO_KEY_PROVIDER_ALLOWLIST = new Set(["ollama", "vllm", "synthetic"]);
const MOONSHOT_ALLOWED_MODELS_ENV = "FASTCLAW_MOONSHOT_ALLOWED_MODEL_IDS";

function readMoonshotAllowedModelIds(env: NodeJS.ProcessEnv = process.env): Set<string> {
  const raw = env[MOONSHOT_ALLOWED_MODELS_ENV];
  if (!raw) {
    return new Set();
  }
  const ids = raw
    .split(",")
    .map((id) => id.trim().toLowerCase())
    .filter(Boolean);
  return new Set(ids);
}

function hasProviderAccess(params: {
  provider: string;
  cfg: ReturnType<typeof loadConfig>;
  store: ReturnType<typeof ensureAuthProfileStore>;
}): boolean {
  const normalized = normalizeProviderId(params.provider);

  if (NO_KEY_PROVIDER_ALLOWLIST.has(normalized)) {
    return true;
  }
  if (listProfilesForProvider(params.store, normalized).length > 0) {
    return true;
  }
  if (resolveEnvApiKey(normalized)?.apiKey) {
    return true;
  }
  if (getCustomProviderApiKey(params.cfg, normalized)) {
    return true;
  }
  if (normalized === "amazon-bedrock") {
    // Bedrock can use AWS credential chain without API keys.
    // We only treat it as available when explicit AWS auth env is present.
    return Boolean(resolveAwsSdkEnvVarName());
  }
  return false;
}

// Test-only escape hatch: model catalog is cached at module scope for the
// process lifetime, which is fine for the real gateway daemon, but makes
// isolated unit tests harder. Keep this intentionally obscure.
export function __resetModelCatalogCacheForTest() {
  resetModelCatalogCacheForTest();
}

export async function loadGatewayModelCatalog(): Promise<GatewayModelChoice[]> {
  const cfg = loadConfig();
  const store = ensureAuthProfileStore();
  const all = await loadModelCatalog({ config: cfg });
  const providerAccess = new Map<string, boolean>();
  const moonshotAllowedModelIds = readMoonshotAllowedModelIds();

  return all.filter((entry) => {
    const normalized = normalizeProviderId(entry.provider);
    if (normalized === "moonshot" && moonshotAllowedModelIds.size > 0) {
      const normalizedModelId = entry.id.trim().toLowerCase();
      if (!moonshotAllowedModelIds.has(normalizedModelId)) {
        return false;
      }
    }
    const cached = providerAccess.get(normalized);
    if (cached !== undefined) {
      return cached;
    }
    const allowed = hasProviderAccess({
      provider: normalized,
      cfg,
      store,
    });
    providerAccess.set(normalized, allowed);
    return allowed;
  });
}
