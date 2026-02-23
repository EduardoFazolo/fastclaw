import { extractText } from "../chat/message-extract.ts";
import type { GatewayBrowserClient } from "../gateway.ts";
import type { SessionsListResult, SessionsUsageResult } from "../types.ts";
import { generateUUID } from "../uuid.ts";

const JUDGE_TIMEOUT_MS = 180_000;
const PIPELINE_STEP_TIMEOUT_MS = 240_000;
const MAX_PLAN_TASKS = 12;
const USAGE_RETRY_ATTEMPTS = 3;
const USAGE_RETRY_DELAY_MS = 1_000;
const MAX_LOAD_BALANCER_LOG_LINES = 600;
const AGENT_WAIT_POLL_INTERVAL_MS = 20_000;
const AGENT_WAIT_MAX_POLL_INTERVAL_MS = 35_000;
const MODEL_CALL_MIN_INTERVAL_MS = 6_000;
const RATE_LIMIT_RETRY_ATTEMPTS = 3;
const RATE_LIMIT_BACKOFF_BASE_MS = 15_000;
const RATE_LIMIT_BACKOFF_MAX_MS = 120_000;
const RATE_LIMIT_BACKOFF_JITTER_MS = 1_500;

const CHEAP_HINTS = [
  "mini",
  "nano",
  "small",
  "lite",
  "flash",
  "haiku",
  "8b",
  "7b",
  "3b",
  "1.5b",
];

const EXPENSIVE_HINTS = [
  "k2.5",
  "k2",
  "gpt-5",
  "gpt-4.1",
  "o3",
  "o4",
  "opus",
  "sonnet",
  "reasoning",
  "max",
];

const CHEAP_ROUTE_HINTS = ["cheap", "budget", "low", "light", "simple", "economical"];
const EXPENSIVE_ROUTE_HINTS = ["expensive", "premium", "high", "heavy", "complex", "reasoning"];

type AgentAcceptedPayload = {
  runId?: unknown;
};

type AgentWaitPayload = {
  status?: unknown;
  error?: unknown;
};

export type LoadBalancerModelRole = "cheap" | "expensive";

export type LoadBalancerModelOption = {
  id: string;
  ref: string;
  name: string;
  provider: string;
  contextWindow?: number;
  reasoning?: boolean;
};

export type LoadBalancerPlanTask = {
  id: string;
  title: string;
  role: LoadBalancerModelRole;
  instructions: string;
  rationale?: string;
};

export type LoadBalancerPlan = {
  summary: string;
  tasks: LoadBalancerPlanTask[];
  judges: string[];
  raw: string;
  sourceTask: string;
  createdAt: number;
};

type ParsedJudgePlan = {
  summary: string;
  tasks: Array<Omit<LoadBalancerPlanTask, "id">>;
};

type JudgeResult = {
  judgeModelId: string;
  raw: string;
  parsed: ParsedJudgePlan | null;
};

type JudgeExecutionResult = {
  raw: string;
  usageDelta: SessionUsageDelta | null;
};

type SessionUsageSnapshot = {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  totalTokens: number;
  perModelTokens: Map<string, number>;
};

type SessionUsageDelta = {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  totalTokens: number;
  perModelTokens: Array<{ label: string; tokens: number }>;
};

export type LoadBalancerState = {
  client: GatewayBrowserClient | null;
  connected: boolean;
  sessionKey: string;
  chatMessage: string;
  sessionsResult: SessionsListResult | null;
  chatLoadBalancerModelsLoading: boolean;
  chatLoadBalancerModels: LoadBalancerModelOption[];
  chatLoadBalancerCheapModel: string;
  chatLoadBalancerExpensiveModel: string;
  chatLoadBalancerJudgeModels: string[];
  chatLoadBalancerTaskInput: string;
  chatLoadBalancerPlanning: boolean;
  chatLoadBalancerPlan: LoadBalancerPlan | null;
  chatLoadBalancerAwaitingApproval: boolean;
  chatLoadBalancerExecuting: boolean;
  chatLoadBalancerLog: string[];
  chatLoadBalancerError: string | null;
};

export type LoadBalancerExecutionHooks = {
  refreshChatHistory: () => Promise<void>;
  refreshSessions?: () => Promise<void>;
};

function appendLog(state: LoadBalancerState, message: string) {
  const timestamp = new Date().toLocaleTimeString();
  const next = [...state.chatLoadBalancerLog, `[${timestamp}] ${message}`];
  state.chatLoadBalancerLog =
    next.length > MAX_LOAD_BALANCER_LOG_LINES
      ? next.slice(next.length - MAX_LOAD_BALANCER_LOG_LINES)
      : next;
}

function toTokenNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function formatSignedTokens(value: number): string {
  const rounded = Math.round(value);
  return `${rounded >= 0 ? "+" : ""}${rounded}`;
}

function formatElapsedDuration(ms: number): string {
  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  if (totalSeconds < 60) {
    return `${totalSeconds}s`;
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}m ${seconds}s`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, Math.max(0, Math.round(ms))));
}

function isRateLimitError(error: unknown): boolean {
  const message = String(error).toLowerCase();
  return (
    message.includes("rate limit") ||
    message.includes("too many requests") ||
    message.includes("status 429") ||
    message.includes("http 429") ||
    message.includes("quota exceeded")
  );
}

function parseRetryAfterMs(error: unknown): number | null {
  const message = String(error);
  const retryAfterMatch = message.match(
    /retry[-\s]*after\s*:?\s*(\d+(?:\.\d+)?)\s*(ms|msec|millisecond|milliseconds|s|sec|second|seconds|m|min|minute|minutes)?/i,
  );
  if (!retryAfterMatch) {
    return null;
  }
  const value = Number(retryAfterMatch[1]);
  if (!Number.isFinite(value) || value <= 0) {
    return null;
  }
  const unit = retryAfterMatch[2]?.toLowerCase() ?? "s";
  if (unit.startsWith("ms")) {
    return Math.round(value);
  }
  if (unit.startsWith("m") && !unit.startsWith("ms")) {
    return Math.round(value * 60_000);
  }
  return Math.round(value * 1_000);
}

function computeRateLimitBackoffMs(error: unknown, attempt: number): number {
  const retryAfterMs = parseRetryAfterMs(error);
  if (retryAfterMs && Number.isFinite(retryAfterMs)) {
    return Math.max(1_000, Math.min(RATE_LIMIT_BACKOFF_MAX_MS, retryAfterMs));
  }
  const exponential = RATE_LIMIT_BACKOFF_BASE_MS * 2 ** Math.max(0, attempt - 1);
  const jitter = Math.floor(Math.random() * (RATE_LIMIT_BACKOFF_JITTER_MS + 1));
  return Math.max(1_000, Math.min(RATE_LIMIT_BACKOFF_MAX_MS, exponential + jitter));
}

let lastModelCallAtMs = 0;

async function enforceModelCallSpacing(state: LoadBalancerState, label: string): Promise<void> {
  const now = Date.now();
  const waitMs = MODEL_CALL_MIN_INTERVAL_MS - (now - lastModelCallAtMs);
  if (waitMs > 0) {
    appendLog(
      state,
      `${label}: pacing next model request to reduce provider pressure (${formatElapsedDuration(waitMs)} wait).`,
    );
    await sleep(waitMs);
  }
  lastModelCallAtMs = Date.now();
}

async function runAgentWithRateLimitGuard(params: {
  state: LoadBalancerState;
  modelId: string;
  label: string;
  run: () => Promise<void>;
}): Promise<void> {
  for (let attempt = 1; attempt <= RATE_LIMIT_RETRY_ATTEMPTS + 1; attempt += 1) {
    await enforceModelCallSpacing(params.state, params.label);
    try {
      await params.run();
      return;
    } catch (error) {
      if (!isRateLimitError(error) || attempt > RATE_LIMIT_RETRY_ATTEMPTS) {
        throw error;
      }
      const backoffMs = computeRateLimitBackoffMs(error, attempt);
      appendLog(
        params.state,
        `${params.label}: rate limit on ${params.modelId}. Retrying in ${formatElapsedDuration(backoffMs)} (attempt ${attempt}/${RATE_LIMIT_RETRY_ATTEMPTS}).`,
      );
      await sleep(backoffMs);
    }
  }
}

function buildModelUsageLabel(provider: string | undefined, model: string | undefined): string {
  const providerLabel = provider?.trim() || "unknown-provider";
  const modelLabel = model?.trim() || "unknown-model";
  return `${providerLabel}/${modelLabel}`;
}

function readSnapshotForSession(
  result: SessionsUsageResult | null | undefined,
  sessionKey: string,
): SessionUsageSnapshot | null {
  if (!result || !Array.isArray(result.sessions) || result.sessions.length === 0) {
    return null;
  }
  const usageEntry =
    result.sessions.find((entry) => entry?.key === sessionKey) ?? result.sessions[0] ?? null;
  if (!usageEntry?.usage) {
    return {
      inputTokens: 0,
      outputTokens: 0,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
      totalTokens: 0,
      perModelTokens: new Map(),
    };
  }

  const perModelTokens = new Map<string, number>();
  const modelUsage = Array.isArray(usageEntry.usage.modelUsage) ? usageEntry.usage.modelUsage : [];
  for (const item of modelUsage) {
    const label = buildModelUsageLabel(item.provider, item.model);
    const next = toTokenNumber(item.totals?.totalTokens);
    const prev = perModelTokens.get(label) ?? 0;
    perModelTokens.set(label, prev + next);
  }

  return {
    inputTokens: toTokenNumber(usageEntry.usage.input),
    outputTokens: toTokenNumber(usageEntry.usage.output),
    cacheReadTokens: toTokenNumber(usageEntry.usage.cacheRead),
    cacheWriteTokens: toTokenNumber(usageEntry.usage.cacheWrite),
    totalTokens: toTokenNumber(usageEntry.usage.totalTokens),
    perModelTokens,
  };
}

async function fetchSessionUsageSnapshot(
  client: GatewayBrowserClient,
  sessionKey: string,
): Promise<SessionUsageSnapshot | null> {
  try {
    const result = await client.request<SessionsUsageResult>("sessions.usage", {
      key: sessionKey,
      limit: 1,
    });
    return readSnapshotForSession(result, sessionKey);
  } catch {
    return null;
  }
}

function computeUsageDelta(
  before: SessionUsageSnapshot | null,
  after: SessionUsageSnapshot | null,
): SessionUsageDelta | null {
  if (!before && !after) {
    return null;
  }
  const start = before ?? {
    inputTokens: 0,
    outputTokens: 0,
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
    totalTokens: 0,
    perModelTokens: new Map<string, number>(),
  };
  const end = after ?? {
    inputTokens: 0,
    outputTokens: 0,
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
    totalTokens: 0,
    perModelTokens: new Map<string, number>(),
  };

  const labels = new Set<string>([...start.perModelTokens.keys(), ...end.perModelTokens.keys()]);
  const perModelTokens = [...labels]
    .map((label) => {
      const delta = (end.perModelTokens.get(label) ?? 0) - (start.perModelTokens.get(label) ?? 0);
      return { label, tokens: delta };
    })
    .filter((entry) => entry.tokens !== 0)
    .sort((a, b) => {
      const diff = Math.abs(b.tokens) - Math.abs(a.tokens);
      if (diff !== 0) {
        return diff;
      }
      return a.label.localeCompare(b.label);
    });

  return {
    inputTokens: end.inputTokens - start.inputTokens,
    outputTokens: end.outputTokens - start.outputTokens,
    cacheReadTokens: end.cacheReadTokens - start.cacheReadTokens,
    cacheWriteTokens: end.cacheWriteTokens - start.cacheWriteTokens,
    totalTokens: end.totalTokens - start.totalTokens,
    perModelTokens,
  };
}

function recordPipelineModelTotals(
  aggregate: Map<string, number>,
  delta: SessionUsageDelta | null,
): void {
  if (!delta) {
    return;
  }
  for (const entry of delta.perModelTokens) {
    const prev = aggregate.get(entry.label) ?? 0;
    aggregate.set(entry.label, prev + entry.tokens);
  }
}

function logUsageDelta(
  state: LoadBalancerState,
  delta: SessionUsageDelta | null,
  label: string,
): void {
  if (!delta) {
    appendLog(state, `${label} token usage unavailable (usage endpoint returned no data).`);
    return;
  }
  appendLog(
    state,
    `${label} token usage: total ${formatSignedTokens(delta.totalTokens)} (input ${formatSignedTokens(delta.inputTokens)}, output ${formatSignedTokens(delta.outputTokens)}, cache read ${formatSignedTokens(delta.cacheReadTokens)}, cache write ${formatSignedTokens(delta.cacheWriteTokens)}).`,
  );
  if (delta.perModelTokens.length === 0) {
    appendLog(state, `${label} per-model tokens: no model deltas reported.`);
    return;
  }
  const parts = delta.perModelTokens.map(
    (entry) => `${entry.label} ${formatSignedTokens(entry.tokens)}`,
  );
  appendLog(state, `${label} per-model tokens: ${parts.join(", ")}`);
}

function logPipelineModelTotals(state: LoadBalancerState, aggregate: Map<string, number>): void {
  const ordered = [...aggregate.entries()]
    .filter(([, tokens]) => tokens !== 0)
    .sort((a, b) => {
      const diff = Math.abs(b[1]) - Math.abs(a[1]);
      if (diff !== 0) {
        return diff;
      }
      return a[0].localeCompare(b[0]);
    });
  if (ordered.length === 0) {
    appendLog(state, "Pipeline model totals: no model token deltas were recorded.");
    return;
  }
  const preview = ordered.slice(0, 8).map(([label, tokens]) => `${label} ${formatSignedTokens(tokens)}`);
  const suffix = ordered.length > 8 ? ` (+${ordered.length - 8} more)` : "";
  appendLog(state, `Pipeline model totals: ${preview.join(", ")}${suffix}`);
}

async function fetchUsageSnapshotAfterStep(params: {
  client: GatewayBrowserClient;
  sessionKey: string;
  beforeTotalTokens: number | null;
}): Promise<SessionUsageSnapshot | null> {
  let latest: SessionUsageSnapshot | null = null;
  for (let attempt = 0; attempt < USAGE_RETRY_ATTEMPTS; attempt++) {
    latest = await fetchSessionUsageSnapshot(params.client, params.sessionKey);
    if (params.beforeTotalTokens === null) {
      return latest;
    }
    if (!latest) {
      // Continue polling for eventual consistency.
    } else if (latest.totalTokens > params.beforeTotalTokens || attempt === USAGE_RETRY_ATTEMPTS - 1) {
      return latest;
    }
    if (attempt < USAGE_RETRY_ATTEMPTS - 1) {
      await new Promise((resolve) => setTimeout(resolve, USAGE_RETRY_DELAY_MS));
    }
  }
  return latest;
}

function normalizeModelOption(value: unknown): LoadBalancerModelOption | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const entry = value as {
    id?: unknown;
    name?: unknown;
    provider?: unknown;
    contextWindow?: unknown;
    reasoning?: unknown;
  };
  const id = typeof entry.id === "string" ? entry.id.trim() : "";
  if (!id) {
    return null;
  }
  const name = typeof entry.name === "string" && entry.name.trim() ? entry.name.trim() : id;
  const provider =
    typeof entry.provider === "string" && entry.provider.trim() ? entry.provider.trim() : "unknown";
  const contextWindow =
    typeof entry.contextWindow === "number" && Number.isFinite(entry.contextWindow)
      ? Math.max(1, Math.floor(entry.contextWindow))
      : undefined;
  const reasoning = typeof entry.reasoning === "boolean" ? entry.reasoning : undefined;
  const providerPrefix = `${provider.toLowerCase()}/`;
  const normalizedId = id.toLowerCase();
  const ref = normalizedId.startsWith(providerPrefix) ? id : `${provider}/${id}`;
  return {
    id,
    ref,
    name,
    provider,
    contextWindow,
    reasoning,
  };
}

function compareModels(a: LoadBalancerModelOption, b: LoadBalancerModelOption) {
  const provider = a.provider.localeCompare(b.provider);
  if (provider !== 0) {
    return provider;
  }
  return a.id.localeCompare(b.id);
}

function modelFingerprint(model: LoadBalancerModelOption): string {
  return `${model.provider}/${model.id}/${model.name}`.toLowerCase();
}

function scoreHints(model: LoadBalancerModelOption, hints: readonly string[]) {
  const fingerprint = modelFingerprint(model);
  return hints.reduce((score, hint) => score + (fingerprint.includes(hint) ? 1 : 0), 0);
}

function resolveDefaultCheapModel(models: LoadBalancerModelOption[], current?: string): string {
  const keepCurrent = current?.trim();
  if (keepCurrent && models.some((model) => model.ref === keepCurrent)) {
    return keepCurrent;
  }
  const ranked = [...models].sort((a, b) => {
    const aScore = scoreHints(a, CHEAP_HINTS) - scoreHints(a, EXPENSIVE_HINTS);
    const bScore = scoreHints(b, CHEAP_HINTS) - scoreHints(b, EXPENSIVE_HINTS);
    if (aScore !== bScore) {
      return bScore - aScore;
    }
    if (a.reasoning !== b.reasoning) {
      return Number(a.reasoning) - Number(b.reasoning);
    }
    const aWindow = a.contextWindow ?? Number.MAX_SAFE_INTEGER;
    const bWindow = b.contextWindow ?? Number.MAX_SAFE_INTEGER;
    if (aWindow !== bWindow) {
      return aWindow - bWindow;
    }
    return compareModels(a, b);
  });
  return ranked[0]?.ref ?? "";
}

function resolveDefaultExpensiveModel(
  models: LoadBalancerModelOption[],
  cheapModelId: string,
  current?: string,
): string {
  const keepCurrent = current?.trim();
  if (keepCurrent && models.some((model) => model.ref === keepCurrent)) {
    return keepCurrent;
  }
  const ranked = [...models].sort((a, b) => {
    const aScore = scoreHints(a, EXPENSIVE_HINTS) - scoreHints(a, CHEAP_HINTS);
    const bScore = scoreHints(b, EXPENSIVE_HINTS) - scoreHints(b, CHEAP_HINTS);
    if (aScore !== bScore) {
      return bScore - aScore;
    }
    if (a.reasoning !== b.reasoning) {
      return Number(b.reasoning) - Number(a.reasoning);
    }
    const aWindow = a.contextWindow ?? 0;
    const bWindow = b.contextWindow ?? 0;
    if (aWindow !== bWindow) {
      return bWindow - aWindow;
    }
    return compareModels(a, b);
  });
  const preferred = ranked.find((entry) => entry.ref !== cheapModelId);
  return preferred?.ref ?? ranked[0]?.ref ?? "";
}

function normalizeAssignments(state: LoadBalancerState) {
  const models = state.chatLoadBalancerModels;
  if (models.length === 0) {
    state.chatLoadBalancerCheapModel = "";
    state.chatLoadBalancerExpensiveModel = "";
    state.chatLoadBalancerJudgeModels = [];
    return;
  }

  const cheap = resolveDefaultCheapModel(models, state.chatLoadBalancerCheapModel);
  const expensive = resolveDefaultExpensiveModel(
    models,
    cheap,
    state.chatLoadBalancerExpensiveModel,
  );

  state.chatLoadBalancerCheapModel = cheap;
  state.chatLoadBalancerExpensiveModel = expensive;

  const validJudgeIds = state.chatLoadBalancerJudgeModels.filter((id) =>
    models.some((model) => model.ref === id),
  );
  if (validJudgeIds.length > 0) {
    state.chatLoadBalancerJudgeModels = [...new Set(validJudgeIds)];
    return;
  }

  if (expensive) {
    state.chatLoadBalancerJudgeModels = [expensive];
    return;
  }
  state.chatLoadBalancerJudgeModels = [cheap];
}

function sessionModelForKey(sessions: SessionsListResult | null, key: string): string | null {
  const entry = sessions?.sessions?.find((row) => row.key === key);
  const model = typeof entry?.model === "string" ? entry.model.trim() : "";
  return model || null;
}

function isModelAvailabilityError(error: unknown): boolean {
  const message = String(error).toLowerCase();
  return (
    (message.includes("404") && message.includes("model")) ||
    message.includes("not found the model") ||
    message.includes("model not found") ||
    message.includes("permission denied")
  );
}

function pickModelFallback(params: {
  currentModelId: string;
  cheapModelId: string;
  expensiveModelId: string;
  models: LoadBalancerModelOption[];
}): string | null {
  const current = params.currentModelId.trim();
  if (!current) {
    return null;
  }
  const byId = new Set(params.models.map((model) => model.ref));
  const fallbackCandidates: string[] = [];
  for (const candidate of [params.expensiveModelId, params.cheapModelId]) {
    const normalized = candidate.trim();
    if (!normalized || normalized === current) {
      continue;
    }
    if (!byId.has(normalized)) {
      continue;
    }
    if (!fallbackCandidates.includes(normalized)) {
      fallbackCandidates.push(normalized);
    }
  }
  if (fallbackCandidates.length > 0) {
    return fallbackCandidates[0] ?? null;
  }
  return null;
}

async function patchSessionModel(
  client: GatewayBrowserClient,
  key: string,
  model: string | null,
): Promise<void> {
  await client.request("sessions.patch", {
    key,
    model,
  });
}

function sanitizePromptValue(value: string): string {
  return value.replaceAll("```", "'''").trim();
}

function normalizeRole(value: unknown): LoadBalancerModelRole | null {
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim().toLowerCase();
  if (!normalized) {
    return null;
  }
  if (CHEAP_ROUTE_HINTS.some((hint) => normalized.includes(hint))) {
    return "cheap";
  }
  if (EXPENSIVE_ROUTE_HINTS.some((hint) => normalized.includes(hint))) {
    return "expensive";
  }
  return null;
}

function readStringField(record: Record<string, unknown>, keys: string[]): string {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}

function parseJsonPlan(value: unknown): ParsedJudgePlan | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const root = value as {
    summary?: unknown;
    tasks?: unknown;
    plan?: unknown;
  };
  const rawTasks = Array.isArray(root.tasks)
    ? root.tasks
    : Array.isArray(root.plan)
      ? root.plan
      : null;
  if (!rawTasks) {
    return null;
  }
  const parsedTasks: Array<Omit<LoadBalancerPlanTask, "id">> = [];
  for (const rawTask of rawTasks) {
    if (!rawTask || typeof rawTask !== "object") {
      continue;
    }
    const task = rawTask as Record<string, unknown>;
    const role = normalizeRole(task.route ?? task.tier ?? task.bucket ?? task.complexity);
    if (!role) {
      continue;
    }
    const title = readStringField(task, ["title", "task", "name", "step"]);
    if (!title) {
      continue;
    }
    const instructions = readStringField(task, ["instructions", "prompt", "description"]) || title;
    const rationale = readStringField(task, ["rationale", "reason", "why"]) || undefined;
    parsedTasks.push({
      title,
      role,
      instructions,
      rationale,
    });
    if (parsedTasks.length >= MAX_PLAN_TASKS) {
      break;
    }
  }
  if (parsedTasks.length === 0) {
    return null;
  }
  const summary =
    typeof root.summary === "string" && root.summary.trim()
      ? root.summary.trim()
      : "Judge produced a routed task plan.";
  return { summary, tasks: parsedTasks };
}

function extractJsonCandidates(raw: string): string[] {
  const candidates: string[] = [];
  const fenced = raw.matchAll(/```(?:json)?\s*([\s\S]*?)```/gi);
  for (const match of fenced) {
    const body = match[1]?.trim();
    if (body) {
      candidates.push(body);
    }
  }
  const firstBrace = raw.indexOf("{");
  const lastBrace = raw.lastIndexOf("}");
  if (firstBrace >= 0 && lastBrace > firstBrace) {
    candidates.push(raw.slice(firstBrace, lastBrace + 1).trim());
  }
  return [...new Set(candidates)];
}

function parseLineBasedPlan(raw: string): ParsedJudgePlan | null {
  const lines = raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) {
    return null;
  }
  const tasks: Array<Omit<LoadBalancerPlanTask, "id">> = [];
  for (const line of lines) {
    const match = line.match(
      /^(?:[-*]|\d+[.)])?\s*(cheap|expensive|budget|premium|simple|complex)\s*[:\-]\s*(.+)$/i,
    );
    if (!match) {
      continue;
    }
    const role = normalizeRole(match[1]);
    const title = match[2]?.trim();
    if (!role || !title) {
      continue;
    }
    tasks.push({
      title,
      role,
      instructions: title,
    });
    if (tasks.length >= MAX_PLAN_TASKS) {
      break;
    }
  }
  if (tasks.length === 0) {
    return null;
  }
  return {
    summary: lines[0] ?? "Judge produced a routed task plan.",
    tasks,
  };
}

function parseJudgePlan(raw: string): ParsedJudgePlan | null {
  const candidates = extractJsonCandidates(raw);
  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate) as unknown;
      const normalized = parseJsonPlan(parsed);
      if (normalized) {
        return normalized;
      }
    } catch {
      // Continue trying next candidate.
    }
  }
  return parseLineBasedPlan(raw);
}

function findLatestAssistantText(messages: unknown[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (!message || typeof message !== "object") {
      continue;
    }
    const role = (message as { role?: unknown }).role;
    if (role !== "assistant") {
      continue;
    }
    const text = extractText(message)?.trim();
    if (text) {
      return text;
    }
  }
  return "";
}

function buildJudgePrompt(params: {
  task: string;
  cheapModel: string;
  expensiveModel: string;
}): string {
  return [
    "You are a routing judge for a model load-balancer pipeline.",
    `Cheap model id: ${params.cheapModel}`,
    `Expensive model id: ${params.expensiveModel}`,
    "",
    "Decide if the user task should be split into smaller tasks.",
    "Assign each task to route \"cheap\" or \"expensive\".",
    "Return ONLY valid JSON (no markdown, no explanation):",
    '{"summary":"...","tasks":[{"title":"...","route":"cheap|expensive","instructions":"...","rationale":"..."}]}',
    "",
    "Rules:",
    "- Use cheap for straightforward extraction, formatting, boilerplate edits, and deterministic steps.",
    "- Use expensive for ambiguity, deep reasoning, major architecture decisions, and risky edits.",
    "- Create 1 to 8 tasks.",
    "",
    "User task:",
    sanitizePromptValue(params.task),
  ].join("\n");
}

function buildPipelineStepPrompt(params: {
  sourceTask: string;
  step: LoadBalancerPlanTask;
  index: number;
  total: number;
  modelId: string;
}): string {
  return [
    `You are executing model-load-balancer step ${params.index + 1} of ${params.total}.`,
    `Assigned route: ${params.step.role} (${params.modelId}).`,
    `Step title: ${params.step.title}`,
    "",
    "Original user request:",
    sanitizePromptValue(params.sourceTask),
    "",
    "Execution instructions:",
    sanitizePromptValue(params.step.instructions),
    "",
    "Return only the result for this step. Keep it concise and actionable.",
  ].join("\n");
}

function consolidateJudgeResults(results: JudgeResult[], sourceTask: string): {
  summary: string;
  tasks: LoadBalancerPlanTask[];
  raw: string;
} {
  const summaries = results
    .map((entry) => entry.parsed?.summary?.trim() || "")
    .filter((value) => Boolean(value));
  const mergedTasks: LoadBalancerPlanTask[] = [];
  const seen = new Set<string>();
  for (const result of results) {
    const tasks = result.parsed?.tasks ?? [];
    for (const task of tasks) {
      const dedupeKey = `${task.role}:${task.title.toLowerCase()}`;
      if (seen.has(dedupeKey)) {
        continue;
      }
      seen.add(dedupeKey);
      mergedTasks.push({
        id: `lb-task-${mergedTasks.length + 1}`,
        title: task.title,
        role: task.role,
        instructions: task.instructions,
        rationale: task.rationale,
      });
      if (mergedTasks.length >= MAX_PLAN_TASKS) {
        break;
      }
    }
    if (mergedTasks.length >= MAX_PLAN_TASKS) {
      break;
    }
  }

  if (mergedTasks.length === 0) {
    mergedTasks.push(
      {
        id: "lb-task-1",
        title: "Break down the request into concrete implementation steps.",
        role: "cheap",
        instructions: `Analyze and decompose this request: ${sourceTask}`,
        rationale: "Fallback decomposition when judge output is unstructured.",
      },
      {
        id: "lb-task-2",
        title: "Execute the final high-quality solution end to end.",
        role: "expensive",
        instructions: sourceTask,
        rationale: "Final synthesis requires stronger reasoning.",
      },
    );
  }

  const summary =
    summaries[0] ??
    "Fallback plan generated because judge output could not be parsed into structured JSON.";
  const raw = results
    .map((result) => `Judge ${result.judgeModelId}\n${result.raw.trim()}`)
    .join("\n\n-----\n\n");
  return {
    summary,
    tasks: mergedTasks,
    raw,
  };
}

function buildJudgeSessionKey(baseSessionKey: string, judgeModelId: string, index: number): string {
  const slug = judgeModelId
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 24);
  const safeSlug = slug || "judge";
  const stamp = Date.now().toString(36);
  return `${baseSessionKey}:load-balancer:judge:${safeSlug}:${stamp}:${index + 1}`;
}

async function runAgentAndWait(params: {
  client: GatewayBrowserClient;
  sessionKey: string;
  message: string;
  timeoutMs: number;
  pollIntervalMs?: number;
  onProgress?: (elapsedMs: number) => void;
}): Promise<void> {
  const accepted = await params.client.request<AgentAcceptedPayload>("agent", {
    sessionKey: params.sessionKey,
    message: params.message,
    deliver: false,
    idempotencyKey: generateUUID(),
  });
  const runId = typeof accepted.runId === "string" ? accepted.runId.trim() : "";
  if (!runId) {
    throw new Error("gateway did not return runId for agent execution");
  }

  const pollIntervalMs = Math.max(
    2_000,
    Math.min(params.pollIntervalMs ?? AGENT_WAIT_POLL_INTERVAL_MS, params.timeoutMs),
  );
  let dynamicPollIntervalMs = pollIntervalMs;
  const startedAt = Date.now();
  while (true) {
    const elapsedMs = Date.now() - startedAt;
    const remainingMs = Math.max(0, params.timeoutMs - elapsedMs);
    if (remainingMs <= 0) {
      throw new Error(`gateway timeout after ${params.timeoutMs}ms`);
    }

    const waited = await params.client.request<AgentWaitPayload>("agent.wait", {
      runId,
      timeoutMs: Math.min(dynamicPollIntervalMs, remainingMs),
    });
    const status =
      typeof waited.status === "string" ? waited.status.trim().toLowerCase() : "timeout";
    if (status === "ok") {
      return;
    }
    if (
      status === "timeout" ||
      status === "running" ||
      status === "queued" ||
      status === "pending" ||
      status === "in_progress"
    ) {
      params.onProgress?.(Date.now() - startedAt);
      dynamicPollIntervalMs = Math.min(
        AGENT_WAIT_MAX_POLL_INTERVAL_MS,
        Math.max(pollIntervalMs, Math.round(dynamicPollIntervalMs * 1.3)),
      );
      continue;
    }
    const detail = typeof waited.error === "string" && waited.error.trim() ? waited.error : status;
    throw new Error(detail);
  }
}

async function runJudge(params: {
  state: LoadBalancerState;
  client: GatewayBrowserClient;
  baseSessionKey: string;
  judgeModelId: string;
  index: number;
  task: string;
  cheapModel: string;
  expensiveModel: string;
  onProgress?: (elapsedMs: number) => void;
}): Promise<JudgeExecutionResult> {
  const judgeSessionKey = buildJudgeSessionKey(
    params.baseSessionKey,
    params.judgeModelId,
    params.index,
  );
  let sessionReady = false;
  try {
    await patchSessionModel(params.client, judgeSessionKey, params.judgeModelId);
    sessionReady = true;
    const beforeUsage = await fetchSessionUsageSnapshot(params.client, judgeSessionKey);

    await runAgentWithRateLimitGuard({
      state: params.state,
      modelId: params.judgeModelId,
      label: `Judge ${params.index + 1}`,
      run: () =>
        runAgentAndWait({
          client: params.client,
          sessionKey: judgeSessionKey,
          message: buildJudgePrompt({
            task: params.task,
            cheapModel: params.cheapModel,
            expensiveModel: params.expensiveModel,
          }),
          timeoutMs: JUDGE_TIMEOUT_MS,
          onProgress: params.onProgress,
        }),
    });
    const afterUsage = await fetchUsageSnapshotAfterStep({
      client: params.client,
      sessionKey: judgeSessionKey,
      beforeTotalTokens: beforeUsage?.totalTokens ?? null,
    });
    const usageDelta = computeUsageDelta(beforeUsage, afterUsage);

    const history = await params.client.request<{ messages?: unknown[] }>("chat.history", {
      sessionKey: judgeSessionKey,
      limit: 80,
    });
    const messages = Array.isArray(history.messages) ? history.messages : [];
    const text = findLatestAssistantText(messages);
    if (!text) {
      throw new Error("judge returned no assistant output");
    }
    return {
      raw: text,
      usageDelta,
    };
  } finally {
    if (sessionReady) {
      try {
        await params.client.request("sessions.delete", {
          key: judgeSessionKey,
          deleteTranscript: true,
        });
      } catch {
        // Best-effort cleanup only.
      }
    }
  }
}

export function resetLoadBalancerWorkflow(state: LoadBalancerState) {
  state.chatLoadBalancerPlanning = false;
  state.chatLoadBalancerPlan = null;
  state.chatLoadBalancerAwaitingApproval = false;
  state.chatLoadBalancerExecuting = false;
  state.chatLoadBalancerLog = [];
  state.chatLoadBalancerError = null;
}

export async function loadLoadBalancerModels(
  state: LoadBalancerState,
  opts?: { force?: boolean },
): Promise<void> {
  if (!state.client || !state.connected) {
    return;
  }
  if (state.chatLoadBalancerModelsLoading) {
    return;
  }
  if (!opts?.force && state.chatLoadBalancerModels.length > 0) {
    return;
  }

  state.chatLoadBalancerModelsLoading = true;
  state.chatLoadBalancerError = null;
  try {
    const payload = await state.client.request<{ models?: unknown[] }>("models.list", {});
    const rawModels = Array.isArray(payload.models) ? payload.models : [];
    const models = rawModels
      .map((entry) => normalizeModelOption(entry))
      .filter((entry): entry is LoadBalancerModelOption => entry !== null)
      .sort(compareModels);
    state.chatLoadBalancerModels = models;
    normalizeAssignments(state);
    if (state.chatLoadBalancerTaskInput.trim() === "") {
      state.chatLoadBalancerTaskInput = state.chatMessage.trim();
    }
  } catch (err) {
    state.chatLoadBalancerError = `Model catalog load failed: ${String(err)}`;
  } finally {
    state.chatLoadBalancerModelsLoading = false;
  }
}

export async function planLoadBalancedPipeline(state: LoadBalancerState): Promise<boolean> {
  if (!state.client || !state.connected) {
    state.chatLoadBalancerError = "Gateway is offline. Connect before planning.";
    return false;
  }
  if (state.chatLoadBalancerPlanning || state.chatLoadBalancerExecuting) {
    return false;
  }
  if (state.chatLoadBalancerModels.length === 0) {
    await loadLoadBalancerModels(state, { force: true });
  }

  const sourceTask = state.chatLoadBalancerTaskInput.trim() || state.chatMessage.trim();
  if (!sourceTask) {
    state.chatLoadBalancerError = "Add a task in the load balancer panel (or chat draft) first.";
    return false;
  }
  const cheapModel = state.chatLoadBalancerCheapModel.trim();
  const expensiveModel = state.chatLoadBalancerExpensiveModel.trim();
  if (!cheapModel || !expensiveModel) {
    state.chatLoadBalancerError = "Select both a cheap model and an expensive model.";
    return false;
  }
  if (cheapModel === expensiveModel) {
    state.chatLoadBalancerError =
      "Cheap and expensive models must be different to run a balanced pipeline.";
    return false;
  }

  const judgeModels =
    state.chatLoadBalancerJudgeModels.length > 0
      ? [...new Set(state.chatLoadBalancerJudgeModels.map((id) => id.trim()).filter(Boolean))]
      : [expensiveModel];
  if (judgeModels.length === 0) {
    state.chatLoadBalancerError = "Select at least one judge model.";
    return false;
  }

  state.chatLoadBalancerTaskInput = sourceTask;
  state.chatLoadBalancerPlanning = true;
  state.chatLoadBalancerAwaitingApproval = false;
  state.chatLoadBalancerPlan = null;
  state.chatLoadBalancerError = null;
  state.chatLoadBalancerLog = [];
  appendLog(
    state,
    `Planning started with ${judgeModels.length} judge model${judgeModels.length > 1 ? "s" : ""}.`,
  );

  const results: JudgeResult[] = [];
  try {
    for (let index = 0; index < judgeModels.length; index++) {
      const judgeModelId = judgeModels[index];
      appendLog(state, `Running judge ${index + 1}/${judgeModels.length}: ${judgeModelId}`);
      const judgeResult = await runJudge({
        state,
        client: state.client,
        baseSessionKey: state.sessionKey,
        judgeModelId,
        index,
        task: sourceTask,
        cheapModel,
        expensiveModel,
        onProgress: (elapsedMs) => {
          appendLog(
            state,
            `Judge ${judgeModelId} is still evaluating (${formatElapsedDuration(elapsedMs)} elapsed).`,
          );
        },
      });
      if (!judgeResult || typeof judgeResult.raw !== "string") {
        throw new Error(`Judge ${judgeModelId} returned invalid output.`);
      }
      logUsageDelta(state, judgeResult.usageDelta, `Judge ${judgeModelId}`);
      const raw = judgeResult.raw;
      const parsed = parseJudgePlan(raw);
      if (parsed) {
        appendLog(state, `Judge ${judgeModelId} proposed ${parsed.tasks.length} task(s).`);
      } else {
        appendLog(state, `Judge ${judgeModelId} output was unstructured; using fallback merge.`);
      }
      results.push({
        judgeModelId,
        raw,
        parsed,
      });
    }

    const merged = consolidateJudgeResults(results, sourceTask);
    state.chatLoadBalancerPlan = {
      summary: merged.summary,
      tasks: merged.tasks,
      judges: judgeModels,
      raw: merged.raw,
      sourceTask,
      createdAt: Date.now(),
    };
    state.chatLoadBalancerAwaitingApproval = true;
    appendLog(
      state,
      `Plan ready with ${merged.tasks.length} pipeline step${merged.tasks.length > 1 ? "s" : ""}.`,
    );
    appendLog(state, "Review the plan and click Start pipeline to execute.");
    return true;
  } catch (err) {
    const message = String(err);
    state.chatLoadBalancerError = `Planning failed: ${message}`;
    appendLog(state, `Planning failed: ${message}`);
    return false;
  } finally {
    state.chatLoadBalancerPlanning = false;
  }
}

export async function executeLoadBalancedPipeline(
  state: LoadBalancerState,
  hooks: LoadBalancerExecutionHooks,
): Promise<boolean> {
  if (!state.client || !state.connected) {
    state.chatLoadBalancerError = "Gateway is offline. Connect before running the pipeline.";
    return false;
  }
  if (state.chatLoadBalancerPlanning || state.chatLoadBalancerExecuting) {
    return false;
  }
  const plan = state.chatLoadBalancerPlan;
  if (!plan || !state.chatLoadBalancerAwaitingApproval) {
    state.chatLoadBalancerError = "Generate a plan first, then approve execution.";
    return false;
  }
  if (plan.tasks.length === 0) {
    state.chatLoadBalancerError = "Plan has no tasks to execute.";
    return false;
  }

  const cheapModel = state.chatLoadBalancerCheapModel.trim();
  const expensiveModel = state.chatLoadBalancerExpensiveModel.trim();
  if (!cheapModel || !expensiveModel) {
    state.chatLoadBalancerError = "Select both cheap and expensive models before execution.";
    return false;
  }
  if (cheapModel === expensiveModel) {
    state.chatLoadBalancerError =
      "Cheap and expensive models must be different to execute a balanced pipeline.";
    return false;
  }

  state.chatLoadBalancerExecuting = true;
  state.chatLoadBalancerError = null;
  appendLog(state, `Pipeline execution started (${plan.tasks.length} step(s)).`);

  const previousModel = sessionModelForKey(state.sessionsResult, state.sessionKey);
  const pipelineModelTotals = new Map<string, number>();
  try {
    for (let index = 0; index < plan.tasks.length; index++) {
      const step = plan.tasks[index];
      let modelId = step.role === "cheap" ? cheapModel : expensiveModel;
      appendLog(
        state,
        `Step ${index + 1}/${plan.tasks.length}: ${step.role.toUpperCase()} model ${modelId}`,
      );

      const runStepWithModel = async (stepModelId: string): Promise<SessionUsageDelta | null> => {
        const beforeUsage = await fetchSessionUsageSnapshot(state.client!, state.sessionKey);
        await patchSessionModel(state.client!, state.sessionKey, stepModelId);
        await runAgentWithRateLimitGuard({
          state,
          modelId: stepModelId,
          label: `Step ${index + 1}/${plan.tasks.length}`,
          run: () =>
            runAgentAndWait({
              client: state.client!,
              sessionKey: state.sessionKey,
              message: buildPipelineStepPrompt({
                sourceTask: plan.sourceTask,
                step,
                index,
                total: plan.tasks.length,
                modelId: stepModelId,
              }),
              timeoutMs: PIPELINE_STEP_TIMEOUT_MS,
              onProgress: (elapsedMs) => {
                appendLog(
                  state,
                  `Step ${index + 1}/${plan.tasks.length} on ${stepModelId} is still running (${formatElapsedDuration(elapsedMs)} elapsed).`,
                );
              },
            }),
        });
        const afterUsage = await fetchUsageSnapshotAfterStep({
          client: state.client!,
          sessionKey: state.sessionKey,
          beforeTotalTokens: beforeUsage?.totalTokens ?? null,
        });
        return computeUsageDelta(beforeUsage, afterUsage);
      };

      let usageDelta: SessionUsageDelta | null = null;
      try {
        usageDelta = await runStepWithModel(modelId);
      } catch (error) {
        const availabilityFailure = isModelAvailabilityError(error);
        const rateLimitFailure = isRateLimitError(error);
        if (!availabilityFailure && !rateLimitFailure) {
          throw error;
        }
        const fallbackModelId = pickModelFallback({
          currentModelId: modelId,
          cheapModelId: cheapModel,
          expensiveModelId: expensiveModel,
          models: state.chatLoadBalancerModels,
        });
        if (!fallbackModelId) {
          throw error;
        }
        appendLog(
          state,
          `Model ${modelId} failed (${String(error)}). Retrying step ${index + 1} with ${fallbackModelId}${rateLimitFailure ? " after rate-limit exhaustion" : ""}.`,
        );
        modelId = fallbackModelId;
        usageDelta = await runStepWithModel(modelId);
      }

      logUsageDelta(
        state,
        usageDelta,
        `Step ${index + 1}/${plan.tasks.length} (${modelId})`,
      );
      recordPipelineModelTotals(pipelineModelTotals, usageDelta);
      appendLog(state, `Step ${index + 1}/${plan.tasks.length} completed with ${modelId}.`);
      await hooks.refreshChatHistory();
    }

    state.chatLoadBalancerAwaitingApproval = false;
    logPipelineModelTotals(state, pipelineModelTotals);
    appendLog(state, "Pipeline execution completed successfully.");
    return true;
  } catch (err) {
    const message = String(err);
    state.chatLoadBalancerError = `Pipeline failed: ${message}`;
    appendLog(state, `Pipeline failed: ${message}`);
    return false;
  } finally {
    try {
      await patchSessionModel(state.client, state.sessionKey, previousModel);
      appendLog(state, "Session model override restored.");
    } catch (err) {
      appendLog(state, `Warning: failed to restore previous model (${String(err)}).`);
    }
    if (hooks.refreshSessions) {
      await hooks.refreshSessions();
    }
    await hooks.refreshChatHistory();
    state.chatLoadBalancerExecuting = false;
  }
}
