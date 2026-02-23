import { html, nothing } from "lit";
import { ref } from "lit/directives/ref.js";
import { repeat } from "lit/directives/repeat.js";
import {
  renderMessageGroup,
  renderReadingIndicatorGroup,
  renderStreamingGroup,
} from "../chat/grouped-render.ts";
import { normalizeMessage, normalizeRoleForGrouping } from "../chat/message-normalizer.ts";
import type {
  LoadBalancerModelOption,
  LoadBalancerPlan,
} from "../controllers/model-load-balancer.ts";
import { icons } from "../icons.ts";
import { detectTextDirection } from "../text-direction.ts";
import type { SessionsListResult } from "../types.ts";
import type { ChatItem, MessageGroup } from "../types/chat-types.ts";
import type { ChatAttachment, ChatQueueItem } from "../ui-types.ts";
import { renderMarkdownSidebar } from "./markdown-sidebar.ts";
import "../components/resizable-divider.ts";

export type CompactionIndicatorStatus = {
  active: boolean;
  startedAt: number | null;
  completedAt: number | null;
};

export type ChatProps = {
  sessionKey: string;
  onSessionKeyChange: (next: string) => void;
  thinkingLevel: string | null;
  showThinking: boolean;
  loading: boolean;
  sending: boolean;
  canAbort?: boolean;
  compactionStatus?: CompactionIndicatorStatus | null;
  messages: unknown[];
  toolMessages: unknown[];
  stream: string | null;
  streamStartedAt: number | null;
  assistantAvatarUrl?: string | null;
  draft: string;
  queue: ChatQueueItem[];
  connected: boolean;
  canSend: boolean;
  disabledReason: string | null;
  error: string | null;
  sessions: SessionsListResult | null;
  // Focus mode
  focusMode: boolean;
  // Sidebar state
  sidebarOpen?: boolean;
  sidebarContent?: string | null;
  sidebarError?: string | null;
  splitRatio?: number;
  assistantName: string;
  assistantAvatar: string | null;
  // Image attachments
  attachments?: ChatAttachment[];
  onAttachmentsChange?: (attachments: ChatAttachment[]) => void;
  // Scroll control
  showNewMessages?: boolean;
  onScrollToBottom?: () => void;
  // Event handlers
  onRefresh: () => void;
  onToggleFocusMode: () => void;
  onDraftChange: (next: string) => void;
  onSend: () => void;
  onAbort?: () => void;
  onQueueRemove: (id: string) => void;
  onNewSession: () => void;
  onOpenSidebar?: (content: string) => void;
  onCloseSidebar?: () => void;
  onSplitRatioChange?: (ratio: number) => void;
  onChatScroll?: (event: Event) => void;
  // Model load balancer
  loadBalancerOpen?: boolean;
  loadBalancerModelsLoading?: boolean;
  loadBalancerModels?: LoadBalancerModelOption[];
  loadBalancerCheapModel?: string;
  loadBalancerExpensiveModel?: string;
  loadBalancerJudgeModels?: string[];
  loadBalancerTaskInput?: string;
  loadBalancerPlanning?: boolean;
  loadBalancerPlan?: LoadBalancerPlan | null;
  loadBalancerAwaitingApproval?: boolean;
  loadBalancerExecuting?: boolean;
  loadBalancerLog?: string[];
  loadBalancerError?: string | null;
  onLoadBalancerToggle?: (open?: boolean) => void;
  onLoadBalancerRefreshModels?: (force?: boolean) => void;
  onLoadBalancerCheapModelChange?: (modelId: string) => void;
  onLoadBalancerExpensiveModelChange?: (modelId: string) => void;
  onLoadBalancerToggleJudge?: (modelId: string) => void;
  onLoadBalancerTaskInputChange?: (next: string) => void;
  onLoadBalancerUseDraft?: () => void;
  onLoadBalancerPlan?: () => void;
  onLoadBalancerExecute?: () => void;
  onLoadBalancerReset?: () => void;
};

const COMPACTION_TOAST_DURATION_MS = 5000;

function adjustTextareaHeight(el: HTMLTextAreaElement) {
  el.style.height = "auto";
  el.style.height = `${el.scrollHeight}px`;
}

function renderCompactionIndicator(status: CompactionIndicatorStatus | null | undefined) {
  if (!status) {
    return nothing;
  }

  // Show "compacting..." while active
  if (status.active) {
    return html`
      <div class="compaction-indicator compaction-indicator--active" role="status" aria-live="polite">
        ${icons.loader} Compacting context...
      </div>
    `;
  }

  // Show "compaction complete" briefly after completion
  if (status.completedAt) {
    const elapsed = Date.now() - status.completedAt;
    if (elapsed < COMPACTION_TOAST_DURATION_MS) {
      return html`
        <div class="compaction-indicator compaction-indicator--complete" role="status" aria-live="polite">
          ${icons.check} Context compacted
        </div>
      `;
    }
  }

  return nothing;
}

function generateAttachmentId(): string {
  return `att-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function handlePaste(e: ClipboardEvent, props: ChatProps) {
  const items = e.clipboardData?.items;
  if (!items || !props.onAttachmentsChange) {
    return;
  }

  const imageItems: DataTransferItem[] = [];
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (item.type.startsWith("image/")) {
      imageItems.push(item);
    }
  }

  if (imageItems.length === 0) {
    return;
  }

  e.preventDefault();

  for (const item of imageItems) {
    const file = item.getAsFile();
    if (!file) {
      continue;
    }

    const reader = new FileReader();
    reader.addEventListener("load", () => {
      const dataUrl = reader.result as string;
      const newAttachment: ChatAttachment = {
        id: generateAttachmentId(),
        dataUrl,
        mimeType: file.type,
      };
      const current = props.attachments ?? [];
      props.onAttachmentsChange?.([...current, newAttachment]);
    });
    reader.readAsDataURL(file);
  }
}

function renderAttachmentPreview(props: ChatProps) {
  const attachments = props.attachments ?? [];
  if (attachments.length === 0) {
    return nothing;
  }

  return html`
    <div class="chat-attachments">
      ${attachments.map(
        (att) => html`
          <div class="chat-attachment">
            <img
              src=${att.dataUrl}
              alt="Attachment preview"
              class="chat-attachment__img"
            />
            <button
              class="chat-attachment__remove"
              type="button"
              aria-label="Remove attachment"
              @click=${() => {
                const next = (props.attachments ?? []).filter((a) => a.id !== att.id);
                props.onAttachmentsChange?.(next);
              }}
            >
              ${icons.x}
            </button>
          </div>
        `,
      )}
    </div>
  `;
}

function formatModelOptionLabel(model: LoadBalancerModelOption): string {
  const base = model.name && model.name !== model.id ? `${model.name} (${model.id})` : model.id;
  return model.provider ? `${model.provider} · ${base}` : base;
}

function renderLoadBalancerPlan(plan: LoadBalancerPlan | null) {
  if (!plan) {
    return nothing;
  }
  return html`
    <div class="chat-load-balancer__plan">
      <div class="chat-load-balancer__plan-header">
        <div class="chat-load-balancer__plan-title">Planned Pipeline</div>
        <div class="chat-load-balancer__plan-meta">
          ${plan.tasks.length} step${plan.tasks.length === 1 ? "" : "s"} · Judge
          ${plan.judges.length === 1 ? "" : "s"}: ${plan.judges.join(", ")}
        </div>
      </div>
      <div class="chat-load-balancer__plan-summary">${plan.summary}</div>
      <div class="chat-load-balancer__task-list">
        ${plan.tasks.map(
          (task, index) => html`
            <div class="chat-load-balancer__task">
              <div class="chat-load-balancer__task-line">
                <span
                  class="chat-load-balancer__route chat-load-balancer__route--${task.role}"
                  >${task.role.toUpperCase()}</span>
                <span class="chat-load-balancer__task-index">Step ${index + 1}</span>
              </div>
              <div class="chat-load-balancer__task-title">${task.title}</div>
              <div class="chat-load-balancer__task-instructions">${task.instructions}</div>
              ${task.rationale
                ? html`<div class="chat-load-balancer__task-rationale">${task.rationale}</div>`
                : nothing}
            </div>
          `,
        )}
      </div>
      <details class="chat-load-balancer__raw">
        <summary>Judge output (raw)</summary>
        <pre class="code-block">${plan.raw}</pre>
      </details>
    </div>
  `;
}

function renderLoadBalancerPanel(props: ChatProps) {
  if (!props.loadBalancerOpen) {
    return nothing;
  }
  const models = props.loadBalancerModels ?? [];
  const modelCount = models.length;
  const judges = new Set(props.loadBalancerJudgeModels ?? []);
  const cheapModel = props.loadBalancerCheapModel ?? "";
  const expensiveModel = props.loadBalancerExpensiveModel ?? "";
  const taskInput = props.loadBalancerTaskInput ?? "";
  const planning = Boolean(props.loadBalancerPlanning);
  const executing = Boolean(props.loadBalancerExecuting);
  const awaitingApproval = Boolean(props.loadBalancerAwaitingApproval);
  const log = props.loadBalancerLog ?? [];
  const liveLogLines = log.slice(-80);
  const canPlan =
    modelCount > 0 &&
    cheapModel !== "" &&
    expensiveModel !== "" &&
    cheapModel !== expensiveModel &&
    taskInput.trim().length > 0 &&
    judges.size > 0 &&
    !planning &&
    !executing;
  const canStartPipeline =
    awaitingApproval &&
    !planning &&
    !executing &&
    cheapModel !== "" &&
    expensiveModel !== "" &&
    cheapModel !== expensiveModel;

  return html`
    <section class="chat-load-balancer">
      <div class="chat-load-balancer__header">
        <div>
          <div class="chat-load-balancer__title">Model Load Balancer</div>
          <div class="chat-load-balancer__subtitle">
            Route easy work to cheap models, hard work to expensive models, and execute only after
            manual approval.
          </div>
        </div>
        <div class="chat-load-balancer__header-actions">
          <button
            class="btn btn--sm"
            ?disabled=${Boolean(props.loadBalancerModelsLoading) || planning}
            @click=${() => props.onLoadBalancerRefreshModels?.(true)}
          >
            ${props.loadBalancerModelsLoading ? "Loading..." : "Refresh models"}
          </button>
          <button
            class="btn btn--sm"
            ?disabled=${planning || executing}
            @click=${() => props.onLoadBalancerReset?.()}
          >
            Clear state
          </button>
          <button
            class="btn btn--sm"
            ?disabled=${planning || executing}
            @click=${() => props.onLoadBalancerToggle?.(false)}
          >
            ${icons.x}
          </button>
        </div>
      </div>

      <div class="chat-load-balancer__body">
        <div class="chat-load-balancer__activity">
          <div class="chat-load-balancer__activity-header">
            <span>Live model activity</span>
            <span>${liveLogLines.length} entr${liveLogLines.length === 1 ? "y" : "ies"}</span>
          </div>
          <pre class="chat-load-balancer__activity-stream">${liveLogLines.length > 0
            ? liveLogLines.join("\n")
            : "No model activity yet. Generate a plan or run the pipeline to stream logs here."}</pre>
        </div>

        ${
          props.loadBalancerError
            ? html`<div class="callout danger">${props.loadBalancerError}</div>`
            : nothing
        }

        ${modelCount === 0
          ? html`
              <div class="callout">No models discovered yet. Click "Refresh models".</div>
            `
          : html`
              <div class="chat-load-balancer__selectors">
                <label class="field">
                  <span>Cheap model</span>
                  <select
                    .value=${cheapModel}
                    ?disabled=${planning || executing}
                    @change=${(event: Event) =>
                      props.onLoadBalancerCheapModelChange?.(
                        (event.target as HTMLSelectElement).value,
                      )}
                  >
                    ${models.map(
                      (model) =>
                        html`<option value=${model.ref}>${formatModelOptionLabel(model)}</option>`,
                    )}
                  </select>
                </label>
                <label class="field">
                  <span>Expensive model</span>
                  <select
                    .value=${expensiveModel}
                    ?disabled=${planning || executing}
                    @change=${(event: Event) =>
                      props.onLoadBalancerExpensiveModelChange?.(
                        (event.target as HTMLSelectElement).value,
                      )}
                  >
                    ${models.map(
                      (model) =>
                        html`<option value=${model.ref}>${formatModelOptionLabel(model)}</option>`,
                    )}
                  </select>
                </label>
              </div>
              <div class="chat-load-balancer__judges">
                <div class="chat-load-balancer__judges-title">Judges (choose one or more)</div>
                <div class="chat-load-balancer__judge-list">
                  ${models.map((model) => {
                    const checked = judges.has(model.ref);
                    return html`
                      <label class="chat-load-balancer__judge">
                        <input
                          type="checkbox"
                          .checked=${checked}
                          ?disabled=${planning || executing}
                          @change=${() => props.onLoadBalancerToggleJudge?.(model.ref)}
                        />
                        <span>${formatModelOptionLabel(model)}</span>
                      </label>
                    `;
                  })}
                </div>
              </div>
            `}

        <label class="field">
          <span>Task for judges</span>
          <textarea
            .value=${taskInput}
            ?disabled=${planning || executing}
            rows="4"
            placeholder="Describe the user task to split and route..."
            @input=${(event: Event) =>
              props.onLoadBalancerTaskInputChange?.((event.target as HTMLTextAreaElement).value)}
          ></textarea>
        </label>
        <div class="chat-load-balancer__actions">
          <button
            class="btn btn--sm"
            ?disabled=${planning || executing}
            @click=${() => props.onLoadBalancerUseDraft?.()}
          >
            Use chat draft
          </button>
          <button
            class="btn btn--sm primary"
            ?disabled=${!canPlan}
            @click=${() => props.onLoadBalancerPlan?.()}
          >
            ${planning ? "Planning..." : "Generate plan"}
          </button>
          <button
            class="btn btn--sm primary"
            ?disabled=${!canStartPipeline}
            @click=${() => props.onLoadBalancerExecute?.()}
          >
            ${executing ? "Running..." : "Start pipeline"}
          </button>
        </div>

        ${renderLoadBalancerPlan(props.loadBalancerPlan ?? null)}
      </div>
    </section>
  `;
}

export function renderChat(props: ChatProps) {
  const canCompose = props.connected;
  const isBusy = props.sending || props.stream !== null;
  const canAbort = Boolean(props.canAbort && props.onAbort);
  const activeSession = props.sessions?.sessions?.find((row) => row.key === props.sessionKey);
  const reasoningLevel = activeSession?.reasoningLevel ?? "off";
  const showReasoning = props.showThinking && reasoningLevel !== "off";
  const assistantIdentity = {
    name: props.assistantName,
    avatar: props.assistantAvatar ?? props.assistantAvatarUrl ?? null,
  };

  const hasAttachments = (props.attachments?.length ?? 0) > 0;
  const composePlaceholder = props.connected
    ? hasAttachments
      ? "Add a message or paste more images..."
      : "Message (↩ to send, Shift+↩ for line breaks, paste images)"
    : "Connect to the gateway to start chatting…";

  const splitRatio = props.splitRatio ?? 0.6;
  const sidebarOpen = Boolean(props.sidebarOpen && props.onCloseSidebar);
  const thread = html`
    <div
      class="chat-thread"
      role="log"
      aria-live="polite"
      @scroll=${props.onChatScroll}
    >
      ${
        props.loading
          ? html`
              <div class="muted">Loading chat…</div>
            `
          : nothing
      }
      ${repeat(
        buildChatItems(props),
        (item) => item.key,
        (item) => {
          if (item.kind === "divider") {
            return html`
              <div class="chat-divider" role="separator" data-ts=${String(item.timestamp)}>
                <span class="chat-divider__line"></span>
                <span class="chat-divider__label">${item.label}</span>
                <span class="chat-divider__line"></span>
              </div>
            `;
          }

          if (item.kind === "reading-indicator") {
            return renderReadingIndicatorGroup(assistantIdentity);
          }

          if (item.kind === "stream") {
            return renderStreamingGroup(
              item.text,
              item.startedAt,
              props.onOpenSidebar,
              assistantIdentity,
            );
          }

          if (item.kind === "group") {
            return renderMessageGroup(item, {
              onOpenSidebar: props.onOpenSidebar,
              showReasoning,
              assistantName: props.assistantName,
              assistantAvatar: assistantIdentity.avatar,
            });
          }

          return nothing;
        },
      )}
    </div>
  `;

  return html`
    <section class="card chat">
      ${props.disabledReason ? html`<div class="callout">${props.disabledReason}</div>` : nothing}

      ${props.error ? html`<div class="callout danger">${props.error}</div>` : nothing}

      ${renderLoadBalancerPanel(props)}

      ${
        props.focusMode
          ? html`
            <button
              class="chat-focus-exit"
              type="button"
              @click=${props.onToggleFocusMode}
              aria-label="Exit focus mode"
              title="Exit focus mode"
            >
              ${icons.x}
            </button>
          `
          : nothing
      }

      <div
        class="chat-split-container ${sidebarOpen ? "chat-split-container--open" : ""}"
      >
        <div
          class="chat-main"
          style="flex: ${sidebarOpen ? `0 0 ${splitRatio * 100}%` : "1 1 100%"}"
        >
          ${thread}
        </div>

        ${
          sidebarOpen
            ? html`
              <resizable-divider
                .splitRatio=${splitRatio}
                @resize=${(e: CustomEvent) => props.onSplitRatioChange?.(e.detail.splitRatio)}
              ></resizable-divider>
              <div class="chat-sidebar">
                ${renderMarkdownSidebar({
                  content: props.sidebarContent ?? null,
                  error: props.sidebarError ?? null,
                  onClose: props.onCloseSidebar!,
                  onViewRawText: () => {
                    if (!props.sidebarContent || !props.onOpenSidebar) {
                      return;
                    }
                    props.onOpenSidebar(`\`\`\`\n${props.sidebarContent}\n\`\`\``);
                  },
                })}
              </div>
            `
            : nothing
        }
      </div>

      ${
        props.queue.length
          ? html`
            <div class="chat-queue" role="status" aria-live="polite">
              <div class="chat-queue__title">Queued (${props.queue.length})</div>
              <div class="chat-queue__list">
                ${props.queue.map(
                  (item) => html`
                    <div class="chat-queue__item">
                      <div class="chat-queue__text">
                        ${
                          item.text ||
                          (item.attachments?.length ? `Image (${item.attachments.length})` : "")
                        }
                      </div>
                      <button
                        class="btn chat-queue__remove"
                        type="button"
                        aria-label="Remove queued message"
                        @click=${() => props.onQueueRemove(item.id)}
                      >
                        ${icons.x}
                      </button>
                    </div>
                  `,
                )}
              </div>
            </div>
          `
          : nothing
      }

      ${renderCompactionIndicator(props.compactionStatus)}

      ${
        props.showNewMessages
          ? html`
            <button
              class="btn chat-new-messages"
              type="button"
              @click=${props.onScrollToBottom}
            >
              New messages ${icons.arrowDown}
            </button>
          `
          : nothing
      }

      <div class="chat-compose">
        ${renderAttachmentPreview(props)}
        <div class="chat-compose__row">
          <label class="field chat-compose__field">
            <span>Message</span>
            <textarea
              ${ref((el) => el && adjustTextareaHeight(el as HTMLTextAreaElement))}
              .value=${props.draft}
              dir=${detectTextDirection(props.draft)}
              ?disabled=${!props.connected}
              @keydown=${(e: KeyboardEvent) => {
                if (e.key !== "Enter") {
                  return;
                }
                if (e.isComposing || e.keyCode === 229) {
                  return;
                }
                if (e.shiftKey) {
                  return;
                } // Allow Shift+Enter for line breaks
                if (!props.connected) {
                  return;
                }
                e.preventDefault();
                if (canCompose) {
                  props.onSend();
                }
              }}
              @input=${(e: Event) => {
                const target = e.target as HTMLTextAreaElement;
                adjustTextareaHeight(target);
                props.onDraftChange(target.value);
              }}
              @paste=${(e: ClipboardEvent) => handlePaste(e, props)}
              placeholder=${composePlaceholder}
            ></textarea>
          </label>
          <div class="chat-compose__actions">
            <button
              class="btn"
              ?disabled=${!props.connected || (!canAbort && props.sending)}
              @click=${canAbort ? props.onAbort : props.onNewSession}
            >
              ${canAbort ? "Stop" : "New session"}
            </button>
            <button
              class="btn primary"
              ?disabled=${!props.connected}
              @click=${props.onSend}
            >
              ${isBusy ? "Queue" : "Send"}<kbd class="btn-kbd">↵</kbd>
            </button>
          </div>
        </div>
      </div>
    </section>
  `;
}

const CHAT_HISTORY_RENDER_LIMIT = 200;

function groupMessages(items: ChatItem[]): Array<ChatItem | MessageGroup> {
  const result: Array<ChatItem | MessageGroup> = [];
  let currentGroup: MessageGroup | null = null;

  for (const item of items) {
    if (item.kind !== "message") {
      if (currentGroup) {
        result.push(currentGroup);
        currentGroup = null;
      }
      result.push(item);
      continue;
    }

    const normalized = normalizeMessage(item.message);
    const role = normalizeRoleForGrouping(normalized.role);
    const timestamp = normalized.timestamp || Date.now();

    if (!currentGroup || currentGroup.role !== role) {
      if (currentGroup) {
        result.push(currentGroup);
      }
      currentGroup = {
        kind: "group",
        key: `group:${role}:${item.key}`,
        role,
        messages: [{ message: item.message, key: item.key }],
        timestamp,
        isStreaming: false,
      };
    } else {
      currentGroup.messages.push({ message: item.message, key: item.key });
    }
  }

  if (currentGroup) {
    result.push(currentGroup);
  }
  return result;
}

function buildChatItems(props: ChatProps): Array<ChatItem | MessageGroup> {
  const items: ChatItem[] = [];
  const history = Array.isArray(props.messages) ? props.messages : [];
  const tools = Array.isArray(props.toolMessages) ? props.toolMessages : [];
  const historyStart = Math.max(0, history.length - CHAT_HISTORY_RENDER_LIMIT);
  if (historyStart > 0) {
    items.push({
      kind: "message",
      key: "chat:history:notice",
      message: {
        role: "system",
        content: `Showing last ${CHAT_HISTORY_RENDER_LIMIT} messages (${historyStart} hidden).`,
        timestamp: Date.now(),
      },
    });
  }
  for (let i = historyStart; i < history.length; i++) {
    const msg = history[i];
    const normalized = normalizeMessage(msg);
    const raw = msg as Record<string, unknown>;
    const marker = raw.__openclaw as Record<string, unknown> | undefined;
    if (marker && marker.kind === "compaction") {
      items.push({
        kind: "divider",
        key:
          typeof marker.id === "string"
            ? `divider:compaction:${marker.id}`
            : `divider:compaction:${normalized.timestamp}:${i}`,
        label: "Compaction",
        timestamp: normalized.timestamp ?? Date.now(),
      });
      continue;
    }

    if (!props.showThinking && normalized.role.toLowerCase() === "toolresult") {
      continue;
    }

    items.push({
      kind: "message",
      key: messageKey(msg, i),
      message: msg,
    });
  }
  if (props.showThinking) {
    for (let i = 0; i < tools.length; i++) {
      items.push({
        kind: "message",
        key: messageKey(tools[i], i + history.length),
        message: tools[i],
      });
    }
  }

  if (props.stream !== null) {
    const key = `stream:${props.sessionKey}:${props.streamStartedAt ?? "live"}`;
    if (props.stream.trim().length > 0) {
      items.push({
        kind: "stream",
        key,
        text: props.stream,
        startedAt: props.streamStartedAt ?? Date.now(),
      });
    } else {
      items.push({ kind: "reading-indicator", key });
    }
  }

  return groupMessages(items);
}

function messageKey(message: unknown, index: number): string {
  const m = message as Record<string, unknown>;
  const toolCallId = typeof m.toolCallId === "string" ? m.toolCallId : "";
  if (toolCallId) {
    return `tool:${toolCallId}`;
  }
  const id = typeof m.id === "string" ? m.id : "";
  if (id) {
    return `msg:${id}`;
  }
  const messageId = typeof m.messageId === "string" ? m.messageId : "";
  if (messageId) {
    return `msg:${messageId}`;
  }
  const timestamp = typeof m.timestamp === "number" ? m.timestamp : null;
  const role = typeof m.role === "string" ? m.role : "unknown";
  if (timestamp != null) {
    return `msg:${role}:${timestamp}:${index}`;
  }
  return `msg:${role}:${index}`;
}
