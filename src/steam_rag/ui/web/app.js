const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const elements = {
  form: $("#searchForm"), question: $("#question"), submit: $("#submitButton"),
  hero: $("#hero"), landing: $("#landing"), loading: $("#loading"), result: $("#result"), error: $("#error"),
  answer: $("#answer"), games: $("#games"), sources: $("#sources"), trace: $("#agentTrace"),
  gameSection: $("#gameSection"), sourceSection: $("#sourceSection"), variants: $("#queryVariants"),
  coverage: $("#coverage"), coverageBar: $("#coverageBar"), coverageValue: $("#coverageValue"),
  history: $("#history"), conversation: $("#conversationMessages"),
  loadingText: $("#loadingText"), errorMessage: $("#errorMessage"),
};

const loadingMessages = [
  "질문 의도와 게임 이름을 파악하는 중...",
  "별칭과 번역명을 확장해 Steam에서 확인하는 중...",
  "게임별 문서와 최신 정보를 조사하는 중...",
  "검색된 근거가 답변에 충분한지 검증하는 중...",
  "읽기 쉬운 답변으로 정리하는 중...",
];
let loadingTimer;
let lastAnswer = "";
let requestSequence = 0;
let activeMode = "chat";
const conversationRequests = new Map();
let conversations = readConversations();
let activeConversationId = conversations[0]?.id || "";

const modeCopy = {
  chat: {
    eyebrow: "STEAM GAME INTELLIGENCE",
    title: "Steam 게임을 물어보세요",
    description: "추천, 게임 분석과 업데이트 질문을 한 대화에서 이어갈 수 있습니다.",
    placeholder: "메시지를 입력하세요",
  },
};
if (!activeConversationId) activeConversationId = createConversation("chat").id;

function escapeHtml(value = "") {
  return String(value).replace(/[&<>'"]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[char]);
}

function renderMarkdown(markdown = "") {
  const lines = String(markdown).split(/\r?\n/);
  let html = "";
  let list = null;
  const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
  for (const raw of lines) {
    const line = escapeHtml(raw).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\[근거\s*([^\]]+)\]/g, "<em>[근거 $1]</em>");
    if (/^###\s+/.test(line)) { closeList(); html += `<h3>${line.replace(/^###\s+/, "")}</h3>`; continue; }
    if (/^##\s+/.test(line)) { closeList(); html += `<h2>${line.replace(/^##\s+/, "")}</h2>`; continue; }
    const ordered = line.match(/^(\d+)\.\s+(.+)/);
    const bullet = line.match(/^[-*]\s+(.+)/);
    if (ordered) { if (list !== "ol") { closeList(); list = "ol"; html += "<ol>"; } html += `<li>${ordered[2]}</li>`; continue; }
    if (bullet) { if (list !== "ul") { closeList(); list = "ul"; html += "<ul>"; } html += `<li>${bullet[1]}</li>`; continue; }
    closeList();
    if (line.trim()) html += `<p>${line}</p>`;
  }
  closeList();
  return html;
}

function setView(name) {
  elements.hero.classList.toggle("hidden", name !== "landing");
  elements.landing.classList.toggle("hidden", name !== "landing");
  elements.loading.classList.toggle("hidden", name !== "loading");
  elements.result.classList.toggle("hidden", name !== "result");
  elements.error.classList.toggle("hidden", name !== "error");
}

function beginLoading(conversation = activeConversation()) {
  elements.submit.disabled = true;
  renderConversation(conversation);
  renderGames([]);
  renderSources([]);
  renderTrace([], [], {});
  setView("result");
}

function startLoadingAnimation() {
  let step = 0; elements.loadingText.textContent = loadingMessages[0];
  clearInterval(loadingTimer);
  loadingTimer = setInterval(() => { step = Math.min(step + 1, loadingMessages.length - 1); elements.loadingText.textContent = loadingMessages[step]; }, 2300);
}

function endLoading() { clearInterval(loadingTimer); elements.submit.disabled = false; }

function applyMode() {
  activeMode = "chat";
  const copy = modeCopy.chat;
  $("#heroEyebrow").textContent = copy.eyebrow;
  $("#heroTitle").textContent = copy.title;
  $("#heroDescription").textContent = copy.description;
  elements.question.placeholder = copy.placeholder;
}

function selectConversation(id) {
  const current = activeConversation();
  if (current && current.id !== id) current.draft = elements.question.value;
  const conversation = conversations.find((item) => item.id === id);
  if (!conversation) return;
  activeConversationId = conversation.id;
  clearInterval(loadingTimer);
  applyMode();
  elements.question.value = conversation.draft || "";
  elements.submit.disabled = conversation.view === "loading";
  if (conversation.data && conversation.view === "result") {
    renderResult(conversation.data, {conversation, store: false});
  } else if (conversation.view === "error") {
    elements.errorMessage.textContent = conversation.error;
    setView("error");
  } else if (conversation.messages.length) {
    renderConversation(conversation);
    setView("result");
  } else {
    setView(conversation.view === "loading" ? "loading" : "landing");
    if (conversation.view === "loading") startLoadingAnimation();
  }
  persistConversations();
  loadHistory();
  elements.question.focus();
}

function renderGames(games = []) {
  const unique = [...new Map(games.filter((g) => g && g.appid).map((g) => [g.appid, g])).values()];
  elements.gameSection.classList.toggle("hidden", !unique.length);
  $("#gameCount").textContent = `${unique.length}개`;
  elements.games.innerHTML = unique.map((game) => {
    const tags = [...(game.matched_tags || []), ...(game.matched_facets || [])].slice(0, 4);
    if (game.positive_ratio != null) tags.unshift(`긍정 ${Math.round(game.positive_ratio * 100)}%`);
    if (game.discount_percent) tags.unshift(`${game.discount_percent}% 할인`);
    if (!tags.length && game.status) tags.push(game.status);
    const image = game.image || `https://shared.cloudflare.steamstatic.com/store_item_assets/steam/apps/${game.appid}/header.jpg`;
    return `<a class="game-card" href="${escapeHtml(game.url || `https://store.steampowered.com/app/${game.appid}/`)}" target="_blank" rel="noreferrer">
      <img src="${escapeHtml(image)}" data-appid="${escapeHtml(game.appid)}" alt="${escapeHtml(game.name)}" loading="lazy">
      <div class="game-card-body"><h4>${escapeHtml(game.name)}</h4><div class="game-meta">${tags.map((tag, i) => `<span class="${i === 0 && /긍정|할인/.test(tag) ? "game-score" : ""}">${escapeHtml(tag)}</span>`).join("")}</div></div>
    </a>`;
  }).join("");
  elements.games.querySelectorAll("img[data-appid]").forEach((image) => {
    const appid = image.dataset.appid;
    const alternatives = [
      `https://shared.cloudflare.steamstatic.com/store_item_assets/steam/apps/${appid}/header.jpg`,
      `https://cdn.akamai.steamstatic.com/steam/apps/${appid}/header.jpg`,
    ].filter((url) => url !== image.src);
    let fallbackIndex = 0;
    image.addEventListener("error", () => {
      if (fallbackIndex < alternatives.length) {
        image.src = alternatives[fallbackIndex++];
        return;
      }
      image.classList.add("image-unavailable");
      image.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 460 215'%3E%3Crect width='460' height='215' fill='%2314243a'/%3E%3Cpath d='M195 78h70v59h-70z' fill='none' stroke='%23556b86' stroke-width='5'/%3E%3Cpath d='m204 126 18-20 14 13 10-9 11 16' fill='none' stroke='%23556b86' stroke-width='5'/%3E%3C/svg%3E";
    });
  });
}

function renderSources(sources = []) {
  elements.sourceSection.classList.toggle("hidden", !sources.length);
  $("#sourceCount").textContent = `${sources.length}개`;
  elements.sources.innerHTML = sources.map((source, index) => {
    const tag = source.url ? "a" : "div";
    const link = source.url ? ` href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer"` : "";
    return `<${tag} class="source-card"${link}><span class="source-rank">${source.rank || index + 1}</span><div><b>${escapeHtml(source.game || source.title || "근거")}</b><p>${escapeHtml(source.snippet || source.title || source.url || "")}</p></div><small>${escapeHtml(source.section || "source")}${source.date ? ` · ${escapeHtml(source.date)}` : ""}</small></${tag}>`;
  }).join("");
}

function renderTrace(agents = [], variants = [], coverage = {}) {
  elements.trace.innerHTML = agents.map((step) => `<div class="agent-step ${/completed|sufficient/.test(step.status || "") ? "success" : ""}"><i class="agent-dot"></i><b>${escapeHtml(step.agent || "Agent")}</b><p>${escapeHtml(step.detail || step.status || "완료")}</p></div>`).join("");
  elements.variants.classList.toggle("hidden", !variants.length);
  elements.variants.innerHTML = variants.length ? `<b>검색어 확장</b>${variants.map((v) => `<span>${escapeHtml(v)}</span>`).join("")}` : "";
  const ratio = Number(coverage.coverage_ratio);
  const valid = Number.isFinite(ratio);
  elements.coverage.classList.toggle("hidden", !valid);
  if (valid) { const percent = Math.round(ratio * 100); elements.coverageValue.textContent = `${percent}%`; requestAnimationFrame(() => elements.coverageBar.style.width = `${percent}%`); }
}

function renderResult(data, {conversation = activeConversation(), store = true} = {}) {
  if (!conversation) return;
  if (store) { conversation.data = data; conversation.view = "result"; }
  lastAnswer = data.answer || "";
  $("#resultMode").textContent = data.mode === "recommendation" ? "AI RECOMMENDATION" : "EVIDENCE-BASED ANALYSIS";
  renderConversation(conversation);
  renderGames([]);
  renderSources([]);
  renderTrace(data.agents || [], data.query_variants || [], data.evidence_coverage || {});
  setView("result");
}

function renderConversation(conversation = activeConversation()) {
  if (!conversation) return;
  const messages = conversation.messages.map((message) => `
    <article class="chat-message ${message.role}${message.error ? " error" : ""}">
      <div class="chat-avatar">${message.role === "user" ? "나" : "S"}</div>
      <div class="chat-body"><b>${message.role === "user" ? "나" : "SteamLens AI"}</b>
        <div>${message.role === "assistant" ? renderMarkdown(message.content) : `<p>${escapeHtml(message.content)}</p>`}</div>
        ${message.role === "assistant" ? renderTurnArtifacts(message.data) : ""}
      </div>
    </article>`).join("");
  const typing = conversation.view === "loading" ? `
    <article class="chat-message assistant typing-message">
      <div class="chat-avatar">S</div>
      <div class="chat-body"><b>SteamLens AI</b><div class="typing-dots"><i></i><i></i><i></i></div></div>
    </article>` : "";
  elements.conversation.innerHTML = messages + typing;
}

function renderTurnArtifacts(data = null) {
  if (!data) return "";
  const games = [...new Map((data.games || []).filter((game) => game?.appid).map((game) => [game.appid, game])).values()];
  const sources = (data.sources || []).slice(0, 6);
  const gameHtml = games.length ? `<div class="turn-artifacts"><b>확인된 게임</b><div class="turn-games">${games.map((game) => `
    <a href="${escapeHtml(game.url || `https://store.steampowered.com/app/${game.appid}/`)}" target="_blank" rel="noreferrer">
      <span>${escapeHtml(game.name)}</span><small>${escapeHtml(game.status || "Steam에서 확인")}</small>
    </a>`).join("")}</div></div>` : "";
  const sourceHtml = sources.length ? `<details class="turn-sources"><summary>사용한 근거 ${sources.length}개</summary>${sources.map((source, index) => `
    <${source.url ? "a" : "div"} class="turn-source"${source.url ? ` href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer"` : ""}>
      <span>${index + 1}</span><div><b>${escapeHtml(source.game || source.title || "근거")}</b><small>${escapeHtml(source.snippet || source.title || "")}</small></div>
    </${source.url ? "a" : "div"}>`).join("")}</details>` : "";
  return gameHtml + sourceHtml;
}

function compactResponse(data) {
  const answer = String(data.answer || "").trim();
  return {
    mode: data.mode,
    answer: answer || "검색 근거는 확인했지만 설명형 답변을 만들지 못했습니다. 같은 대화에서 원하는 관점을 더 구체적으로 알려주세요.",
    query_variants: data.query_variants || [],
    agents: data.agents || [],
    games: data.games || [],
    sources: data.sources || [],
    evidence_coverage: data.evidence_coverage || {},
    resolved_question: data.resolved_question || "",
    conversation_context_used: Boolean(data.conversation_context_used),
    intent_route: data.intent_route || data.mode || "",
    conversation_state: data.conversation_state || {},
  };
}

function activeConversation() {
  return conversations.find((item) => item.id === activeConversationId) || null;
}

function createConversation(mode = activeMode) {
  const conversation = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    title: "새 대화",
    mode: "chat",
    messages: [],
    data: null,
    view: "landing",
    error: "",
    draft: "",
    state: {},
    updatedAt: Date.now(),
  };
  conversations.unshift(conversation);
  persistConversations();
  return conversation;
}

function readConversations() {
  try {
    const value = JSON.parse(localStorage.getItem("steamlens-conversations-v1") || "[]");
    return Array.isArray(value) ? value.filter((item) => item && item.id && Array.isArray(item.messages)) : [];
  } catch (_) { return []; }
}

function persistConversations() {
  conversations.sort((a, b) => Number(b.updatedAt || 0) - Number(a.updatedAt || 0));
  conversations = conversations.slice(0, 20);
  try { localStorage.setItem("steamlens-conversations-v1", JSON.stringify(conversations)); } catch (_) {}
}

function loadHistory() {
  elements.history.innerHTML = conversations.length ? conversations.map((conversation) => `
    <div class="history-row ${conversation.id === activeConversationId ? "active" : ""}">
      <button class="history-question" data-conversation-id="${escapeHtml(conversation.id)}" title="${escapeHtml(conversation.title)}">${escapeHtml(conversation.title)}</button>
      <button class="history-more" data-history-menu="${escapeHtml(conversation.id)}" aria-label="대화방 메뉴" aria-expanded="false">···</button>
      <div class="history-menu hidden" data-history-popover="${escapeHtml(conversation.id)}">
        <button class="history-delete" data-history-delete="${escapeHtml(conversation.id)}">삭제</button>
      </div>
    </div>`).join("") : '<div class="history-empty">아직 대화가 없습니다</div>';
}

function closeHistoryMenus() {
  $$("[data-history-popover]").forEach((menu) => menu.classList.add("hidden"));
  $$("[data-history-menu]").forEach((button) => button.setAttribute("aria-expanded", "false"));
}

function deleteHistory(id) {
  const index = conversations.findIndex((item) => item.id === id);
  if (index < 0) return;
  const request = conversationRequests.get(id);
  if (request?.controller) request.controller.abort();
  conversationRequests.delete(id);
  conversations.splice(index, 1);
  if (!conversations.length) createConversation(activeMode);
  if (activeConversationId === id) activeConversationId = conversations[0].id;
  persistConversations();
  loadHistory();
  selectConversation(activeConversationId);
}

function makeRequestId() {
  if (globalThis.crypto?.randomUUID) return crypto.randomUUID().replaceAll("-", "_");
  return `${Date.now()}_${Math.random().toString(36).slice(2, 12)}`;
}

async function postChat(payload, signal) {
  let networkError = null;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      return await fetch("/api/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
        signal,
      });
    } catch (error) {
      if (error.name === "AbortError") throw error;
      networkError = error;
      if (attempt === 0) await new Promise((resolve) => setTimeout(resolve, 700));
    }
  }
  try {
    const health = await fetch("/api/health", {cache: "no-store", signal});
    if (health.ok) {
      throw new Error("서버 연결이 일시적으로 중단됐습니다. 같은 질문을 다시 보내 주세요.");
    }
  } catch (healthError) {
    if (healthError.name === "AbortError") throw healthError;
    if (/일시적으로 중단/.test(healthError.message || "")) throw healthError;
  }
  throw new Error(
    "서비스 서버와 연결할 수 없습니다. 실행 중인 터미널을 확인한 뒤 scripts\\run_service.py를 다시 시작해 주세요.",
    {cause: networkError},
  );
}

async function submitQuestion(question) {
  question = String(question || "").trim(); if (!question) return;
  const conversation = activeConversation() || createConversation(activeMode);
  const conversationId = conversation.id;
  const requestState = conversationRequests.get(conversationId) || {controller: null, id: 0};
  if (requestState.controller) requestState.controller.abort();
  const requestController = new AbortController();
  const requestId = ++requestSequence;
  const clientRequestId = makeRequestId();
  requestState.controller = requestController;
  requestState.id = requestId;
  conversationRequests.set(conversationId, requestState);
  const history = conversation.messages
    .filter((message) => message.role === "user" && !message.error)
    .slice(-8)
    .map(({role, content}) => ({role, content}));
  const latestGameTurn = [...conversation.messages]
    .reverse()
    .find((message) => message.role === "assistant" && message.data?.games?.length);
  const conversation_state = conversation.state
    || conversation.data?.conversation_state
    || {};
  const stateGames = Array.isArray(conversation_state.active_games)
    ? conversation_state.active_games
    : [];
  const context_games = [...new Map((stateGames.length ? stateGames : (latestGameTurn?.data?.games || []))
    .filter((game) => game?.appid && game?.name)
    .map((game) => [Number(game.appid), {appid: Number(game.appid), name: String(game.name)}]))
    .values()].slice(0, 10);
  conversation.messages.push({role: "user", content: question});
  conversation.title = conversation.messages.filter((item) => item.role === "user").length === 1
    ? question.slice(0, 36) + (question.length > 36 ? "…" : "")
    : conversation.title;
  conversation.draft = "";
  conversation.view = "loading";
  conversation.error = "";
  conversation.updatedAt = Date.now();
  elements.question.value = "";
  persistConversations(); loadHistory(); beginLoading(conversation);
  try {
    const response = await postChat(
      {question, history, context_games, conversation_state, top_k: 6, request_id: clientRequestId},
      requestController.signal,
    );
    const data = await response.json();
    if (requestId !== requestState.id) return;
    if (!response.ok) throw new Error(data.detail || "서버 요청에 실패했습니다.");
    const storedData = compactResponse(data);
    conversation.messages.push({
      role: "assistant",
      content: storedData.answer || "답변을 생성하지 못했습니다.",
      data: storedData,
    });
    conversation.data = storedData;
    conversation.state = storedData.conversation_state || conversation.state || {};
    conversation.view = "result";
    conversation.updatedAt = Date.now();
    persistConversations(); loadHistory();
    if (activeConversationId === conversationId) renderResult(storedData, {conversation, store: false});
  } catch (error) {
    if (error.name === "AbortError" || requestId !== requestState.id) return;
    const message = error.message || String(error);
    conversation.messages.push({
      role: "assistant",
      content: `요청을 처리하지 못했습니다.\n\n${message}`,
      error: true,
    });
    conversation.view = "result";
    conversation.error = message;
    conversation.updatedAt = Date.now();
    persistConversations();
    if (activeConversationId === conversationId) {
      renderConversation(conversation);
      setView("result");
    }
  } finally {
    if (requestId === requestState.id) {
      requestState.controller = null;
      if (activeConversationId === conversationId) {
        endLoading();
      } else {
        elements.submit.disabled = activeConversation()?.view === "loading";
      }
    }
  }
}

elements.form.addEventListener("submit", (event) => { event.preventDefault(); submitQuestion(elements.question.value); });
document.addEventListener("click", (event) => {
  const deleteTarget = event.target.closest("[data-history-delete]");
  if (deleteTarget) { event.stopPropagation(); deleteHistory(deleteTarget.dataset.historyDelete); return; }
  const menuTarget = event.target.closest("[data-history-menu]");
  if (menuTarget) {
    event.stopPropagation();
    const menu = document.querySelector(`[data-history-popover="${menuTarget.dataset.historyMenu}"]`);
    const willOpen = menu.classList.contains("hidden");
    closeHistoryMenus();
    menu.classList.toggle("hidden", !willOpen);
    menuTarget.setAttribute("aria-expanded", String(willOpen));
    return;
  }
  const historyTarget = event.target.closest("[data-conversation-id]");
  if (historyTarget) {
    closeHistoryMenus();
    selectConversation(historyTarget.dataset.conversationId);
    return;
  }
  closeHistoryMenus();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeHistoryMenus();
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") { event.preventDefault(); elements.question.focus(); }
});
elements.question.addEventListener("input", () => {
  elements.question.style.height = "auto";
  elements.question.style.height = `${Math.min(elements.question.scrollHeight, 120)}px`;
  const conversation = activeConversation();
  if (conversation) { conversation.draft = elements.question.value; persistConversations(); }
});
$("#newConversation").addEventListener("click", () => {
  const conversation = createConversation(activeMode);
  selectConversation(conversation.id);
});
$("#copyAnswer").addEventListener("click", async () => { await navigator.clipboard.writeText(lastAnswer); $("#copyAnswer").textContent = "복사됨"; setTimeout(() => $("#copyAnswer").textContent = "답변 복사", 1200); });
async function loadHealth() {
  try {
    const response = await fetch("/api/health"); const data = await response.json();
    $("#systemStatus").textContent = data.status === "ready" ? "멀티 에이전트 준비됨" : "인덱스 준비 필요";
    $("#systemDetail").textContent = `MD ${data.documents}개 · 청크 ${data.chunks}개`;
  } catch (_) { $("#systemStatus").textContent = "서버 연결 확인 필요"; $("#systemDetail").textContent = "상태를 불러오지 못했습니다"; }
}

selectConversation(activeConversationId); loadHealth(); elements.question.focus();
