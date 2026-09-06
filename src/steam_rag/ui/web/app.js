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
  composer: $("#composer"), playMessages: $("#playMessages"), threadList: $("#threadList"),
  libraryList: $("#libraryList"), preferenceList: $("#preferenceList"),
  compareBasket: $("#compareBasket"), compareResult: $("#compareResult"),
  budget: $("#budgetCard"), budgetValue: $("#budgetValue"), budgetDetail: $("#budgetDetail"),
  attemptList: $("#attemptList"),
};

// 기획안 §4.4: 탐색 공간과 게임별 플레이 공간은 화면과 대화를 모두 분리한다.
const SPACE_PANELS = {compare: "#compareSpace", library: "#librarySpace", taste: "#tasteSpace", play: "#playSpace"};
const DISCOVERY_PANELS = ["#hero", "#landing", "#loading", "#result", "#error"];
const USER_ID = "local";
let activeSpace = "discovery";
let playContext = {appid: 0, name: "", threadId: "", playthrough: 1, messages: [], state: null, threads: []};
let compareBasket = readCompareBasket();

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
  if (activeSpace !== "discovery") return;
  elements.hero.classList.toggle("hidden", name !== "landing");
  elements.landing.classList.toggle("hidden", name !== "landing");
  elements.loading.classList.toggle("hidden", name !== "loading");
  elements.result.classList.toggle("hidden", name !== "result");
  elements.error.classList.toggle("hidden", name !== "error");
}

function setSpace(name) {
  activeSpace = name;
  $$("[data-space]").forEach((button) => button.classList.toggle("active", button.dataset.space === name));
  Object.values(SPACE_PANELS).forEach((selector) => $(selector).classList.add("hidden"));
  elements.composer.classList.toggle("hidden", name !== "discovery" && name !== "play");
  if (name === "discovery") {
    selectConversation(activeConversationId);
    return;
  }
  DISCOVERY_PANELS.forEach((selector) => $(selector).classList.add("hidden"));
  $(SPACE_PANELS[name]).classList.remove("hidden");
  elements.question.placeholder = name === "play"
    ? "이 게임에 대해 물어보세요"
    : "메시지를 입력하세요";
  if (name === "compare") renderCompareBasket();
  if (name === "library") loadLibrary();
  if (name === "taste") loadPreferences();
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
  renderBudget(data.budget || {});
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

const CONDITION_BADGES = {
  satisfied: {label: "조건 확인됨", className: "badge-ok"},
  unverified: {label: "일부 미확인", className: "badge-warn"},
  violated: {label: "조건 위반", className: "badge-bad"},
};

// 기획안 §4.2: 카드에 잘 맞는 점, 선택 전 확인, 정보 상태를 함께 보여준다.
function renderCandidateDetail(game) {
  const badge = CONDITION_BADGES[game.condition_status || "satisfied"] || CONDITION_BADGES.satisfied;
  const fit = (game.fit_reasons || []).slice(0, 4);
  const checks = (game.checks_before_choosing || []).slice(0, 4);
  const status = game.information_status || {};
  const checked = (status.checked_sources || []).slice(0, 3)
    .map((item) => `${item.source} · ${String(item.checked_at).slice(0, 10)}`);
  const unverified = (status.unverified_items || []).slice(0, 4);
  const blocks = [
    fit.length ? `<div class="card-block"><b>잘 맞는 점</b><ul>${fit.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : "",
    checks.length ? `<div class="card-block warn"><b>선택 전 확인</b><ul>${checks.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></div>` : "",
    (checked.length || unverified.length) ? `<div class="card-block muted"><b>정보 상태</b>
      ${checked.length ? `<small>확인한 출처: ${escapeHtml(checked.join(" / "))}</small>` : ""}
      ${unverified.length ? `<small>아직 확인하지 못함: ${escapeHtml(unverified.join(", "))}</small>` : ""}</div>` : "",
  ].filter(Boolean).join("");
  if (!blocks) return `<span class="condition-badge ${badge.className}">${badge.label}</span>`;
  return `<span class="condition-badge ${badge.className}">${badge.label}</span>${blocks}`;
}

function renderCandidateActions(game) {
  const payload = escapeHtml(JSON.stringify({appid: game.appid, name: game.name, image: game.image || ""}));
  return `<div class="card-actions">
    <button type="button" class="ghost-button" data-open-play="${payload}">공략 보기</button>
    <button type="button" class="ghost-button" data-save-game="${payload}">내 게임에 추가</button>
    <button type="button" class="ghost-button" data-compare-add="${payload}">비교에 추가</button>
  </div>`;
}

function renderTurnArtifacts(data = null) {
  if (!data) return "";
  const games = [...new Map((data.games || []).filter((game) => game?.appid).map((game) => [game.appid, game])).values()];
  const sources = (data.sources || []).slice(0, 6);
  const gameHtml = games.length ? `<div class="turn-artifacts"><b>확인된 게임</b><div class="turn-games">${games.map((game) => `
    <div class="turn-game">
      <a href="${escapeHtml(game.url || `https://store.steampowered.com/app/${game.appid}/`)}" target="_blank" rel="noreferrer">
        <span>${escapeHtml(game.name)}</span><small>${escapeHtml(game.status || "Steam에서 확인")}</small>
      </a>
      ${renderCandidateDetail(game)}
      ${renderCandidateActions(game)}
    </div>`).join("")}</div></div>` : "";
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
    workspace: data.workspace || "discovery",
    budget: data.budget || {},
    expert: data.expert || null,
  };
}

function renderBudget(budget = {}) {
  const searches = Number(budget.extra_searches || 0);
  const experts = Number(budget.expert_calls || 0);
  const visible = Boolean(budget.max_expert_calls);
  elements.budget.classList.toggle("hidden", !visible);
  if (!visible) return;
  elements.budgetValue.textContent = `${searches + experts}회`;
  elements.budgetDetail.textContent =
    `추가 검색 ${searches}/${budget.max_extra_searches} · 전문가 호출 ${experts}/${budget.max_expert_calls}`;
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

// ---------------------------------------------------------------------------
// 내 게임 · 내 취향 · 비교 · 게임별 플레이 공간 (기획안 §4.3, §4.4, §4.5)
// ---------------------------------------------------------------------------
async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {"Content-Type": "application/json"},
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || "요청에 실패했습니다.");
  return data;
}

function readCompareBasket() {
  try {
    const value = JSON.parse(localStorage.getItem("steamlens-compare-v1") || "[]");
    return Array.isArray(value) ? value.filter((item) => item?.appid).slice(0, 3) : [];
  } catch (_) { return []; }
}

function persistCompareBasket() {
  try { localStorage.setItem("steamlens-compare-v1", JSON.stringify(compareBasket)); } catch (_) {}
}

function addToCompare(game) {
  if (compareBasket.some((item) => Number(item.appid) === Number(game.appid))) return;
  compareBasket = [...compareBasket, game].slice(-3);
  persistCompareBasket();
  renderCompareBasket();
}

function renderCompareBasket() {
  elements.compareBasket.innerHTML = compareBasket.length
    ? compareBasket.map((game) => `<span class="basket-chip">${escapeHtml(game.name)}
        <button type="button" data-compare-remove="${escapeHtml(String(game.appid))}" aria-label="비교에서 제거">×</button></span>`).join("")
    : '<p class="muted-text">탐색 결과 카드에서 "비교에 추가"를 눌러 2~3개를 선택하세요.</p>';
}

async function runCompare() {
  if (compareBasket.length < 2) {
    elements.compareResult.innerHTML = '<p class="muted-text">비교하려면 게임을 2개 이상 선택해 주세요.</p>';
    return;
  }
  elements.compareResult.innerHTML = '<p class="muted-text">같은 기준으로 확인하는 중...</p>';
  try {
    const data = await api("/api/compare", {
      method: "POST",
      body: JSON.stringify({appids: compareBasket.map((game) => Number(game.appid))}),
    });
    const table = data.comparison || {};
    const games = table.games || [];
    const rows = (table.axes || []).map((axis) => `<tr class="${axis.differs ? "axis-differs" : ""}">
      <th>${escapeHtml(axis.label)}</th>
      ${axis.cells.map((cell) => `<td class="${cell.verified ? "" : "unverified"}">${escapeHtml(cell.display)}</td>`).join("")}
    </tr>`).join("");
    const missing = (data.missing_appids || []).length
      ? `<p class="muted-text">프로필을 찾지 못해 비교하지 못한 게임: ${escapeHtml(data.missing_appids.join(", "))}</p>`
      : "";
    elements.compareResult.innerHTML = `<div class="compare-table-wrap"><table class="compare-table">
      <thead><tr><th>비교 축</th>${games.map((game) => `<th>${escapeHtml(game.name)}</th>`).join("")}</tr></thead>
      <tbody>${rows}</tbody></table></div>${missing}
      <div class="answer-content">${renderMarkdown(data.answer || "")}</div>`;
  } catch (error) {
    elements.compareResult.innerHTML = `<p class="muted-text">${escapeHtml(error.message)}</p>`;
  }
}

async function loadLibrary() {
  elements.libraryList.innerHTML = '<p class="muted-text">저장한 게임을 불러오는 중...</p>';
  try {
    const data = await api(`/api/library?user_id=${encodeURIComponent(USER_ID)}`);
    const games = data.games || [];
    elements.libraryList.innerHTML = games.length ? games.map((game) => `
      <article class="library-card">
        <img src="${escapeHtml(game.header_image || `https://shared.cloudflare.steamstatic.com/store_item_assets/steam/apps/${game.appid}/header.jpg`)}" alt="${escapeHtml(game.name)}" loading="lazy">
        <div>
          <h4>${escapeHtml(game.name)}</h4>
          <small>진행: ${escapeHtml(game.progress || "미입력")} · 공략 대화 ${game.thread_count}개</small>
          <div class="card-actions">
            <button type="button" class="primary-button" data-open-play="${escapeHtml(JSON.stringify({appid: game.appid, name: game.name, image: game.header_image || ""}))}">플레이 공간 열기</button>
            <button type="button" class="ghost-button" data-remove-game="${escapeHtml(String(game.appid))}">삭제</button>
          </div>
        </div>
      </article>`).join("") : '<p class="muted-text">아직 저장한 게임이 없습니다. 탐색 결과에서 "내 게임에 추가"를 눌러 보세요.</p>';
  } catch (error) {
    elements.libraryList.innerHTML = `<p class="muted-text">${escapeHtml(error.message)}</p>`;
  }
}

async function loadPreferences() {
  try {
    const data = await api(`/api/preferences?user_id=${encodeURIComponent(USER_ID)}`);
    const rows = data.preferences || [];
    elements.preferenceList.innerHTML = rows.length ? rows.map((item) => `
      <div class="preference-row">
        <span class="preference-kind ${escapeHtml(item.kind)}">${escapeHtml(item.kind)}</span>
        <div><b>${escapeHtml(item.label || item.value)}</b><small>${escapeHtml(item.evidence || "근거 미기록")} · ${escapeHtml(item.scope)}</small></div>
        <button type="button" class="ghost-button" data-remove-preference="${escapeHtml(String(item.preference_id))}">삭제</button>
      </div>`).join("") : '<p class="muted-text">저장된 취향이 없습니다.</p>';
  } catch (error) {
    elements.preferenceList.innerHTML = `<p class="muted-text">${escapeHtml(error.message)}</p>`;
  }
}

async function openPlaySpace(game) {
  try {
    const handoff = await api("/api/play-space", {
      method: "POST",
      body: JSON.stringify({
        user_id: USER_ID,
        appid: Number(game.appid),
        name: String(game.name),
        header_image: game.image || "",
      }),
    });
    playContext = {
      appid: Number(game.appid),
      name: String(game.name),
      threadId: handoff.threads?.[0]?.thread_id || "",
      playthrough: handoff.threads?.[0]?.playthrough || 1,
      messages: [],
      state: handoff.game_state || null,
      threads: handoff.threads || [],
    };
    $("#playTitle").textContent = playContext.name;
    $("#playImage").src = game.image
      || `https://shared.cloudflare.steamstatic.com/store_item_assets/steam/apps/${game.appid}/header.jpg`;
    const topics = (handoff.support?.topic_labels || []).join(", ");
    $("#playScope").textContent = handoff.expert_verified
      ? `검증된 지원 범위: ${topics || "없음"}`
      : "검증된 지원 범위가 지정되지 않은 게임입니다. 확인된 자료로 답할 수 있는 범위까지만 안내합니다.";
    setSpace("play");
    await refreshPlaySpace();
  } catch (error) {
    elements.errorMessage.textContent = error.message;
    setSpace("discovery");
    setView("error");
  }
}

async function refreshPlaySpace() {
  if (!playContext.appid) return;
  const [threads, state] = await Promise.all([
    api(`/api/games/${playContext.appid}/threads?user_id=${encodeURIComponent(USER_ID)}`),
    api(`/api/games/${playContext.appid}/state?user_id=${encodeURIComponent(USER_ID)}&playthrough=${playContext.playthrough}`),
  ]);
  playContext.threads = threads.threads || [];
  playContext.state = state;
  if (!playContext.threadId && playContext.threads.length) playContext.threadId = playContext.threads[0].thread_id;
  renderThreadList();
  renderStateForm(state);
  await loadThreadMessages();
}

function renderThreadList() {
  elements.threadList.innerHTML = playContext.threads.map((thread) => `
    <button type="button" class="thread-row ${thread.thread_id === playContext.threadId ? "active" : ""}" data-thread-id="${escapeHtml(thread.thread_id)}">
      <b>${escapeHtml(thread.title || thread.topic)}</b><small>${escapeHtml(thread.updated_at.slice(0, 10))}</small>
    </button>`).join("") || '<p class="muted-text">주제를 추가해 보세요.</p>';
}

function renderStateForm(state = {}) {
  $("#stateProgress").value = state.progress || "";
  $("#stateBuild").value = state.character_build || "";
  $("#stateEquipment").value = (state.equipment || []).join(", ");
  $("#stateSpoiler").value = state.spoiler_level || "no_spoiler";
  const attempts = state.attempts || [];
  elements.attemptList.innerHTML = attempts.length
    ? `<div class="section-heading"><h3>이전에 시도한 방법</h3></div>${attempts.map((item) => `
        <div class="attempt-row"><b>${escapeHtml(item.action)}</b><small>${escapeHtml(item.outcome || "결과 미기록")}</small></div>`).join("")}`
    : "";
}

async function loadThreadMessages() {
  if (!playContext.threadId) { elements.playMessages.innerHTML = ""; return; }
  const data = await api(`/api/games/threads/${encodeURIComponent(playContext.threadId)}/messages?user_id=${encodeURIComponent(USER_ID)}`);
  playContext.messages = data.messages || [];
  renderPlayMessages();
}

function renderPlayMessages(loading = false) {
  const rows = playContext.messages.map((message) => `
    <article class="chat-message ${message.role}">
      <div class="chat-avatar">${message.role === "user" ? "나" : "S"}</div>
      <div class="chat-body"><b>${message.role === "user" ? "나" : `${escapeHtml(playContext.name)} 전문가`}</b>
        <div>${message.role === "assistant" ? renderMarkdown(message.content) : `<p>${escapeHtml(message.content)}</p>`}</div>
      </div>
    </article>`).join("");
  const typing = loading ? `
    <article class="chat-message assistant typing-message">
      <div class="chat-avatar">S</div>
      <div class="chat-body"><b>${escapeHtml(playContext.name)} 전문가</b><div class="typing-dots"><i></i><i></i><i></i></div></div>
    </article>` : "";
  elements.playMessages.innerHTML = rows + typing
    || '<p class="muted-text">이 게임에 대해 물어보세요. 진행도와 스포일러 설정 안에서만 답합니다.</p>';
}

async function submitPlayQuestion(question) {
  if (!playContext.appid) return;
  playContext.messages.push({role: "user", content: question});
  elements.question.value = "";
  elements.submit.disabled = true;
  renderPlayMessages(true);
  try {
    const response = await postChat({
      question,
      top_k: 6,
      workspace: "play",
      user_id: USER_ID,
      game_id: playContext.appid,
      thread_id: playContext.threadId,
      playthrough: playContext.playthrough,
      request_id: makeRequestId(),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "서버 요청에 실패했습니다.");
    playContext.threadId = data.thread?.thread_id || playContext.threadId;
    await refreshPlaySpace();
    renderBudget(data.budget || {});
  } catch (error) {
    playContext.messages.push({role: "assistant", content: `요청을 처리하지 못했습니다.\n\n${error.message}`});
    renderPlayMessages();
  } finally {
    elements.submit.disabled = false;
  }
}

async function submitQuestion(question) {
  question = String(question || "").trim(); if (!question) return;
  if (activeSpace === "play") { await submitPlayQuestion(question); return; }
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
      {
        question, history, context_games, conversation_state, top_k: 6,
        request_id: clientRequestId,
        workspace: "discovery",
        user_id: USER_ID,
        session_id: conversationId.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64),
      },
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
  const spaceTarget = event.target.closest("[data-space]");
  if (spaceTarget) { setSpace(spaceTarget.dataset.space); return; }
  const playTarget = event.target.closest("[data-open-play]");
  if (playTarget) { openPlaySpace(JSON.parse(playTarget.dataset.openPlay)); return; }
  const saveTarget = event.target.closest("[data-save-game]");
  if (saveTarget) {
    const game = JSON.parse(saveTarget.dataset.saveGame);
    api("/api/library", {
      method: "POST",
      body: JSON.stringify({user_id: USER_ID, appid: Number(game.appid), name: String(game.name), header_image: game.image || ""}),
    }).then(() => { saveTarget.textContent = "내 게임에 추가됨"; }).catch(() => { saveTarget.textContent = "추가 실패"; });
    return;
  }
  const compareAdd = event.target.closest("[data-compare-add]");
  if (compareAdd) { addToCompare(JSON.parse(compareAdd.dataset.compareAdd)); compareAdd.textContent = "비교에 추가됨"; return; }
  const compareRemove = event.target.closest("[data-compare-remove]");
  if (compareRemove) {
    compareBasket = compareBasket.filter((item) => String(item.appid) !== compareRemove.dataset.compareRemove);
    persistCompareBasket(); renderCompareBasket(); return;
  }
  const removeGame = event.target.closest("[data-remove-game]");
  if (removeGame) {
    api(`/api/library/${removeGame.dataset.removeGame}?user_id=${encodeURIComponent(USER_ID)}`, {method: "DELETE"})
      .then(loadLibrary).catch(() => loadLibrary());
    return;
  }
  const removePreference = event.target.closest("[data-remove-preference]");
  if (removePreference) {
    api(`/api/preferences/${removePreference.dataset.removePreference}?user_id=${encodeURIComponent(USER_ID)}`, {method: "DELETE"})
      .then(loadPreferences).catch(() => loadPreferences());
    return;
  }
  const threadTarget = event.target.closest("[data-thread-id]");
  if (threadTarget) {
    playContext.threadId = threadTarget.dataset.threadId;
    renderThreadList();
    loadThreadMessages();
    return;
  }
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
  setSpace("discovery");
  selectConversation(conversation.id);
});
$("#runCompare").addEventListener("click", runCompare);
$("#backToLibrary").addEventListener("click", () => setSpace("library"));
$("#newThread").addEventListener("click", async () => {
  if (!playContext.appid) return;
  const title = window.prompt("새 공략 주제 이름", "새 공략 대화");
  if (!title) return;
  const thread = await api("/api/games/threads", {
    method: "POST",
    body: JSON.stringify({
      user_id: USER_ID,
      appid: playContext.appid,
      topic: "general",
      title: title.slice(0, 120),
      playthrough: playContext.playthrough,
    }),
  });
  playContext.threadId = thread.thread_id;
  await refreshPlaySpace();
});
$("#stateForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  await api(`/api/games/${playContext.appid}/state`, {
    method: "PUT",
    body: JSON.stringify({
      user_id: USER_ID,
      appid: playContext.appid,
      playthrough: playContext.playthrough,
      progress: $("#stateProgress").value,
      character_build: $("#stateBuild").value,
      equipment: $("#stateEquipment").value.split(",").map((item) => item.trim()).filter(Boolean),
      spoiler_level: $("#stateSpoiler").value,
    }),
  });
  await refreshPlaySpace();
});
$("#preferenceForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const value = $("#preferenceValue").value.trim();
  if (!value) return;
  await api("/api/preferences", {
    method: "POST",
    body: JSON.stringify({
      user_id: USER_ID,
      kind: $("#preferenceKind").value,
      value,
      label: value,
      evidence: $("#preferenceEvidence").value.trim(),
    }),
  });
  $("#preferenceValue").value = ""; $("#preferenceEvidence").value = "";
  await loadPreferences();
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
