const state = {
  runs: [],
  selectedRun: "",
  records: [],
  filtered: [],
};

const el = {
  runSelect: document.getElementById("runSelect"),
  refreshBtn: document.getElementById("refreshBtn"),
  reloadBtn: document.getElementById("reloadBtn"),
  loaderMessage: document.getElementById("loaderMessage"),
  outputBadges: document.getElementById("outputBadges"),
  emptyState: document.getElementById("emptyState"),
  emptyStateText: document.getElementById("emptyStateText"),
  dashboardContent: document.getElementById("dashboardContent"),
  summaryCards: document.getElementById("summaryCards"),
  summaryTableBody: document.getElementById("summaryTableBody"),
  runMeta: document.getElementById("runMeta"),
  searchInput: document.getElementById("searchInput"),
  paperFilter: document.getElementById("paperFilter"),
  questionFilter: document.getElementById("questionFilter"),
  typeFilter: document.getElementById("typeFilter"),
  criticFilter: document.getElementById("criticFilter"),
  questionList: document.getElementById("questionList"),
  questionCount: document.getElementById("questionCount"),
};

const RUN_NAME_PART = "le" + "na";
const BUNDLED_RUNS = [
  {
    name: "p4_run4_answers_20260519_175128_papers33_p01-33_q1-49_nq1617.jsonl",
    url: new URL("./data/p4_run4_answers_20260519_175128_papers33_p01-33_q1-49_nq1617.jsonl", window.location.href).href,
    source: "bundled",
  },
];
const OUTPUTS_DIR_URL = new URL(`../outputs_${RUN_NAME_PART}_p4/`, window.location.href);

const TYPE_BADGE_CLASS = {
  NUMERIC: "bg-blue-100 text-blue-800 border-blue-200",
  LIST: "bg-purple-100 text-purple-800 border-purple-200",
  FREE_TEXT: "bg-slate-100 text-slate-700 border-slate-200",
  STRING: "bg-gray-100 text-gray-700 border-gray-200",
  CATEGORICAL: "bg-amber-100 text-amber-800 border-amber-200",
};

const CRITIC_BADGE_CLASS = {
  applied: "bg-slate-100 text-slate-700 border-slate-200",
  bypassed: "bg-sky-100 text-sky-800 border-sky-200",
  abstained: "bg-rose-100 text-rose-800 border-rose-200",
};

function text(value) {
  return String(value ?? "");
}

function esc(value) {
  return text(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function toNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function formatNumber(value, digits = 0) {
  return toNumber(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatUsd(value) {
  return toNumber(value).toLocaleString(undefined, {
    minimumFractionDigits: 6,
    maximumFractionDigits: 8,
  });
}

function setLoaderMessage(message, kind = "info") {
  const styleByKind = {
    info: "border-slate-200 bg-slate-50 text-slate-600",
    success: "border-emerald-200 bg-emerald-50 text-emerald-700",
    error: "border-red-200 bg-red-50 text-red-700",
  };
  el.loaderMessage.className = `mt-4 rounded-lg border px-3 py-2 text-sm ${styleByKind[kind] || styleByKind.info}`;
  el.loaderMessage.textContent = message;
}

function normalizeRecord(record, index) {
  const generation = record.generation && typeof record.generation === "object" ? record.generation : {};
  const retrieval = record.retrieval && typeof record.retrieval === "object" ? record.retrieval : {};
  const logs = record.logs && typeof record.logs === "object" ? record.logs : {};
  const run4Key = `${RUN_NAME_PART}_run4`;
  const run4 = record[run4Key] && typeof record[run4Key] === "object" ? record[run4Key] : {};
  const quotes = Array.isArray(generation.evidence_quotes)
    ? generation.evidence_quotes.map((quote) => text(quote).trim()).filter(Boolean)
    : [];
  const topChunks = Array.isArray(retrieval.top_chunks)
    ? retrieval.top_chunks.filter((chunk) => chunk && typeof chunk === "object")
    : [];

  return {
    index,
    run_id: text(record.run_id),
    pipeline_name: text(record.pipeline_name),
    paper_id: toNumber(record.paper_id, 0),
    paper_label: text(run4.paper_label || `Paper ${record.paper_id ?? ""}`),
    source_file_name: text(run4.source_file_name),
    question_id: toNumber(record.question_id, 0),
    question_type: text(record.question_type || "FREE_TEXT").toUpperCase(),
    question: text(record.question),
    reference_answer: text(record.reference_answer),
    model_answer: text(generation.model_answer),
    reasoning: text(generation.reasoning),
    evidence_quotes: quotes,
    critique_logic: text(generation.critique_logic),
    is_abstained: Boolean(generation.is_abstained),
    retrieved_chunks: topChunks.map((chunk, chunkIndex) => ({
      rank: toNumber(chunk.rank, chunkIndex + 1),
      score: toNumber(chunk.score, 0),
      chunk_id: text(chunk.chunk_id),
      para_ids: Array.isArray(chunk.para_ids) ? chunk.para_ids.map(text) : [],
      text: text(chunk.text),
    })),
    latency_ms: toNumber(logs.latency_ms, 0),
    tokens_input: toNumber(logs.tokens_input, 0),
    tokens_output: toNumber(logs.tokens_output, 0),
    estimated_cost_usd: toNumber(logs.estimated_cost_usd, 0),
  };
}

function criticMode(record) {
  const logic = record.critique_logic.toLowerCase();
  if (record.is_abstained) {
    return "abstained";
  }
  if (logic.includes("bypassed critic")) {
    return "bypassed";
  }
  return "applied";
}

function haystack(record) {
  return [
    record.paper_label,
    record.source_file_name,
    record.question,
    record.reference_answer,
    record.model_answer,
    record.reasoning,
    record.critique_logic,
    ...record.evidence_quotes,
  ].join(" ").toLowerCase();
}

async function fetchText(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.text();
}

function parseDirectoryListing(html) {
  const doc = new DOMParser().parseFromString(html, "text/html");
  return Array.from(doc.querySelectorAll("a"))
    .map((anchor) => {
      const href = anchor.getAttribute("href") || "";
      const url = new URL(href, OUTPUTS_DIR_URL);
      const name = decodeURIComponent(url.pathname.split("/").filter(Boolean).pop() || "");
      return { name, url: url.href };
    })
    .filter((run) => run.name.endsWith(".jsonl"))
    .sort((a, b) => b.name.localeCompare(a.name));
}

function parseJsonl(rawText, fileName) {
  const records = [];
  rawText.split(/\r?\n/).forEach((line, index) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    try {
      const payload = JSON.parse(trimmed);
      if (payload && typeof payload === "object" && !Array.isArray(payload)) {
        records.push(payload);
      }
    } catch (error) {
      throw new Error(`${fileName}: invalid JSONL line ${index + 1}: ${error.message}`);
    }
  });
  return records;
}

async function loadRuns() {
  setLoaderMessage("Loading bundled P4 output...");
  let discoveredRuns = [];
  try {
    const html = await fetchText(OUTPUTS_DIR_URL.href);
    discoveredRuns = parseDirectoryListing(html).map((run) => ({ ...run, source: "output-folder" }));
  } catch (error) {
    console.warn("Could not read local output folder; using bundled web UI data only.", error);
  }

  const bundledNames = new Set(BUNDLED_RUNS.map((run) => run.name));
  state.runs = [
    ...BUNDLED_RUNS,
    ...discoveredRuns.filter((run) => !bundledNames.has(run.name)),
  ];
  renderRunSelect();
  renderOutputBadges();

  if (!state.runs.length) {
    state.selectedRun = "";
    state.records = [];
    state.filtered = [];
    renderAll();
    setLoaderMessage("No JSONL outputs found in the run 4 output folder.", "error");
    return;
  }

  const currentStillExists = state.runs.some((run) => run.name === state.selectedRun);
  state.selectedRun = currentStillExists ? state.selectedRun : BUNDLED_RUNS[0]?.name || state.runs[0].name;
  el.runSelect.value = state.selectedRun;
  await loadSelectedRun({ fallbackNonEmpty: true });
}

async function loadSelectedRun(options = {}) {
  if (!state.selectedRun) return;
  let run = state.runs.find((item) => item.name === state.selectedRun);
  if (!run) return;

  let rawRecords = [];
  for (let attemptIndex = state.runs.indexOf(run); attemptIndex < state.runs.length; attemptIndex += 1) {
    run = state.runs[attemptIndex];
    setLoaderMessage(`Loading ${run.name}...`);
    const jsonlText = await fetchText(run.url);
    rawRecords = parseJsonl(jsonlText, run.name);
    run.record_count = rawRecords.length;
    run.size_bytes = new Blob([jsonlText]).size;
    state.selectedRun = run.name;
    el.runSelect.value = run.name;
    if (!options.fallbackNonEmpty || rawRecords.length > 0) {
      break;
    }
  }

  state.records = rawRecords.map(normalizeRecord);
  renderRunSelect();
  renderOutputBadges();
  el.runSelect.value = state.selectedRun;
  renderFilterOptions();
  applyFilters();
  setLoaderMessage(`Loaded ${state.records.length} records from ${state.selectedRun}.`, "success");
}

function renderRunSelect() {
  el.runSelect.innerHTML = state.runs.length
    ? state.runs
        .map((run) => {
          const count = run.record_count;
          const sourceLabel = run.source === "bundled" ? "bundled" : "output";
          const labelSuffix = count === undefined ? "JSONL" : count === 0 ? "empty" : `${formatNumber(count)} records`;
          return `<option value="${esc(run.name)}">${esc(run.name)} (${esc(sourceLabel)}, ${esc(labelSuffix)})</option>`;
        })
        .join("")
    : '<option value="">No outputs found</option>';
}

function renderOutputBadges() {
  if (!state.runs.length) {
    el.outputBadges.innerHTML = '<span class="rounded-md border border-slate-200 bg-slate-100 px-2 py-1 text-xs text-slate-500">No outputs detected</span>';
    return;
  }
  el.outputBadges.innerHTML = state.runs
    .map((run) => {
      const isSelected = run.name === state.selectedRun;
      const count = run.record_count === undefined ? "JSONL" : run.record_count === 0 ? "empty" : `${formatNumber(run.record_count)} records`;
      const sourceLabel = run.source === "bundled" ? "Bundled" : "Output";
      const className = isSelected
        ? "border-sky-300 bg-sky-50 text-sky-800"
        : "border-slate-200 bg-slate-100 text-slate-600";
      return `<span class="rounded-full border px-3 py-1 text-xs font-semibold ${className}">${esc(sourceLabel)} · ${esc(count)}</span>`;
    })
    .join("");
}

function renderFilterOptions() {
  const papers = [...new Set(state.records.map((record) => record.paper_id))]
    .filter((paperId) => paperId > 0)
    .sort((a, b) => a - b);
  const questionIds = [...new Set(state.records.map((record) => record.question_id))]
    .filter((qid) => qid > 0)
    .sort((a, b) => a - b);
  const types = [...new Set(state.records.map((record) => record.question_type))]
    .filter(Boolean)
    .sort();

  el.paperFilter.innerHTML =
    '<option value="">All papers</option>' +
    papers.map((paperId) => `<option value="${esc(paperId)}">Paper ${esc(paperId)}</option>`).join("");
  el.questionFilter.innerHTML =
    '<option value="">All questions</option>' +
    questionIds.map((qid) => `<option value="${esc(qid)}">Q${esc(qid)}</option>`).join("");
  el.typeFilter.innerHTML =
    '<option value="">All types</option>' +
    types.map((type) => `<option value="${esc(type)}">${esc(type)}</option>`).join("");
}

function applyFilters() {
  const query = el.searchInput.value.trim().toLowerCase();
  const paper = el.paperFilter.value;
  const qid = el.questionFilter.value;
  const type = el.typeFilter.value;
  const critic = el.criticFilter.value;

  state.filtered = state.records.filter((record) => {
    if (paper && String(record.paper_id) !== paper) return false;
    if (qid && String(record.question_id) !== qid) return false;
    if (type && record.question_type !== type) return false;
    if (critic && criticMode(record) !== critic) return false;
    if (query && !haystack(record).includes(query)) return false;
    return true;
  });
  renderAll();
}

function selectedRunSummary() {
  const papers = new Set(state.records.map((record) => record.paper_id)).size;
  const answers = state.records.length;
  const shown = state.filtered.length;
  const abstained = state.records.filter((record) => record.is_abstained).length;
  const avgLatency = answers
    ? state.records.reduce((sum, record) => sum + record.latency_ms, 0) / answers
    : 0;
  const totalCost = state.records.reduce((sum, record) => sum + record.estimated_cost_usd, 0);
  return { papers, answers, shown, abstained, avgLatency, totalCost };
}

function renderAll() {
  const hasRecords = state.records.length > 0;
  el.emptyState.classList.toggle("hidden", hasRecords);
  el.dashboardContent.classList.toggle("hidden", !hasRecords);
  el.emptyStateText.textContent = state.runs.length
    ? "Selected output has no records."
    : "Run the pipeline or refresh the output list.";

  renderMeta();
  renderSummaryCards();
  renderSummaryTable();
  renderQuestionCards();
}

function renderMeta() {
  const run = state.runs.find((item) => item.name === state.selectedRun);
  if (!run) {
    el.runMeta.textContent = "";
    return;
  }
  const count = run.record_count === undefined ? "not loaded" : `${formatNumber(run.record_count)} records`;
  const size = run.size_bytes === undefined ? "" : ` | ${formatNumber(run.size_bytes)} bytes`;
  el.runMeta.textContent = `${count}${size}`;
}

function renderSummaryCards() {
  const summary = selectedRunSummary();
  const cards = [
    ["Papers", `${formatNumber(summary.papers)} papers`, "metric-papers"],
    ["Questions", `${formatNumber(summary.answers)} questions`, "metric-answers"],
    ["Shown", `${formatNumber(summary.shown)} shown`, "metric-shown"],
    ["Abstained", summary.abstained, "metric-abstained"],
    ["Avg Latency", `${formatNumber(summary.avgLatency, 1)} ms`, "metric-latency"],
    ["Total Cost", `$${formatUsd(summary.totalCost)}`, "metric-cost"],
  ];
  el.summaryCards.innerHTML = cards
    .map(
      ([label, value, metricClass]) => `
        <div class="metric-card ${metricClass} rounded-xl border border-slate-200 bg-slate-50 p-4 pl-5">
          <p class="text-xs font-bold uppercase tracking-wider text-slate-500">${esc(label)}</p>
          <p class="mt-2 text-2xl font-bold text-slate-900">${esc(value)}</p>
        </div>
      `,
    )
    .join("");
}

function renderSummaryTable() {
  const summary = selectedRunSummary();
  el.summaryTableBody.innerHTML = `
    <tr>
      <td class="px-3 py-2 text-left font-medium text-slate-800">${esc(state.selectedRun || "-")}</td>
      <td class="px-3 py-2 text-right">${esc(summary.papers)}</td>
      <td class="px-3 py-2 text-right">${esc(summary.answers)}</td>
      <td class="px-3 py-2 text-right">${esc(summary.abstained)}</td>
      <td class="px-3 py-2 text-right">${esc(formatNumber(summary.avgLatency, 1))} ms</td>
      <td class="px-3 py-2 text-right">$${esc(formatUsd(summary.totalCost))}</td>
    </tr>
  `;
}

function typeBadgeClass(type) {
  return TYPE_BADGE_CLASS[type] || TYPE_BADGE_CLASS.FREE_TEXT;
}

function renderQuestionCards() {
  el.questionCount.textContent = `${state.filtered.length} question${state.filtered.length === 1 ? "" : "s"}`;
  if (!state.filtered.length) {
    el.questionList.innerHTML = '<p class="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">No questions match the current filters.</p>';
    return;
  }

  el.questionList.innerHTML = state.filtered
    .map((record) => {
      const mode = criticMode(record);
      const typeClass = typeBadgeClass(record.question_type);
      const criticClass = CRITIC_BADGE_CLASS[mode] || CRITIC_BADGE_CLASS.applied;
      return `
        <details class="question-details" ${record.index === state.filtered[0].index ? "open" : ""}>
          <summary class="cursor-pointer px-4 py-3">
            <div class="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
              <div class="min-w-0">
                <div class="mb-2 flex flex-wrap gap-2">
                  <span class="rounded-full border px-2 py-1 text-xs font-semibold ${typeClass}">${esc(record.question_type)}</span>
                  <span class="rounded-full border px-2 py-1 text-xs font-semibold ${criticClass}">${esc(mode.toUpperCase())}</span>
                  <span class="rounded-full border border-slate-200 bg-white px-2 py-1 text-xs font-semibold text-slate-600">Paper ${esc(record.paper_id)}</span>
                  <span class="rounded-full border border-slate-200 bg-white px-2 py-1 text-xs font-semibold text-slate-600">Q${esc(record.question_id)}</span>
                </div>
                <h3 class="text-base font-semibold text-slate-900">${esc(record.question)}</h3>
              </div>
              <span class="summary-chevron text-slate-400">›</span>
            </div>
          </summary>
          <div class="space-y-4 border-t border-slate-200 px-4 py-4">
            ${renderAnswerGrid(record)}
            ${renderEvidenceQuotes(record)}
            ${renderRetrievedChunks(record)}
          </div>
        </details>
      `;
    })
    .join("");
}

function panel(title, body, accentClass = "") {
  return `
    <div class="rounded-xl border border-slate-200 bg-slate-50 p-4">
      <h4 class="mb-2 text-sm font-semibold text-slate-700">${esc(title)}</h4>
      <div class="answer-text rounded-lg border border-slate-200 bg-white p-3 text-sm leading-6 text-slate-800 ${accentClass}">${body}</div>
    </div>
  `;
}

function renderAnswerGrid(record) {
  return `
    <div class="grid grid-cols-1 gap-4 lg:grid-cols-2">
      ${panel("Final Answer", esc(record.model_answer || "No answer"), "answer-final")}
      ${panel("Gold Answer", esc(record.reference_answer || "No gold answer in workbook."), "answer-gold")}
      ${panel("Reasoning", esc(record.reasoning || "No reasoning returned."), "answer-reasoning")}
      ${panel("Critic Logic", esc(record.critique_logic || "No critic logic returned."), "answer-critic")}
    </div>
  `;
}

function renderEvidenceQuotes(record) {
  const content = record.evidence_quotes.length
    ? record.evidence_quotes
        .map((quote) => `<blockquote class="mb-2 rounded-md border-l-4 border-sky-300 bg-sky-50 px-3 py-2 text-sm leading-6 text-slate-800">${esc(quote)}</blockquote>`)
        .join("")
    : '<p class="text-sm text-slate-600">No evidence quotes returned.</p>';
  return `
    <details class="evidence-details rounded-xl border border-slate-200 bg-white" open>
      <summary class="cursor-pointer px-4 py-3 text-sm font-semibold text-slate-800">Evidence Quotes</summary>
      <div class="border-t border-slate-200 px-4 py-3">${content}</div>
    </details>
  `;
}

function renderRetrievedChunks(record) {
  const content = record.retrieved_chunks.length
    ? record.retrieved_chunks
        .map(
          (chunk) => `
            <div class="mb-3 rounded-md border border-sky-200 bg-sky-50 p-3">
              <p class="mb-2 text-xs font-semibold text-sky-800">Rank ${esc(chunk.rank)} | score ${esc(formatNumber(chunk.score, 4))}${chunk.chunk_id ? ` | ${esc(chunk.chunk_id)}` : ""}</p>
              <pre class="evidence-pre text-xs text-slate-800">${esc(chunk.text || "(empty chunk text)")}</pre>
            </div>
          `,
        )
        .join("")
    : '<p class="text-sm text-slate-600">No retrieved chunks recorded.</p>';
  return `
    <details class="evidence-details rounded-xl border border-slate-200 bg-white">
      <summary class="cursor-pointer px-4 py-3 text-sm font-semibold text-slate-800">Retrieved Chunks</summary>
      <div class="border-t border-slate-200 px-4 py-3">${content}</div>
    </details>
  `;
}

el.runSelect.addEventListener("change", async () => {
  state.selectedRun = el.runSelect.value;
  await loadSelectedRun();
});
el.refreshBtn.addEventListener("click", loadRuns);
el.reloadBtn.addEventListener("click", loadSelectedRun);

for (const input of [el.searchInput, el.paperFilter, el.questionFilter, el.typeFilter, el.criticFilter]) {
  input.addEventListener("input", applyFilters);
  input.addEventListener("change", applyFilters);
}

loadRuns().catch((error) => {
  console.error(error);
  setLoaderMessage(`Failed to load P4 outputs: ${error.message}`, "error");
});
