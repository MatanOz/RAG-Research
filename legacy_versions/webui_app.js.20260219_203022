const state = {
  records: [],
  filtered: [],
  selectedIndex: -1,
  sourceLabel: "",
};

const el = {
  fileInput: document.getElementById("fileInput"),
  runPathInput: document.getElementById("runPathInput"),
  loadPathBtn: document.getElementById("loadPathBtn"),
  loadLatestBtn: document.getElementById("loadLatestBtn"),
  paperFilter: document.getElementById("paperFilter"),
  typeFilter: document.getElementById("typeFilter"),
  answerableFilter: document.getElementById("answerableFilter"),
  searchInput: document.getElementById("searchInput"),
  status: document.getElementById("status"),
  runMeta: document.getElementById("runMeta"),
  tableBody: document.getElementById("tableBody"),
  tableCount: document.getElementById("tableCount"),
  detailContent: document.getElementById("detailContent"),
  detailTemplate: document.getElementById("detailTemplate"),
  mRecords: document.getElementById("mRecords"),
  mPapers: document.getElementById("mPapers"),
  mLatency: document.getElementById("mLatency"),
  mCost: document.getElementById("mCost"),
  mTop1: document.getElementById("mTop1"),
  mTokens: document.getElementById("mTokens"),
};

function setStatus(message, isError = false) {
  el.status.textContent = message;
  el.status.style.color = isError ? "#ef4444" : "#9fb0c6";
}

function parseJsonl(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, i) => {
      try {
        return JSON.parse(line);
      } catch (err) {
        throw new Error(`Invalid JSON at line ${i + 1}: ${err.message}`);
      }
    });
}

function formatNum(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function short(text, max = 90) {
  if (!text) return "";
  return text.length <= max ? text : `${text.slice(0, max - 1)}...`;
}

function refreshFilterOptions() {
  const papers = [...new Set(state.records.map((r) => r.paper_id))].sort((a, b) => a - b);
  const types = [...new Set(state.records.map((r) => r.question_type))].sort();

  el.paperFilter.innerHTML = `<option value="all">All</option>${papers
    .map((id) => `<option value="${id}">Paper ${id}</option>`)
    .join("")}`;

  el.typeFilter.innerHTML = `<option value="all">All</option>${types
    .map((t) => `<option value="${t}">${t}</option>`)
    .join("")}`;
}

function applyFilters() {
  const paper = el.paperFilter.value;
  const type = el.typeFilter.value;
  const answerable = el.answerableFilter.value;
  const q = el.searchInput.value.trim().toLowerCase();

  state.filtered = state.records.filter((r) => {
    if (paper !== "all" && String(r.paper_id) !== paper) return false;
    if (type !== "all" && r.question_type !== type) return false;
    if (answerable !== "all" && String(Boolean(r.is_answerable)) !== answerable) return false;

    if (!q) return true;

    const textPool = [
      r.question,
      r.reference_answer,
      r.generation?.model_answer,
      ...(r.retrieval?.retrieved_para_ids || []),
      ...(r.retrieval?.top_chunks || []).map((c) => c.chunk_id),
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();

    return textPool.includes(q);
  });

  state.selectedIndex = state.filtered.length ? 0 : -1;
  renderMetrics();
  renderTable();
  renderDetail();
}

function renderMetrics() {
  const data = state.filtered;
  const papers = new Set(data.map((r) => r.paper_id));
  const avgLatency = data.length
    ? data.reduce((acc, r) => acc + (r.logs?.latency_ms || 0), 0) / data.length
    : null;
  const totalCost = data.reduce((acc, r) => acc + (r.logs?.estimated_cost_usd || 0), 0);
  const avgTop1 =
    data.length > 0
      ? data.reduce((acc, r) => acc + ((r.retrieval?.retrieval_scores || [0])[0] || 0), 0) / data.length
      : null;
  const totalTokens = data.reduce((acc, r) => acc + (r.logs?.total_tokens || 0), 0);

  el.mRecords.textContent = String(data.length || 0);
  el.mPapers.textContent = String(papers.size || 0);
  el.mLatency.textContent = avgLatency === null ? "-" : formatNum(avgLatency, 1);
  el.mCost.textContent = `$${formatNum(totalCost, 6)}`;
  el.mTop1.textContent = avgTop1 === null ? "-" : formatNum(avgTop1, 4);
  el.mTokens.textContent = totalTokens ? totalTokens.toLocaleString() : "0";
}

function renderTable() {
  el.tableBody.innerHTML = "";
  state.filtered.forEach((r, idx) => {
    const tr = document.createElement("tr");
    if (idx === state.selectedIndex) tr.classList.add("active");

    const top1 = (r.retrieval?.retrieval_scores || [])[0];
    tr.innerHTML = `
      <td>${r.paper_id}</td>
      <td>${r.question_id}</td>
      <td>${r.question_type || "-"}</td>
      <td>${short(r.question, 120)}</td>
      <td>${top1 !== undefined ? formatNum(top1, 4) : "-"}</td>
      <td>${r.logs?.latency_ms ? formatNum(r.logs.latency_ms, 1) : "-"}</td>
      <td>${r.logs?.estimated_cost_usd ? `$${formatNum(r.logs.estimated_cost_usd, 6)}` : "$0.000000"}</td>
    `;

    tr.addEventListener("click", () => {
      state.selectedIndex = idx;
      renderTable();
      renderDetail();
    });

    el.tableBody.appendChild(tr);
  });

  el.tableCount.textContent = `${state.filtered.length} row${state.filtered.length === 1 ? "" : "s"}`;
}

function scoreBar(rank, score) {
  const wrapper = document.createElement("div");
  wrapper.className = "score-row";

  const label = document.createElement("div");
  label.textContent = `Rank ${rank}`;
  label.className = "mono";

  const track = document.createElement("div");
  track.className = "track";
  const fill = document.createElement("div");
  fill.className = "fill";
  const clamped = Math.max(0, Math.min(1, Number(score)));
  fill.style.width = `${clamped * 100}%`;
  track.appendChild(fill);

  const value = document.createElement("div");
  value.className = "mono";
  value.textContent = formatNum(score, 4);

  wrapper.appendChild(label);
  wrapper.appendChild(track);
  wrapper.appendChild(value);
  return wrapper;
}

function chunkCard(chunk) {
  const card = document.createElement("article");
  card.className = "chunk-card";

  const meta = document.createElement("div");
  meta.className = "chunk-meta";

  const tags = [
    `chunk: ${chunk.chunk_id || "-"}`,
    `rank: ${chunk.rank}`,
    `score: ${formatNum(chunk.score, 4)}`,
    ...((chunk.para_ids || []).slice(1, 4) || []),
  ];

  tags.forEach((text) => {
    const t = document.createElement("span");
    t.className = "tag";
    t.textContent = text;
    meta.appendChild(t);
  });

  const text = document.createElement("div");
  text.className = "chunk-text";
  text.textContent = chunk.text || "";

  card.appendChild(meta);
  card.appendChild(text);
  return card;
}

function renderDetail() {
  el.detailContent.innerHTML = "";
  if (state.selectedIndex < 0 || !state.filtered[state.selectedIndex]) {
    el.detailContent.textContent = "Select a row to inspect question details.";
    el.detailContent.className = "detail-empty";
    return;
  }

  el.detailContent.className = "";
  const r = state.filtered[state.selectedIndex];
  const fragment = el.detailTemplate.content.cloneNode(true);

  fragment.querySelector('[data-field="qid"]').textContent = `paper ${r.paper_id} | question ${r.question_id}`;
  fragment.querySelector('[data-field="question"]').textContent = r.question || "";
  fragment.querySelector('[data-field="reference"]').textContent = r.reference_answer || "";
  fragment.querySelector('[data-field="answer"]').textContent = r.generation?.model_answer || "";

  const scoresEl = fragment.querySelector('[data-field="scores"]');
  const chunks = r.retrieval?.top_chunks || [];
  if (!chunks.length) {
    const empty = document.createElement("div");
    empty.className = "mono";
    empty.textContent = "No retrieval chunks available.";
    scoresEl.appendChild(empty);
  } else {
    chunks.forEach((c) => scoresEl.appendChild(scoreBar(c.rank, c.score)));
  }

  const chunkList = fragment.querySelector('[data-field="chunks"]');
  chunks.forEach((chunk) => chunkList.appendChild(chunkCard(chunk)));

  el.detailContent.appendChild(fragment);
}

function deriveRunLabel(records, source) {
  if (!records.length) return "No run loaded";
  const first = records[0];
  return `${first.run_id || "run"} | ${first.pipeline_name || "pipeline"} | ${records.length} records | ${source}`;
}

function applyData(records, sourceLabel) {
  state.records = records;
  state.sourceLabel = sourceLabel;
  refreshFilterOptions();
  applyFilters();
  el.runMeta.textContent = deriveRunLabel(records, sourceLabel);
  setStatus(`Loaded ${records.length} records from ${sourceLabel}.`);
}

async function loadFromPath(path) {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} while fetching ${path}`);
  }
  const text = await res.text();
  const records = parseJsonl(text);
  applyData(records, path);
}

async function resolveLatestRunPath() {
  const indexRes = await fetch("/outputs/P0/");
  if (!indexRes.ok) {
    throw new Error("Cannot access /outputs/P0/. Serve the project root with an HTTP server.");
  }
  const html = await indexRes.text();
  const matches = [...html.matchAll(/href="(run_\d+_\d+\.jsonl)"/g)].map((m) => m[1]);
  if (!matches.length) {
    throw new Error("No run_*.jsonl files found in /outputs/P0/.");
  }
  matches.sort();
  return `/outputs/P0/${matches[matches.length - 1]}`;
}

el.fileInput.addEventListener("change", async (evt) => {
  const file = evt.target.files?.[0];
  if (!file) return;

  try {
    const text = await file.text();
    const records = parseJsonl(text);
    applyData(records, file.name);
  } catch (err) {
    setStatus(err.message, true);
  }
});

el.loadPathBtn.addEventListener("click", async () => {
  const path = el.runPathInput.value.trim();
  if (!path) {
    setStatus("Please enter a path.", true);
    return;
  }
  try {
    await loadFromPath(path);
  } catch (err) {
    setStatus(err.message, true);
  }
});

el.loadLatestBtn.addEventListener("click", async () => {
  try {
    const latest = await resolveLatestRunPath();
    el.runPathInput.value = latest;
    await loadFromPath(latest);
  } catch (err) {
    setStatus(err.message, true);
  }
});

[el.paperFilter, el.typeFilter, el.answerableFilter].forEach((node) => {
  node.addEventListener("change", applyFilters);
});
el.searchInput.addEventListener("input", applyFilters);

setStatus("Load a run file to begin.");
