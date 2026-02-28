const state = {
  payloads: [],
  merged: null,
  pipelineLabels: [],
  weights: {
    w_qa: 0.6,
    w_gr: 0.2,
    w_ret: 0.1,
    w_eff: 0.1,
  },
  charts: {
    radar: null,
    bar: null,
  },
};

const el = {
  fileInput: document.getElementById("fileInput"),
  dropzone: document.getElementById("dropzone"),
  clearBtn: document.getElementById("clearBtn"),
  loaderMessage: document.getElementById("loaderMessage"),
  pipelineBadges: document.getElementById("pipelineBadges"),
  emptyState: document.getElementById("emptyState"),
  dashboardContent: document.getElementById("dashboardContent"),
  radarChart: document.getElementById("radarChart"),
  barChart: document.getElementById("barChart"),
  summaryTableBody: document.getElementById("summaryTableBody"),
  questionList: document.getElementById("questionList"),
  questionCount: document.getElementById("questionCount"),
  wQa: document.getElementById("wQa"),
  wGr: document.getElementById("wGr"),
  wRet: document.getElementById("wRet"),
  wEff: document.getElementById("wEff"),
  wQaValue: document.getElementById("wQaValue"),
  wGrValue: document.getElementById("wGrValue"),
  wRetValue: document.getElementById("wRetValue"),
  wEffValue: document.getElementById("wEffValue"),
  weightsSumLabel: document.getElementById("weightsSumLabel"),
  weightsWarning: document.getElementById("weightsWarning"),
};

const TYPE_BADGE_CLASS = {
  NUMERIC: "bg-blue-100 text-blue-800 border-blue-200",
  LIST: "bg-purple-100 text-purple-800 border-purple-200",
  FREE_TEXT: "bg-slate-100 text-slate-700 border-slate-200",
  STRING: "bg-gray-100 text-gray-700 border-gray-200",
  CATEGORICAL: "bg-amber-100 text-amber-800 border-amber-200",
};

function setLoaderMessage(message, kind = "info") {
  const styleByKind = {
    info: "border-slate-200 bg-slate-50 text-slate-600",
    success: "border-emerald-200 bg-emerald-50 text-emerald-700",
    error: "border-red-200 bg-red-50 text-red-700",
  };
  el.loaderMessage.className = `mt-4 rounded-lg border px-3 py-2 text-sm ${styleByKind[kind] || styleByKind.info}`;
  el.loaderMessage.textContent = message;
}

function ensurePayloadShape(payload, fileName) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error(`${fileName}: root must be a JSON object`);
  }

  const summary = payload.summary_metrics;
  const questions = payload.per_question_comparisons;

  if ((!summary || typeof summary !== "object" || Array.isArray(summary)) && !Array.isArray(questions)) {
    throw new Error(`${fileName}: expected summary_metrics object and/or per_question_comparisons array`);
  }

  return {
    summary_metrics: summary && typeof summary === "object" && !Array.isArray(summary) ? summary : {},
    per_question_comparisons: Array.isArray(questions) ? questions : [],
  };
}

function toNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function average(values) {
  if (!values.length) {
    return 0;
  }
  return values.reduce((sum, x) => sum + x, 0) / values.length;
}

function formatDecimal(value, digits = 3) {
  return toNumber(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function formatLatency(value) {
  if (!Number.isFinite(value)) {
    return "-";
  }
  return toNumber(value).toLocaleString(undefined, {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
}

function formatUsd(value) {
  const numeric = toNumber(value);
  return numeric.toLocaleString(undefined, {
    minimumFractionDigits: 6,
    maximumFractionDigits: 8,
  });
}

function normalizeRetrievedChunks(chunks) {
  if (!Array.isArray(chunks)) {
    return [];
  }
  return chunks
    .filter((chunk) => chunk && typeof chunk === "object")
    .map((chunk, idx) => ({
      rank: toNumber(chunk.rank, idx + 1),
      chunk_id: String(chunk.chunk_id || ""),
      score: toNumber(chunk.score, 0),
      text: String(chunk.text || ""),
      para_ids: Array.isArray(chunk.para_ids) ? chunk.para_ids.map((x) => String(x)) : [],
    }));
}

function normalizeEvidenceQuotes(quotes) {
  if (Array.isArray(quotes)) {
    return quotes
      .map((quote) => String(quote || "").trim())
      .filter((quote) => quote.length > 0);
  }
  if (quotes === undefined || quotes === null) {
    return [];
  }
  const single = String(quotes).trim();
  return single ? [single] : [];
}

function normalizeLoopHistory(history) {
  if (!Array.isArray(history)) {
    return [];
  }
  return history
    .filter((step) => step && typeof step === "object")
    .map((step, idx) => ({
      loop_index: toNumber(step.loop_index, idx + 1),
      query_used: String(step.query_used || ""),
      draft_answer: String(step.draft_answer || ""),
      critic_status: String(step.critic_status || ""),
      critic_feedback: String(step.critic_feedback || ""),
      critique_logic: String(step.critique_logic || ""),
      new_search_query: String(step.new_search_query || ""),
      final_answer: String(step.final_answer || ""),
      is_abstained: Boolean(step.is_abstained),
      forced_finalize: Boolean(step.forced_finalize),
      retrieved_chunk_count: toNumber(step.retrieved_chunk_count, 0),
      retrieved_chunk_ids: Array.isArray(step.retrieved_chunk_ids)
        ? step.retrieved_chunk_ids.map((x) => String(x || ""))
        : [],
    }));
}

function normalizePaperId(value) {
  if (value === undefined || value === null || value === "") {
    return null;
  }
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return null;
  }
  const pid = Math.trunc(n);
  return pid > 0 ? pid : null;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, toNumber(value, 0)));
}

function normalizeMinMax(value, min, max) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
    return 0;
  }
  return (toNumber(value, min) - min) / (max - min);
}

function readWeightsFromControls() {
  state.weights = {
    w_qa: toNumber(el.wQa?.value, 0.6),
    w_gr: toNumber(el.wGr?.value, 0.2),
    w_ret: toNumber(el.wRet?.value, 0.1),
    w_eff: toNumber(el.wEff?.value, 0.1),
  };
}

function renderWeightValues() {
  if (el.wQaValue) el.wQaValue.textContent = formatDecimal(state.weights.w_qa, 2);
  if (el.wGrValue) el.wGrValue.textContent = formatDecimal(state.weights.w_gr, 2);
  if (el.wRetValue) el.wRetValue.textContent = formatDecimal(state.weights.w_ret, 2);
  if (el.wEffValue) el.wEffValue.textContent = formatDecimal(state.weights.w_eff, 2);
}

function updateWeightWarning() {
  const weightSum =
    state.weights.w_qa + state.weights.w_gr + state.weights.w_ret + state.weights.w_eff;
  const isValid = Math.abs(weightSum - 1) < 1e-9;

  if (el.weightsSumLabel) {
    el.weightsSumLabel.textContent = `Sum: ${formatDecimal(weightSum, 2)}`;
    el.weightsSumLabel.classList.toggle("invalid", !isValid);
  }
  if (el.weightsWarning) {
    el.weightsWarning.classList.toggle("hidden", isValid);
  }
}

function handleWeightsChange() {
  readWeightsFromControls();
  renderWeightValues();
  updateWeightWarning();
  if (state.payloads.length) {
    rerender();
  }
}

function pickFirstNonEmpty(values) {
  for (const value of values) {
    if (value !== undefined && value !== null && String(value).trim() !== "") {
      return value;
    }
  }
  return "";
}

function mergePayloads(payloadEntries) {
  const merged = {
    summary_metrics: {},
    per_question_comparisons: [],
  };

  const questionMap = new Map();

  for (const entry of payloadEntries) {
    const payload = entry.payload;

    for (const [pipelineLabel, metrics] of Object.entries(payload.summary_metrics || {})) {
      const existing = merged.summary_metrics[pipelineLabel] || {};
      merged.summary_metrics[pipelineLabel] = {
        ...existing,
        ...(metrics && typeof metrics === "object" ? metrics : {}),
      };
    }

    for (const questionRow of payload.per_question_comparisons || []) {
      if (!questionRow || typeof questionRow !== "object") {
        continue;
      }

      const qid = questionRow.question_id;
      if (qid === undefined || qid === null) {
        continue;
      }
      const paperId = normalizePaperId(questionRow.paper_id);
      const qidNum = toNumber(qid, 0);
      const legacyKeySuffix = `${String(questionRow.question_text || "").trim()}::${String(questionRow.gold_answer || "").trim()}`;
      const key = paperId !== null ? `${paperId}::${qidNum}` : `legacy::${qidNum}::${legacyKeySuffix}`;

      const current = questionMap.get(key) || {
        paper_id: paperId,
        question_id: qidNum,
        question_type: "FREE_TEXT",
        question_text: "",
        gold_answer: "",
        pipelines: {},
      };

      current.paper_id = normalizePaperId(questionRow.paper_id) ?? current.paper_id;
      current.question_type = String(
        pickFirstNonEmpty([current.question_type, questionRow.question_type, "FREE_TEXT"]),
      ).toUpperCase();
      current.question_text = String(pickFirstNonEmpty([current.question_text, questionRow.question_text]));
      current.gold_answer = String(pickFirstNonEmpty([current.gold_answer, questionRow.gold_answer]));

      const pipelines = questionRow.pipelines;
      if (pipelines && typeof pipelines === "object" && !Array.isArray(pipelines)) {
        for (const [pipelineLabel, pipelineData] of Object.entries(pipelines)) {
          if (!pipelineData || typeof pipelineData !== "object") {
            continue;
          }
          const existingPipeline = current.pipelines[pipelineLabel] || {};
          const existingGeneration =
            existingPipeline.generation && typeof existingPipeline.generation === "object"
              ? existingPipeline.generation
              : {};
          const incomingGeneration =
            pipelineData.generation && typeof pipelineData.generation === "object"
              ? pipelineData.generation
              : {};
          const existingLoopHistory = normalizeLoopHistory(existingGeneration.loop_history);
          const incomingLoopHistory = normalizeLoopHistory(incomingGeneration.loop_history);
          const generationLoopHistory =
            incomingLoopHistory.length >= existingLoopHistory.length
              ? incomingLoopHistory
              : existingLoopHistory;
          const generationLoopCount = toNumber(
            incomingGeneration.loop_count,
            toNumber(existingGeneration.loop_count, 0),
          );
          const generationCriticStatus = String(
            pickFirstNonEmpty([existingGeneration.critic_status, incomingGeneration.critic_status, ""]),
          );
          const generationCriticFeedback = String(
            pickFirstNonEmpty([existingGeneration.critic_feedback, incomingGeneration.critic_feedback, ""]),
          );
          const generationNewSearchQuery = String(
            pickFirstNonEmpty([existingGeneration.new_search_query, incomingGeneration.new_search_query, ""]),
          );
          const generationReasoning = String(
            pickFirstNonEmpty([
              existingGeneration.reasoning,
              existingPipeline.reasoning,
              incomingGeneration.reasoning,
              pipelineData.reasoning,
              "",
            ]),
          );
          const generationCritiqueLogic = String(
            pickFirstNonEmpty([
              existingGeneration.critique_logic,
              existingPipeline.critique_logic,
              incomingGeneration.critique_logic,
              pipelineData.critique_logic,
              "",
            ]),
          );
          const generationEvidenceQuotes = (() => {
            const existingQuotes = normalizeEvidenceQuotes(existingGeneration.evidence_quotes).length
              ? normalizeEvidenceQuotes(existingGeneration.evidence_quotes)
              : normalizeEvidenceQuotes(existingPipeline.evidence_quotes);
            if (existingQuotes.length > 0) {
              return existingQuotes;
            }
            const incomingQuotes = normalizeEvidenceQuotes(incomingGeneration.evidence_quotes).length
              ? normalizeEvidenceQuotes(incomingGeneration.evidence_quotes)
              : normalizeEvidenceQuotes(pipelineData.evidence_quotes);
            return incomingQuotes;
          })();
          current.pipelines[pipelineLabel] = {
            ...existingPipeline,
            model_answer: String(
              pickFirstNonEmpty([existingPipeline.model_answer, pipelineData.model_answer, ""]),
            ),
            reasoning: generationReasoning,
            critique_logic: generationCritiqueLogic,
            evidence_quotes: generationEvidenceQuotes,
            generation: {
              reasoning: generationReasoning,
              evidence_quotes: generationEvidenceQuotes,
              critique_logic: generationCritiqueLogic,
              loop_history: generationLoopHistory,
              loop_count: generationLoopCount,
              critic_status: generationCriticStatus,
              critic_feedback: generationCriticFeedback,
              new_search_query: generationNewSearchQuery,
            },
            qa_score: toNumber(pipelineData.qa_score, toNumber(existingPipeline.qa_score, 0)),
            groundedness: toNumber(
              pipelineData.groundedness,
              toNumber(existingPipeline.groundedness, 0),
            ),
            judge_explanation: String(
              pickFirstNonEmpty([
                existingPipeline.judge_explanation,
                pipelineData.judge_explanation,
                "",
              ]),
            ),
            latency_ms: toNumber(pipelineData.latency_ms, toNumber(existingPipeline.latency_ms, 0)),
            cost_usd: toNumber(pipelineData.cost_usd, toNumber(existingPipeline.cost_usd, 0)),
            retrieval_score: toNumber(
              pipelineData.retrieval_score,
              toNumber(existingPipeline.retrieval_score, 0),
            ),
            gold_evidence_text: String(
              pickFirstNonEmpty([
                existingPipeline.gold_evidence_text,
                pipelineData.gold_evidence_text,
                "",
              ]),
            ),
            retrieved_context: String(
              pickFirstNonEmpty([
                existingPipeline.retrieved_context,
                pipelineData.retrieved_context,
                "",
              ]),
            ),
            retrieved_chunks:
              normalizeRetrievedChunks(existingPipeline.retrieved_chunks).length > 0
                ? normalizeRetrievedChunks(existingPipeline.retrieved_chunks)
                : normalizeRetrievedChunks(pipelineData.retrieved_chunks),
          };
        }
      }

      questionMap.set(key, current);
    }
  }

  merged.per_question_comparisons = [...questionMap.values()].sort(
    (a, b) =>
      toNumber(a.paper_id, Number.MAX_SAFE_INTEGER) - toNumber(b.paper_id, Number.MAX_SAFE_INTEGER) ||
      toNumber(a.question_id) - toNumber(b.question_id),
  );

  return merged;
}

function collectPipelineLabels(merged) {
  const labels = new Set(Object.keys(merged.summary_metrics || {}));
  for (const row of merged.per_question_comparisons || []) {
    for (const label of Object.keys(row.pipelines || {})) {
      labels.add(label);
    }
  }
  return [...labels].sort();
}

function buildPipelineStats(merged, labels) {
  return labels.map((label) => {
    const summary = merged.summary_metrics[label] || {};

    const questionRecords = merged.per_question_comparisons
      .map((q) => q.pipelines?.[label])
      .filter((item) => item && typeof item === "object");

    const qaFromQuestions = average(questionRecords.map((q) => toNumber(q.qa_score, 0)));
    const groundedFromQuestions = average(questionRecords.map((q) => toNumber(q.groundedness, 0)));
    const retrievalFromQuestions = average(questionRecords.map((q) => toNumber(q.retrieval_score, 0)));
    const latencyFromQuestions = average(questionRecords.map((q) => toNumber(q.latency_ms, 0)));
    const costFromQuestions = average(questionRecords.map((q) => toNumber(q.cost_usd, 0)));

    const hallucinations = questionRecords.filter((q) => toNumber(q.groundedness, 0) < 0.5).length;
    const hallucinationRateFromQuestions =
      questionRecords.length > 0 ? hallucinations / questionRecords.length : 0;

    return {
      label,
      qaMicroScore: toNumber(summary.qa_micro_score, qaFromQuestions),
      meanGroundedness: toNumber(summary.mean_groundedness, groundedFromQuestions),
      meanRetrievalScore: toNumber(summary.mean_retrieval_score, retrievalFromQuestions),
      meanCostUsd: toNumber(summary.mean_cost_usd, costFromQuestions),
      hallucinationRate: toNumber(summary.hallucination_rate, hallucinationRateFromQuestions),
      meanLatencyMs: toNumber(summary.mean_latency_ms, latencyFromQuestions),
      efficiency: 0,
      finalScore: toNumber(summary.final_score, 0),
    };
  });
}

function applyDynamicScores(stats) {
  const latencies = stats.map((s) => s.meanLatencyMs);
  const costs = stats.map((s) => s.meanCostUsd);
  const minLatency = Math.min(...latencies);
  const maxLatency = Math.max(...latencies);
  const minCost = Math.min(...costs);
  const maxCost = Math.max(...costs);

  for (const s of stats) {
    const normLatency = normalizeMinMax(s.meanLatencyMs, minLatency, maxLatency);
    const normCost = normalizeMinMax(s.meanCostUsd, minCost, maxCost);

    const efficiencyScore = 0.5 * (1 - normLatency) + 0.5 * (1 - normCost);
    s.efficiency = clamp01(efficiencyScore);
    s.finalScore =
      state.weights.w_qa * s.qaMicroScore +
      state.weights.w_gr * s.meanGroundedness +
      state.weights.w_ret * s.meanRetrievalScore +
      state.weights.w_eff * s.efficiency;
  }
}

function pipelineColor(index, total) {
  const hue = Math.round((360 * index) / Math.max(total, 1));
  return {
    stroke: `hsl(${hue} 78% 42%)`,
    fill: `hsla(${hue}, 78%, 42%, 0.15)`,
    solid: `hsl(${hue} 70% 48%)`,
    soft: `hsla(${hue}, 70%, 48%, 0.14)`,
  };
}

function destroyCharts() {
  if (state.charts.radar) {
    state.charts.radar.destroy();
    state.charts.radar = null;
  }
  if (state.charts.bar) {
    state.charts.bar.destroy();
    state.charts.bar = null;
  }
}

function renderCharts(stats) {
  destroyCharts();

  const radarLabels = ["Final Score", "QA Score", "Groundedness", "Retrieval", "Efficiency"];
  const total = stats.length;

  const radarDatasets = stats.map((row, idx) => {
    const color = pipelineColor(idx, total);
    return {
      label: row.label,
      data: [
        row.finalScore,
        row.qaMicroScore,
        row.meanGroundedness,
        row.meanRetrievalScore,
        row.efficiency,
      ],
      borderColor: color.stroke,
      backgroundColor: color.fill,
      borderWidth: 2,
      pointRadius: 3,
      pointBackgroundColor: color.stroke,
    };
  });

  state.charts.radar = new Chart(el.radarChart.getContext("2d"), {
    type: "radar",
    data: {
      labels: radarLabels,
      datasets: radarDatasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 0,
          max: 1,
          ticks: {
            stepSize: 0.25,
          },
        },
      },
      plugins: {
        legend: {
          position: "bottom",
        },
      },
    },
  });

  state.charts.bar = new Chart(el.barChart.getContext("2d"), {
    type: "bar",
    data: {
      labels: stats.map((s) => s.label),
      datasets: [
        {
          label: "Final Score",
          data: stats.map((s) => s.finalScore),
          backgroundColor: stats.map((_, idx) => pipelineColor(idx, total).solid),
          borderColor: stats.map((_, idx) => pipelineColor(idx, total).stroke),
          borderWidth: 1,
          borderRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 1,
        },
      },
      plugins: {
        legend: {
          display: false,
        },
      },
    },
  });
}

function bestValueByColumn(stats, columnKey, mode) {
  const values = stats.map((s) => s[columnKey]).filter((v) => Number.isFinite(v));
  if (!values.length) {
    return null;
  }
  return mode === "max" ? Math.max(...values) : Math.min(...values);
}

function renderSummaryTable(stats) {
  el.summaryTableBody.innerHTML = "";

  const bestMax = {
    finalScore: bestValueByColumn(stats, "finalScore", "max"),
    qaMicroScore: bestValueByColumn(stats, "qaMicroScore", "max"),
    meanGroundedness: bestValueByColumn(stats, "meanGroundedness", "max"),
    meanRetrievalScore: bestValueByColumn(stats, "meanRetrievalScore", "max"),
  };
  const bestMin = {
    hallucinationRate: bestValueByColumn(stats, "hallucinationRate", "min"),
    meanLatencyMs: bestValueByColumn(stats, "meanLatencyMs", "min"),
    meanCostUsd: bestValueByColumn(stats, "meanCostUsd", "min"),
  };

  for (const row of stats) {
    const tr = document.createElement("tr");

    const values = [
      { key: "label", content: row.label, align: "text-left", best: false },
      {
        key: "finalScore",
        content: formatDecimal(row.finalScore, 4),
        align: "text-right",
        best: row.finalScore === bestMax.finalScore,
      },
      {
        key: "qaMicroScore",
        content: formatDecimal(row.qaMicroScore, 4),
        align: "text-right",
        best: row.qaMicroScore === bestMax.qaMicroScore,
      },
      {
        key: "meanGroundedness",
        content: formatDecimal(row.meanGroundedness, 4),
        align: "text-right",
        best: row.meanGroundedness === bestMax.meanGroundedness,
      },
      {
        key: "meanRetrievalScore",
        content: formatDecimal(row.meanRetrievalScore, 4),
        align: "text-right",
        best: row.meanRetrievalScore === bestMax.meanRetrievalScore,
      },
      {
        key: "hallucinationRate",
        content: formatDecimal(row.hallucinationRate, 4),
        align: "text-right",
        best: row.hallucinationRate === bestMin.hallucinationRate,
      },
      {
        key: "meanLatencyMs",
        content: formatLatency(row.meanLatencyMs),
        align: "text-right",
        best: row.meanLatencyMs === bestMin.meanLatencyMs,
      },
      {
        key: "meanCostUsd",
        content: formatUsd(row.meanCostUsd),
        align: "text-right",
        best: row.meanCostUsd === bestMin.meanCostUsd,
      },
    ];

    for (const cell of values) {
      const td = document.createElement("td");
      td.className = `px-3 py-2 ${cell.align}`;
      if (cell.best && cell.key !== "label") {
        td.classList.add("best-value");
      }
      td.textContent = cell.content;
      tr.appendChild(td);
    }

    el.summaryTableBody.appendChild(tr);
  }
}

function qaBadgeClass(score) {
  const s = toNumber(score, 0);
  if (s >= 0.8) {
    return "bg-emerald-100 text-emerald-800 border-emerald-200";
  }
  if (s >= 0.5) {
    return "bg-amber-100 text-amber-800 border-amber-200";
  }
  return "bg-rose-100 text-rose-800 border-rose-200";
}

function typeBadgeClass(type) {
  const key = String(type || "FREE_TEXT").toUpperCase();
  return TYPE_BADGE_CLASS[key] || TYPE_BADGE_CLASS.FREE_TEXT;
}

function renderPipelineBadges(labels) {
  el.pipelineBadges.innerHTML = "";
  if (!labels.length) {
    const span = document.createElement("span");
    span.className = "rounded-md border border-slate-200 bg-slate-100 px-2 py-1 text-xs text-slate-500";
    span.textContent = "No pipelines detected";
    el.pipelineBadges.appendChild(span);
    return;
  }

  labels.forEach((label, idx) => {
    const color = pipelineColor(idx, labels.length);
    const span = document.createElement("span");
    span.className = "rounded-full border px-3 py-1 text-xs font-semibold";
    span.style.borderColor = color.stroke;
    span.style.backgroundColor = color.soft;
    span.style.color = color.stroke;
    span.textContent = label;
    el.pipelineBadges.appendChild(span);
  });
}

function createMetricPill(label, value, className = "") {
  const pill = document.createElement("span");
  pill.className = `rounded-full border px-2 py-1 text-xs font-medium ${className}`.trim();
  pill.textContent = `${label}: ${value}`;
  return pill;
}

function createEvidencePanel(pdata) {
  const details = document.createElement("details");
  details.className = "evidence-details mt-2 rounded-md border border-slate-200 bg-white";

  const summary = document.createElement("summary");
  summary.className = "cursor-pointer px-3 py-2 text-xs font-semibold text-slate-700";
  summary.textContent = "Evidence: Gold vs Retrieved Chunks";
  details.appendChild(summary);

  const body = document.createElement("div");
  body.className = "space-y-3 border-t border-slate-200 px-3 py-3";

  const goldWrap = document.createElement("div");
  const goldTitle = document.createElement("p");
  goldTitle.className = "mb-1 text-xs font-semibold text-amber-700";
  goldTitle.textContent = "Gold Chunk";
  const goldText = document.createElement("pre");
  goldText.className = "evidence-pre rounded-md border border-amber-200 bg-amber-50 p-2 text-xs text-amber-900";
  goldText.textContent =
    pdata.gold_evidence_text ||
    "No gold evidence text available. This file may have been generated before evidence export was added.";
  goldWrap.appendChild(goldTitle);
  goldWrap.appendChild(goldText);
  body.appendChild(goldWrap);

  const retrievedWrap = document.createElement("div");
  const retrievedTitle = document.createElement("p");
  retrievedTitle.className = "mb-1 text-xs font-semibold text-sky-700";
  retrievedTitle.textContent = "Retrieved Chunks";
  retrievedWrap.appendChild(retrievedTitle);

  const chunks = normalizeRetrievedChunks(pdata.retrieved_chunks);
  if (chunks.length) {
    let addedDivider = false;
    chunks.forEach((chunk) => {
      const isReretrievalChunk = toNumber(chunk.rank, 0) > 5;
      // Add visual separator for Re-Retrieval chunks.
      if (isReretrievalChunk && !addedDivider) {
        const divider = document.createElement("div");
        divider.className = "my-4 flex items-center gap-2";
        divider.innerHTML = `<span class="h-px w-full bg-blue-200"></span><span class="text-[10px] font-bold uppercase text-blue-600 whitespace-nowrap bg-blue-50 px-2 rounded-full border border-blue-200">Re-Retrieval Chunks</span><span class="h-px w-full bg-blue-200"></span>`;
        retrievedWrap.appendChild(divider);
        addedDivider = true;
      }

      const chunkBox = document.createElement("div");
      chunkBox.className = isReretrievalChunk
        ? "mb-2 rounded-md border border-blue-200 bg-blue-50 p-2"
        : "mb-2 rounded-md border border-sky-200 bg-sky-50 p-2";

      const chunkMeta = document.createElement("p");
      chunkMeta.className = isReretrievalChunk
        ? "mb-1 text-xs font-semibold text-blue-800"
        : "mb-1 text-xs font-semibold text-sky-800";
      chunkMeta.textContent = `Rank ${toNumber(chunk.rank, 0)} | score ${formatDecimal(chunk.score, 4)}${chunk.chunk_id ? ` | ${chunk.chunk_id}` : ""}`;
      chunkBox.appendChild(chunkMeta);

      const chunkText = document.createElement("pre");
      chunkText.className = "evidence-pre text-xs text-slate-800";
      chunkText.textContent = chunk.text || "(empty chunk text)";
      chunkBox.appendChild(chunkText);

      retrievedWrap.appendChild(chunkBox);
    });
  } else {
    const contextText = document.createElement("pre");
    contextText.className = "evidence-pre rounded-md border border-slate-200 bg-slate-50 p-2 text-xs text-slate-700";
    contextText.textContent =
      pdata.retrieved_context ||
      "No retrieved context available. This file may have been generated before evidence export was added.";
    retrievedWrap.appendChild(contextText);
  }

  body.appendChild(retrievedWrap);
  details.appendChild(body);
  return details;
}

function renderQuestionCards(merged, labels) {
  const questions = merged.per_question_comparisons || [];
  el.questionCount.textContent = `${questions.length} question${questions.length === 1 ? "" : "s"}`;
  el.questionList.innerHTML = "";

  if (!questions.length) {
    const empty = document.createElement("div");
    empty.className = "rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600";
    empty.textContent = "No per-question comparison data available.";
    el.questionList.appendChild(empty);
    return;
  }

  for (const question of questions) {
    const details = document.createElement("details");
    details.className = "question-details";

    const summary = document.createElement("summary");
    summary.className =
      "flex cursor-pointer items-start justify-between gap-3 rounded-xl px-4 py-3 hover:bg-slate-50";

    const left = document.createElement("div");
    left.className = "min-w-0";

    const topLine = document.createElement("div");
    topLine.className = "mb-1 flex flex-wrap items-center gap-2";

    const pid = document.createElement("span");
    pid.className = "rounded-md border border-slate-200 bg-slate-100 px-2 py-1 text-xs font-semibold text-slate-700";
    const paperId = normalizePaperId(question.paper_id);
    pid.textContent = paperId !== null ? `P${paperId}` : "Paper ?";

    const qid = document.createElement("span");
    qid.className = "rounded-md border border-slate-200 bg-slate-100 px-2 py-1 text-xs font-semibold text-slate-700";
    qid.textContent = `Q${toNumber(question.question_id, 0)}`;

    const type = document.createElement("span");
    type.className = `rounded-full border px-2 py-1 text-xs font-semibold ${typeBadgeClass(question.question_type)}`;
    type.textContent = String(question.question_type || "FREE_TEXT").toUpperCase();

    topLine.appendChild(pid);
    topLine.appendChild(qid);
    topLine.appendChild(type);

    const text = document.createElement("p");
    text.className = "truncate text-sm font-medium text-slate-800";
    text.textContent = question.question_text || "(No question text)";

    left.appendChild(topLine);
    left.appendChild(text);

    const chevron = document.createElement("span");
    chevron.className = "summary-chevron mt-1 text-slate-400";
    chevron.textContent = ">";

    summary.appendChild(left);
    summary.appendChild(chevron);

    const body = document.createElement("div");
    body.className = "border-t border-slate-200 px-4 pb-4 pt-3";

    const goldWrap = document.createElement("div");
    goldWrap.className = "mb-4 rounded-lg border border-amber-200 bg-amber-50 p-3";

    const goldTitle = document.createElement("h4");
    goldTitle.className = "mb-1 text-sm font-semibold text-amber-800";
    goldTitle.textContent = "Gold Answer";

    const goldText = document.createElement("p");
    goldText.className = "answer-text text-sm text-amber-900";
    goldText.textContent = question.gold_answer || "(No gold answer)";

    goldWrap.appendChild(goldTitle);
    goldWrap.appendChild(goldText);

    const grid = document.createElement("div");
    grid.className = "grid grid-cols-1 gap-3 lg:grid-cols-2";

    labels.forEach((label, idx) => {
      const pdata = question.pipelines?.[label];
      const color = pipelineColor(idx, labels.length);

      const card = document.createElement("article");
      card.className = "rounded-lg border border-slate-200 bg-slate-50 p-3";
      card.style.borderLeftWidth = "4px";
      card.style.borderLeftColor = color.stroke;

      const head = document.createElement("div");
      head.className = "mb-2 flex items-center justify-between";

      const title = document.createElement("h5");
      title.className = "text-sm font-semibold text-slate-800";
      title.textContent = label;

      const criticStatus = pdata?.generation?.critic_status;
      if (criticStatus) {
        const badge = document.createElement("span");
        let badgeStyle = "bg-slate-100 text-slate-800 border-slate-200";
        if (criticStatus === "ACCEPT") badgeStyle = "bg-emerald-100 text-emerald-800 border-emerald-200";
        else if (criticStatus === "REVISE") badgeStyle = "bg-amber-100 text-amber-800 border-amber-200";
        else if (criticStatus === "RE_RETRIEVE") badgeStyle = "bg-blue-100 text-blue-800 border-blue-200";
        else if (criticStatus === "ABSTAIN") badgeStyle = "bg-rose-100 text-rose-800 border-rose-200";

        badge.className = `ml-2 rounded-full border px-2 py-0.5 text-[10px] font-bold tracking-wide ${badgeStyle}`;
        badge.textContent = criticStatus;
        title.appendChild(badge);
      }

      head.appendChild(title);
      card.appendChild(head);

      if (!pdata) {
        const noData = document.createElement("p");
        noData.className = "text-sm text-slate-500";
        noData.textContent = "No data for this pipeline.";
        card.appendChild(noData);
      } else {
        const answer = document.createElement("p");
        answer.className = "answer-text mb-3 text-sm text-slate-700";
        answer.textContent = pdata.model_answer || "(empty answer)";
        card.appendChild(answer);

        const metricRow = document.createElement("div");
        metricRow.className = "mb-3 flex flex-wrap gap-2";

        metricRow.appendChild(
          createMetricPill("QA", formatDecimal(pdata.qa_score, 3), qaBadgeClass(pdata.qa_score)),
        );
        metricRow.appendChild(
          createMetricPill(
            "Groundedness",
            formatDecimal(pdata.groundedness, 3),
            qaBadgeClass(pdata.groundedness),
          ),
        );
        metricRow.appendChild(
          createMetricPill(
            "Retrieval",
            formatDecimal(pdata.retrieval_score, 3),
            qaBadgeClass(pdata.retrieval_score),
          ),
        );
        metricRow.appendChild(
          createMetricPill(
            "Latency",
            `${formatLatency(pdata.latency_ms)} ms`,
            "bg-slate-100 text-slate-700 border-slate-200",
          ),
        );
        metricRow.appendChild(
          createMetricPill(
            "Cost",
            `$${formatUsd(pdata.cost_usd)}`,
            "bg-slate-100 text-slate-700 border-slate-200",
          ),
        );

        card.appendChild(metricRow);

        const judge = document.createElement("div");
        judge.className = "judge-note rounded-md border border-slate-200 bg-white p-2 text-xs text-slate-600";
        judge.textContent = `[AI Judge] ${pdata.judge_explanation || "No explanation provided."}`;
        card.appendChild(judge);

        const reasoning = String(pdata.generation?.reasoning || pdata.reasoning || "").trim();
        const evidenceQuotes = normalizeEvidenceQuotes(
          pdata.generation?.evidence_quotes ?? pdata.evidence_quotes,
        );
        if (reasoning || evidenceQuotes.length > 0) {
          const explainer = document.createElement("div");
          explainer.className = "mt-2 rounded-md border border-indigo-200 bg-indigo-50 p-2 text-xs text-indigo-900";

          const title = document.createElement("p");
          title.className = "mb-1 text-xs font-semibold text-indigo-800";
          title.textContent = "Evidence & Reasoning";
          explainer.appendChild(title);

          if (reasoning) {
            const reasoningText = document.createElement("p");
            reasoningText.className = "mb-2 whitespace-pre-wrap text-xs text-indigo-900";
            reasoningText.textContent = reasoning;
            explainer.appendChild(reasoningText);
          }

          if (evidenceQuotes.length > 0) {
            const quoteList = document.createElement("ul");
            quoteList.className = "list-disc space-y-1 pl-4 text-xs";
            evidenceQuotes.forEach((quote) => {
              const item = document.createElement("li");
              item.textContent = quote;
              quoteList.appendChild(item);
            });
            explainer.appendChild(quoteList);
          }

          card.appendChild(explainer);
        }

        const loopCount = toNumber(pdata.generation?.loop_count, 0);
        const criticFeedback = pdata.generation?.critic_feedback;
        const newQuery = pdata.generation?.new_search_query;
        const criticStatusText = String(pdata.generation?.critic_status || "").trim();
        const critiqueLogic = String(pdata.generation?.critique_logic || pdata.critique_logic || "").trim();
        const displayLogic =
          critiqueLogic ||
          (loopCount > 0
            ? `Critic status: ${criticStatusText || "UNKNOWN"}${
                criticFeedback ? " (feedback provided below)." : " (accepted without additional correction details)."
              }`
            : "");

        if (loopCount > 0 || criticFeedback || newQuery || critiqueLogic) {
          const traceBox = document.createElement("div");
          traceBox.className = "mt-3 rounded-md border border-fuchsia-200 bg-fuchsia-50 p-3 text-xs text-fuchsia-900";

          const traceTitle = document.createElement("p");
          traceTitle.className = "mb-2 text-xs font-bold text-fuchsia-800 uppercase tracking-wider";
          traceTitle.textContent =
            loopCount > 0 ? `🤖 Agent Trace (Loops: ${loopCount})` : "🤖 Critic Filter (Single Pass)";
          traceBox.appendChild(traceTitle);

          if (displayLogic) {
             const logicText = document.createElement("p");
             logicText.className = "whitespace-pre-wrap mb-1";
             logicText.innerHTML = `<strong>Logic:</strong> ${displayLogic}`;
             traceBox.appendChild(logicText);
          }
          if (criticFeedback) {
             const feedbackText = document.createElement("p");
             feedbackText.className = "whitespace-pre-wrap mb-1 text-rose-700";
             feedbackText.innerHTML = `<strong>Critic Feedback:</strong> ${criticFeedback}`;
             traceBox.appendChild(feedbackText);
          }
          if (newQuery) {
             const queryText = document.createElement("p");
             queryText.className = "whitespace-pre-wrap font-mono bg-white border border-fuchsia-200 p-2 mt-2 rounded shadow-sm text-blue-800";
             queryText.innerHTML = `<strong>Re-Retrieval Query:</strong><br/>"${newQuery}"`;
             traceBox.appendChild(queryText);
          }
          card.appendChild(traceBox);
        }

        const loopHistory = normalizeLoopHistory(pdata.generation?.loop_history);
        if (loopHistory.length > 0) {
          const timeline = document.createElement("details");
          timeline.className = "mt-3 rounded-md border border-violet-200 bg-violet-50";

          const timelineSummary = document.createElement("summary");
          timelineSummary.className = "cursor-pointer px-3 py-2 text-xs font-semibold text-violet-800";
          timelineSummary.textContent = `Loop History (${loopHistory.length} step${loopHistory.length === 1 ? "" : "s"})`;
          timeline.appendChild(timelineSummary);

          const timelineBody = document.createElement("div");
          timelineBody.className = "space-y-2 border-t border-violet-200 px-3 py-3";

          loopHistory.forEach((step) => {
            const stepBox = document.createElement("div");
            stepBox.className = "rounded-md border border-violet-200 bg-white p-2 text-xs text-violet-900";

            const status = String(step.critic_status || "").trim();
            let statusClass = "bg-slate-100 text-slate-800 border-slate-200";
            if (status === "ACCEPT") statusClass = "bg-emerald-100 text-emerald-800 border-emerald-200";
            else if (status === "REVISE") statusClass = "bg-amber-100 text-amber-800 border-amber-200";
            else if (status === "RE_RETRIEVE") statusClass = "bg-blue-100 text-blue-800 border-blue-200";
            else if (status === "ABSTAIN") statusClass = "bg-rose-100 text-rose-800 border-rose-200";

            const header = document.createElement("div");
            header.className = "mb-1 flex flex-wrap items-center gap-2";
            const loopLabel = document.createElement("span");
            loopLabel.className = "font-semibold text-violet-800";
            loopLabel.textContent = `Loop ${toNumber(step.loop_index, 0)}`;
            header.appendChild(loopLabel);
            if (status) {
              const statusBadge = document.createElement("span");
              statusBadge.className = `rounded-full border px-2 py-0.5 text-[10px] font-bold tracking-wide ${statusClass}`;
              statusBadge.textContent = status;
              header.appendChild(statusBadge);
            }
            if (step.forced_finalize) {
              const forcedTag = document.createElement("span");
              forcedTag.className = "rounded-full border border-orange-200 bg-orange-100 px-2 py-0.5 text-[10px] font-bold tracking-wide text-orange-800";
              forcedTag.textContent = "MAX LOOP FALLBACK";
              header.appendChild(forcedTag);
            }
            stepBox.appendChild(header);

            const queryText = document.createElement("p");
            queryText.className = "mb-1 whitespace-pre-wrap";
            queryText.innerHTML = `<strong>Query used:</strong> ${step.query_used || "(none)"}`;
            stepBox.appendChild(queryText);

            if (step.new_search_query) {
              const newQueryText = document.createElement("p");
              newQueryText.className = "mb-1 whitespace-pre-wrap text-blue-800";
              newQueryText.innerHTML = `<strong>Rephrase query:</strong> ${step.new_search_query}`;
              stepBox.appendChild(newQueryText);
            }

            if (step.critic_feedback) {
              const feedbackText = document.createElement("p");
              feedbackText.className = "mb-1 whitespace-pre-wrap text-rose-700";
              feedbackText.innerHTML = `<strong>Feedback:</strong> ${step.critic_feedback}`;
              stepBox.appendChild(feedbackText);
            }

            const chunkCountText = document.createElement("p");
            chunkCountText.className = "mb-1";
            chunkCountText.innerHTML = `<strong>Retrieved chunks:</strong> ${toNumber(step.retrieved_chunk_count, 0)}`;
            stepBox.appendChild(chunkCountText);

            if (step.retrieved_chunk_ids.length > 0) {
              const chunkIdsText = document.createElement("p");
              chunkIdsText.className = "mb-1 whitespace-pre-wrap font-mono text-[11px] text-violet-700";
              chunkIdsText.innerHTML = `<strong>Chunk IDs:</strong> ${step.retrieved_chunk_ids.join(", ")}`;
              stepBox.appendChild(chunkIdsText);
            }

            if (step.final_answer) {
              const finalText = document.createElement("p");
              finalText.className = "whitespace-pre-wrap";
              finalText.innerHTML = `<strong>Final answer at this step:</strong> ${step.final_answer}`;
              stepBox.appendChild(finalText);
            }

            timelineBody.appendChild(stepBox);
          });

          timeline.appendChild(timelineBody);
          card.appendChild(timeline);
        }

        card.appendChild(createEvidencePanel(pdata));
      }

      grid.appendChild(card);
    });

    body.appendChild(goldWrap);
    body.appendChild(grid);

    details.appendChild(summary);
    details.appendChild(body);
    el.questionList.appendChild(details);
  }
}

function toggleDashboardVisibility(hasData) {
  el.emptyState.classList.toggle("hidden", hasData);
  el.dashboardContent.classList.toggle("hidden", !hasData);
}

function rerender() {
  if (!state.payloads.length) {
    state.merged = null;
    state.pipelineLabels = [];
    toggleDashboardVisibility(false);
    renderPipelineBadges([]);
    destroyCharts();
    el.summaryTableBody.innerHTML = "";
    el.questionList.innerHTML = "";
    el.questionCount.textContent = "0 questions";
    return;
  }

  state.merged = mergePayloads(state.payloads);
  state.pipelineLabels = collectPipelineLabels(state.merged);

  if (!state.pipelineLabels.length) {
    toggleDashboardVisibility(false);
    renderPipelineBadges([]);
    setLoaderMessage("No valid pipeline metrics detected after merge.", "error");
    return;
  }

  const stats = buildPipelineStats(state.merged, state.pipelineLabels);
  applyDynamicScores(stats);

  renderPipelineBadges(state.pipelineLabels);
  renderCharts(stats);
  renderSummaryTable(stats);
  renderQuestionCards(state.merged, state.pipelineLabels);

  toggleDashboardVisibility(true);
  const missingPaperIds = state.merged.per_question_comparisons.filter((row) => normalizePaperId(row.paper_id) === null).length;
  let missingEvidenceRows = 0;
  state.merged.per_question_comparisons.forEach((row) => {
    Object.values(row.pipelines || {}).forEach((pdata) => {
      if (!pdata || typeof pdata !== "object") {
        return;
      }
      const hasGold = String(pdata.gold_evidence_text || "").trim().length > 0;
      const hasChunks = normalizeRetrievedChunks(pdata.retrieved_chunks).length > 0;
      const hasContext = String(pdata.retrieved_context || "").trim().length > 0;
      if (!hasGold || (!hasChunks && !hasContext)) {
        missingEvidenceRows += 1;
      }
    });
  });

  const warningParts = [];
  if (missingPaperIds > 0) {
    warningParts.push(`${missingPaperIds} legacy rows missing paper_id`);
  }
  if (missingEvidenceRows > 0) {
    warningParts.push(`${missingEvidenceRows} rows missing evidence export fields`);
  }
  const warningSuffix = warningParts.length ? ` | Note: ${warningParts.join("; ")}.` : "";
  setLoaderMessage(
    `Loaded ${state.payloads.length} file(s). Detected pipelines: ${state.pipelineLabels.join(", ")}.${warningSuffix}`,
    warningParts.length ? "info" : "success",
  );
}

async function readJsonFile(file) {
  const text = await file.text();
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    throw new Error(`${file.name}: invalid JSON (${err.message})`);
  }
  return ensurePayloadShape(parsed, file.name);
}

async function handleFiles(fileList) {
  const files = [...fileList];
  if (!files.length) {
    return;
  }

  const errors = [];
  const loadedEntries = [];

  for (const file of files) {
    try {
      const payload = await readJsonFile(file);
      loadedEntries.push({
        name: file.name,
        payload,
      });
    } catch (err) {
      errors.push(String(err.message || err));
    }
  }

  if (loadedEntries.length) {
    state.payloads.push(...loadedEntries);
    rerender();
  }

  if (errors.length) {
    setLoaderMessage(`Some files failed to load: ${errors.join(" | ")}`, "error");
  }
}

function clearAllData() {
  state.payloads = [];
  rerender();
  setLoaderMessage("Data cleared. Upload evaluator JSON files to begin.", "info");
  el.fileInput.value = "";
}

function preventDefaults(evt) {
  evt.preventDefault();
  evt.stopPropagation();
}

["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
  el.dropzone.addEventListener(eventName, preventDefaults);
});

["dragenter", "dragover"].forEach((eventName) => {
  el.dropzone.addEventListener(eventName, () => {
    el.dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  el.dropzone.addEventListener(eventName, () => {
    el.dropzone.classList.remove("dragover");
  });
});

el.dropzone.addEventListener("drop", (evt) => {
  const dt = evt.dataTransfer;
  if (!dt || !dt.files) {
    return;
  }
  handleFiles(dt.files);
});

el.dropzone.addEventListener("click", () => {
  el.fileInput.click();
});

el.fileInput.addEventListener("change", (evt) => {
  const files = evt.target.files;
  if (!files || !files.length) {
    return;
  }
  handleFiles(files);
});

el.clearBtn.addEventListener("click", clearAllData);

[el.wQa, el.wGr, el.wRet, el.wEff].forEach((slider) => {
  slider?.addEventListener("input", handleWeightsChange);
});

readWeightsFromControls();
renderWeightValues();
updateWeightWarning();
rerender();
setLoaderMessage("Please upload ui_dashboard_data JSON files to begin.", "info");
