const DATA_URL = "./data/results.json";

const state = {
  data: null,
  filtered: [],
  expandedAll: false,
};

const el = {
  questionTotal: document.getElementById("questionTotal"),
  paperTotal: document.getElementById("paperTotal"),
  visibleTotal: document.getElementById("visibleTotal"),
  pipelineLegend: document.getElementById("pipelineLegend"),
  searchInput: document.getElementById("searchInput"),
  paperFilter: document.getElementById("paperFilter"),
  typeFilter: document.getElementById("typeFilter"),
  answerFilter: document.getElementById("answerFilter"),
  expandAllBtn: document.getElementById("expandAllBtn"),
  collapseAllBtn: document.getElementById("collapseAllBtn"),
  resetBtn: document.getElementById("resetBtn"),
  message: document.getElementById("message"),
  questionList: document.getElementById("questionList"),
};

function asText(value) {
  if (value === undefined || value === null) {
    return "";
  }
  return String(value);
}

function normalizeType(value) {
  return asText(value || "UNKNOWN").trim().toUpperCase() || "UNKNOWN";
}

function isAnswerMissing(value) {
  const text = asText(value).trim().toLowerCase();
  return (
    text === "" ||
    text === "none" ||
    text === "null" ||
    text === "n/a" ||
    text === "na" ||
    text === "not available"
  );
}

function compareQuestion(a, b) {
  return (
    Number(a.paper_id || 0) - Number(b.paper_id || 0) ||
    Number(a.question_id || 0) - Number(b.question_id || 0)
  );
}

function uniqueSorted(values, numeric = false) {
  const items = [...new Set(values.filter((value) => value !== undefined && value !== null))];
  return items.sort((a, b) => (numeric ? Number(a) - Number(b) : String(a).localeCompare(String(b))));
}

function setMessage(text, visible = true) {
  el.message.textContent = text;
  el.message.classList.toggle("hidden", !visible);
}

function makeOption(value, label) {
  const option = document.createElement("option");
  option.value = String(value);
  option.textContent = label;
  return option;
}

function clearChildren(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function populateFilters() {
  const questions = state.data.questions || [];
  const papers = uniqueSorted(
    questions.map((question) => question.paper_id),
    true,
  );
  const types = uniqueSorted(questions.map((question) => normalizeType(question.question_type)));

  clearChildren(el.paperFilter);
  el.paperFilter.appendChild(makeOption("all", "All papers"));
  papers.forEach((paperId) => {
    el.paperFilter.appendChild(makeOption(String(paperId), `Paper ${paperId}`));
  });

  clearChildren(el.typeFilter);
  el.typeFilter.appendChild(makeOption("all", "All types"));
  types.forEach((type) => {
    el.typeFilter.appendChild(makeOption(type, type.replace("_", " ")));
  });
}

function renderHeader() {
  const questions = state.data.questions || [];
  const papers = uniqueSorted(
    questions.map((question) => question.paper_id),
    true,
  );
  el.questionTotal.textContent = `${questions.length} questions`;
  el.paperTotal.textContent = `${papers.length} papers`;

  clearChildren(el.pipelineLegend);
  (state.data.pipelines || []).forEach((pipeline) => {
    const chip = document.createElement("span");
    chip.className = "legend-chip";
    chip.textContent = pipeline.label;
    el.pipelineLegend.appendChild(chip);
  });
}

function rowHasMissingAnswer(question) {
  return (state.data.pipelines || []).some((pipeline) =>
    isAnswerMissing(question.answers?.[pipeline.key]),
  );
}

function questionMatchesSearch(question, term) {
  if (!term) {
    return true;
  }

  const fields = [
    question.paper_id,
    question.question_id,
    question.question_type,
    question.question,
    question.gold_answer,
    ...Object.values(question.answers || {}),
  ];

  return fields.some((field) => asText(field).toLowerCase().includes(term));
}

function applyFilters() {
  const term = asText(el.searchInput.value).trim().toLowerCase();
  const paper = el.paperFilter.value;
  const type = el.typeFilter.value;
  const answerFilter = el.answerFilter.value;

  state.filtered = (state.data.questions || [])
    .filter((question) => paper === "all" || String(question.paper_id) === paper)
    .filter((question) => type === "all" || normalizeType(question.question_type) === type)
    .filter((question) => {
      if (answerFilter === "complete") {
        return !rowHasMissingAnswer(question);
      }
      if (answerFilter === "missing") {
        return rowHasMissingAnswer(question);
      }
      return true;
    })
    .filter((question) => questionMatchesSearch(question, term))
    .sort(compareQuestion);
}

function typeClass(type) {
  return `type-${normalizeType(type).toLowerCase()}`;
}

function appendTextBlock(parent, text, className = "answer-text") {
  const p = document.createElement("p");
  p.className = className;
  if (isAnswerMissing(text)) {
    p.classList.add("empty-answer");
    p.textContent = "No answer";
  } else {
    p.textContent = asText(text);
  }
  parent.appendChild(p);
}

function renderQuestion(question) {
  const details = document.createElement("details");
  details.className = "question-card";
  details.open = state.expandedAll;

  const summary = document.createElement("summary");

  const summaryText = document.createElement("div");
  const meta = document.createElement("div");
  meta.className = "question-meta";

  const paper = document.createElement("span");
  paper.className = "tag";
  paper.textContent = `Paper ${question.paper_id}`;
  meta.appendChild(paper);

  const qid = document.createElement("span");
  qid.className = "tag";
  qid.textContent = `Q${question.question_id}`;
  meta.appendChild(qid);

  const type = document.createElement("span");
  type.className = `tag ${typeClass(question.question_type)}`;
  type.textContent = normalizeType(question.question_type).replace("_", " ");
  meta.appendChild(type);

  const title = document.createElement("p");
  title.className = "question-title";
  title.textContent = question.question || "(No question text)";

  summaryText.appendChild(meta);
  summaryText.appendChild(title);

  const chevron = document.createElement("span");
  chevron.className = "chevron";
  chevron.textContent = ">";

  summary.appendChild(summaryText);
  summary.appendChild(chevron);
  details.appendChild(summary);

  const body = document.createElement("div");
  body.className = "question-body";

  const gold = document.createElement("section");
  gold.className = "gold-block";
  const goldTitle = document.createElement("p");
  goldTitle.className = "answer-title";
  goldTitle.textContent = "Gold answer";
  gold.appendChild(goldTitle);
  appendTextBlock(gold, question.gold_answer);
  body.appendChild(gold);

  const answersGrid = document.createElement("div");
  answersGrid.className = "answers-grid";

  (state.data.pipelines || []).forEach((pipeline) => {
    const panel = document.createElement("section");
    panel.className = "answer-panel";

    const head = document.createElement("div");
    head.className = "answer-head";
    head.textContent = pipeline.label;
    panel.appendChild(head);

    const answerBody = document.createElement("div");
    answerBody.className = "answer-body";
    appendTextBlock(answerBody, question.answers?.[pipeline.key]);
    panel.appendChild(answerBody);

    answersGrid.appendChild(panel);
  });

  body.appendChild(answersGrid);
  details.appendChild(body);
  return details;
}

function renderQuestions() {
  clearChildren(el.questionList);
  el.visibleTotal.textContent = String(state.filtered.length);

  if (!state.filtered.length) {
    setMessage("No questions match the current filters.", true);
    return;
  }

  setMessage("", false);
  const fragment = document.createDocumentFragment();
  state.filtered.forEach((question) => fragment.appendChild(renderQuestion(question)));
  el.questionList.appendChild(fragment);
}

function render() {
  applyFilters();
  renderQuestions();
}

function resetFilters() {
  el.searchInput.value = "";
  el.paperFilter.value = "all";
  el.typeFilter.value = "all";
  el.answerFilter.value = "all";
  state.expandedAll = false;
  render();
}

async function loadData() {
  try {
    const response = await fetch(DATA_URL, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    state.data = {
      pipelines: Array.isArray(data.pipelines) ? data.pipelines : [],
      questions: Array.isArray(data.questions) ? data.questions : [],
    };
    populateFilters();
    renderHeader();
    render();
  } catch (error) {
    setMessage(`Could not load ${DATA_URL}: ${error.message}`, true);
  }
}

[el.searchInput, el.paperFilter, el.typeFilter, el.answerFilter].forEach((control) => {
  control.addEventListener("input", render);
});

el.expandAllBtn.addEventListener("click", () => {
  state.expandedAll = true;
  document.querySelectorAll(".question-card").forEach((card) => {
    card.open = true;
  });
});

el.collapseAllBtn.addEventListener("click", () => {
  state.expandedAll = false;
  document.querySelectorAll(".question-card").forEach((card) => {
    card.open = false;
  });
});

el.resetBtn.addEventListener("click", resetFilters);

loadData();
