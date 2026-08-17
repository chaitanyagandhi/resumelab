/* The review UI's front end.
 *
 * No framework and no build step: the page is small, and a bundler would put a
 * toolchain between a one-line change and seeing it. Modules are loaded natively.
 *
 * A run takes a minute or more, so the server hands back a job id and this polls it.
 * That is what makes the wait survivable: a reloaded tab loses the page's state but
 * not the run, which is still going and still spending tokens.
 */

/** The pipeline's stages, in order, with what each one is called on screen. */
const STAGES = [
  ["analysis", "Reading the posting"],
  ["strategy", "Planning the repositioning"],
  ["summary", "Writing the summary"],
  ["experience", "Rewriting experience"],
  ["projects", "Repositioning projects"],
  ["skills", "Selecting skills"],
  ["assembly", "Assembling"],
  ["rendering", "Rendering the PDF"],
];

const POLL_INTERVAL_MS = 800;

const el = (role) => document.querySelector(`[data-role="${role}"]`);

const ui = {
  form: el("generate-form"),
  url: el("url"),
  text: el("text"),
  provider: el("provider"),
  generate: el("generate"),
  stages: el("stages"),
  error: el("error"),
  preview: el("preview"),
  placeholder: el("preview-placeholder"),
  download: el("download"),
  runLabel: el("run-label"),
  statusDot: el("status-dot"),
  statusText: el("status-text"),
};

/** The run currently on screen, if any. */
let currentRunId = null;

// --- talking to the server ------------------------------------------------

/** Ask the server whether it is there, and say so in the status bar. */
export async function reportHealth() {
  try {
    const health = await request("/api/health");
    setStatus("ok", `ResumeLab ${health.version}`);
    return health;
  } catch (error) {
    // A dead server is the likeliest reason this page is open and doing nothing,
    // so it is worth saying plainly rather than failing silently in the console.
    setStatus("lost", `Not connected: ${error.message}`);
    return null;
  }
}

/** Fetch JSON, turning anything the server refused into a readable error. */
async function request(path, options = {}) {
  const response = await fetch(path, options);
  if (response.ok) {
    return response.json();
  }
  throw new Error(await failureMessage(response));
}

/** Prefer the server's own explanation; fall back to the status line. */
async function failureMessage(response) {
  try {
    const body = await response.json();
    return detailText(body.detail) ?? `Request failed (HTTP ${response.status})`;
  } catch {
    return `Request failed (HTTP ${response.status})`;
  }
}

/** A FastAPI detail is a string when we raised it and a list when it validated. */
function detailText(detail) {
  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail) && detail.length > 0) {
    return detail.map((item) => item.msg ?? String(item)).join("; ");
  }
  return null;
}

// --- generating -----------------------------------------------------------

ui.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await generate();
});

async function generate() {
  const posting = readPosting();
  if (posting === null) {
    showError("Give a job posting link, or paste the posting itself.");
    return;
  }

  setBusy(true);
  clearError();
  showStages("analysis");

  try {
    const started = await request("/api/generate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(posting),
    });
    const finished = await pollUntilSettled(started.id);
    if (finished.state === "failed") {
      showError(finished.error ?? "The run failed.");
      return;
    }
    showRun(finished.run_id);
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
}

/** Exactly one source, which is what the server will accept. */
function readPosting() {
  const provider = ui.provider.value || null;
  const text = ui.text.value.trim();
  const url = ui.url.value.trim();

  if (text) {
    return { text, provider };
  }
  if (url) {
    return { url, provider };
  }
  return null;
}

/** Follow a run to completion, showing where it has got to. */
async function pollUntilSettled(jobId) {
  for (;;) {
    const job = await request(`/api/jobs/${encodeURIComponent(jobId)}`);
    showStages(job.stage);
    if (job.state !== "running") {
      return job;
    }
    await sleep(POLL_INTERVAL_MS);
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// --- showing the result ---------------------------------------------------

function showRun(runId) {
  currentRunId = runId;
  ui.preview.src = `/api/runs/${encodeURIComponent(runId)}/resume.pdf`;
  ui.preview.hidden = false;
  ui.placeholder.hidden = true;
  ui.download.href = ui.preview.src;
  ui.download.hidden = false;
  ui.runLabel.textContent = runId;
}

function showStages(current) {
  const reached = STAGES.findIndex(([name]) => name === current);
  ui.stages.hidden = false;
  ui.stages.replaceChildren(
    ...STAGES.map(([name, label], index) => {
      const item = document.createElement("li");
      item.className = "stages__item";
      item.textContent = label;
      item.dataset.state = stageState(index, reached);
      item.dataset.stage = name;
      return item;
    }),
  );
}

/** Before the first stage is reported, nothing is done and nothing is current. */
function stageState(index, reached) {
  if (index < reached) {
    return "done";
  }
  return index === reached ? "current" : "pending";
}

function setBusy(busy) {
  ui.generate.disabled = busy;
  ui.generate.textContent = busy ? "Generating…" : "Generate resume";
}

function showError(message) {
  ui.error.textContent = message;
  ui.error.hidden = false;
}

function clearError() {
  ui.error.textContent = "";
  ui.error.hidden = true;
}

function setStatus(state, text) {
  ui.statusDot.dataset.state = state;
  ui.statusText.textContent = text;
}

/** Exposed for the console, and so a reader can see what the page holds. */
export const inspect = () => ({ currentRunId });

reportHealth();
