/* The review UI's front end.
 *
 * No framework and no build step: the page is small, and a bundler would put a
 * toolchain between a one-line change and seeing it. Modules are loaded natively.
 *
 * A run takes a minute or more, so the server hands back a job id and this polls it.
 * That is what makes the wait survivable: a reloaded tab loses the page's state but
 * not the run, which is still going and still spending tokens.
 *
 * The editor edits structure, not the page. Every field here is a field of the
 * resume the server holds, and the PDF is redrawn by the same renderer a run uses.
 * Editing a rendering of the document instead would mean two layout engines - the
 * one you type into and the one you download - and they would drift.
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

/** Long enough to be a pause in typing, short enough to feel like it followed it. */
const SAVE_DEBOUNCE_MS = 700;

const IDENTITY_FIELDS = [
  ["Name", "name"],
  ["Email", "email"],
  ["Phone", "phone"],
  ["Location", "location"],
  ["LinkedIn", "linkedin"],
  ["GitHub", "github"],
];

const EDUCATION_FIELDS = [
  ["Institution", "institution"],
  ["Degree", "degree"],
  ["Field", "field"],
  ["Location", "location"],
  ["Start", "start_date"],
  ["End", "end_date"],
  ["GPA", "gpa"],
];

const EXPERIENCE_FIELDS = [
  ["Title", "title"],
  ["Company", "company"],
  ["Location", "location"],
  ["Start", "start_date"],
  ["End", "end_date"],
];

const PROJECT_FIELDS = [
  ["Name", "name"],
  ["Subtitle", "subtitle"],
  ["Date", "date"],
];

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
  editor: el("editor"),
  mode: el("mode"),
  saveState: el("save-state"),
  download: el("download"),
  runLabel: el("run-label"),
  statusDot: el("status-dot"),
  statusText: el("status-text"),
};

const state = {
  runId: null,
  /** The resume being edited, as the server will take it back. */
  resume: null,
  /** Whether this run has an edit saved, which is what the preview should show. */
  hasEdit: false,
  mode: "preview",
};

let saveTimer = null;
let pendingSave = null;

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

function sendJSON(method, body) {
  return { method, headers: { "content-type": "application/json" }, body: JSON.stringify(body) };
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
    return detail.map(validationText).join("; ");
  }
  return null;
}

/** Name the field as well as the complaint, so a rejected edit can be found. */
function validationText(item) {
  if (typeof item !== "object" || item === null) {
    return String(item);
  }
  const where = Array.isArray(item.loc) ? item.loc.slice(1).join(".") : "";
  return where ? `${where}: ${item.msg}` : (item.msg ?? String(item));
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
    const started = await request("/api/generate", sendJSON("POST", posting));
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
  state.runId = runId;
  state.resume = null;
  state.hasEdit = false;
  ui.runLabel.textContent = runId;
  ui.mode.hidden = false;
  ui.saveState.hidden = true;
  // In the address bar so a reload reopens the run rather than losing it. The run
  // survives this page; without it here, the page's only reference to it would not.
  location.hash = `run=${encodeURIComponent(runId)}`;
  showPreview();
}

/** Reopen whatever run the address bar names. */
function restoreFromHash() {
  const match = /^#run=(.+)$/.exec(location.hash);
  if (match !== null) {
    showRun(decodeURIComponent(match[1]));
  }
}

/** The edited PDF once there is one, because that is what is on screen. */
function pdfPath() {
  const run = encodeURIComponent(state.runId);
  const name = state.hasEdit ? "edit.pdf" : "resume.pdf";
  // The path does not change when an edit is saved, so without this the browser
  // would keep showing the PDF it already has.
  return `/api/runs/${run}/${name}?v=${Date.now()}`;
}

function showPreview() {
  state.mode = "preview";
  const url = pdfPath();
  ui.preview.src = url;
  ui.download.href = url;
  ui.preview.hidden = false;
  ui.download.hidden = false;
  ui.editor.hidden = true;
  ui.placeholder.hidden = true;
  ui.mode.textContent = "Edit like a doc";
}

async function showEditor() {
  try {
    if (state.resume === null) {
      state.resume = await request(
        `/api/runs/${encodeURIComponent(state.runId)}/${state.hasEdit ? "edit" : "resume"}`,
      );
    }
  } catch (error) {
    showError(error.message);
    return;
  }

  state.mode = "edit";
  buildEditor();
  ui.editor.hidden = false;
  ui.preview.hidden = true;
  ui.placeholder.hidden = true;
  ui.mode.textContent = "Back to preview";
}

ui.mode.addEventListener("click", async () => {
  if (state.mode === "edit") {
    // Leaving the editor with a keystroke still in the debounce would show a PDF
    // that is one edit behind what is on screen.
    await flushSave();
    showPreview();
  } else {
    await showEditor();
  }
});

// --- the editor -----------------------------------------------------------

function buildEditor() {
  const resume = state.resume;
  ui.editor.replaceChildren(
    section("Identity", [grid(IDENTITY_FIELDS.map(([label, key]) => field(label, `personal.${key}`)))]),
    section("Summary", [field("Summary", "summary", { multiline: true })]),
    section(
      "Education",
      resume.education.map((_entry, index) =>
        entry(`Entry ${index + 1}`, [
          grid(EDUCATION_FIELDS.map(([label, key]) => field(label, `education.${index}.${key}`))),
          field("Coursework", `education.${index}.coursework`, { list: true }),
        ]),
      ),
    ),
    section(
      "Experience",
      resume.experiences.map((role, index) =>
        entry(role.title || `Role ${index + 1}`, [
          grid(EXPERIENCE_FIELDS.map(([label, key]) => field(label, `experiences.${index}.${key}`))),
          bullets(`experiences.${index}.bullets`, role.bullets),
        ]),
      ),
    ),
    section(
      "Projects",
      resume.projects.map((project, index) =>
        entry(project.name || `Project ${index + 1}`, [
          grid(PROJECT_FIELDS.map(([label, key]) => field(label, `projects.${index}.${key}`))),
          field("Technologies", `projects.${index}.technologies`, { list: true }),
          bullets(`projects.${index}.bullets`, project.bullets),
        ]),
      ),
    ),
    section("Skills", [
      field("Skills, in the order they are shown", "skills", { list: true, multiline: true }),
    ]),
    section("Achievements", [bullets("achievements", resume.achievements)]),
  );
}

function section(title, children) {
  const node = document.createElement("section");
  node.className = "ed-section";
  const heading = document.createElement("h3");
  heading.className = "ed-section__title";
  heading.textContent = title;
  node.append(heading, ...children.flat());
  return node;
}

function entry(title, children) {
  const node = document.createElement("div");
  node.className = "ed-entry";
  const heading = document.createElement("p");
  heading.className = "ed-entry__title";
  heading.textContent = title;
  node.append(heading, ...children.flat());
  return node;
}

function grid(children) {
  const node = document.createElement("div");
  node.className = "ed-grid";
  node.append(...children);
  return node;
}

/** One bound field. ``list`` fields are comma separated in and out. */
function field(label, path, { multiline = false, list = false } = {}) {
  const wrapper = document.createElement("label");
  wrapper.className = "ed-field";

  const caption = document.createElement("span");
  caption.textContent = label;

  const input = document.createElement(multiline ? "textarea" : "input");
  if (multiline) {
    input.rows = list ? 3 : 4;
  }
  const value = readPath(state.resume, path);
  input.value = list ? (value ?? []).join(", ") : (value ?? "");
  input.dataset.path = path;
  input.dataset.list = String(list);
  input.addEventListener("input", onFieldInput);

  wrapper.append(caption, input);
  return wrapper;
}

/** A list of bullets, each removable, with a way to add one. */
function bullets(path, values) {
  const node = document.createElement("div");
  node.className = "ed-bullets";

  values.forEach((_value, index) => {
    const row = document.createElement("div");
    row.className = "ed-bullet";

    const input = document.createElement("textarea");
    input.rows = 2;
    input.value = values[index];
    input.dataset.path = `${path}.${index}`;
    input.dataset.list = "false";
    input.addEventListener("input", onFieldInput);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "ed-remove";
    remove.title = "Remove this bullet";
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      readPath(state.resume, path).splice(index, 1);
      rebuildAndSave();
    });

    row.append(input, remove);
    node.append(row);
  });

  const add = document.createElement("button");
  add.type = "button";
  add.className = "ed-add";
  add.textContent = "Add a bullet";
  add.addEventListener("click", () => {
    readPath(state.resume, path).push("");
    rebuildAndSave();
  });
  node.append(add);
  return node;
}

function onFieldInput(event) {
  const input = event.currentTarget;
  const raw = input.value;
  writePath(state.resume, input.dataset.path, input.dataset.list === "true" ? asList(raw) : raw);
  scheduleSave();
}

/** Adding or removing changes every path after it, so the editor is rebuilt. */
function rebuildAndSave() {
  buildEditor();
  scheduleSave();
}

/** Commas separate; blanks are dropped rather than saved as empty entries. */
function asList(raw) {
  return raw
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

function readPath(root, path) {
  return path.split(".").reduce((node, part) => node[part], root);
}

function writePath(root, path, value) {
  const parts = path.split(".");
  const parent = parts.slice(0, -1).reduce((node, part) => node[part], root);
  parent[parts.at(-1)] = value;
}

// --- saving ---------------------------------------------------------------

function scheduleSave() {
  clearTimeout(saveTimer);
  setSaveState("pending", "Editing…");
  saveTimer = setTimeout(() => {
    pendingSave = save();
  }, SAVE_DEBOUNCE_MS);
}

/** Save now if anything is waiting, and wait for whatever is already in flight. */
async function flushSave() {
  if (saveTimer !== null) {
    clearTimeout(saveTimer);
    saveTimer = null;
    pendingSave = save();
  }
  await pendingSave;
}

async function save() {
  setSaveState("saving", "Saving…");
  try {
    const outcome = await request(
      `/api/runs/${encodeURIComponent(state.runId)}/edit`,
      sendJSON("PUT", { resume: state.resume }),
    );
    state.hasEdit = true;
    // Reported, not corrected. Someone editing their own resume is allowed a second
    // page; they are not allowed to find out about it at the printer.
    const pages = `${outcome.page_count} page${outcome.page_count === 1 ? "" : "s"}`;
    setSaveState(outcome.fits_on_one_page ? "ok" : "warn", `Saved · ${pages}`);
  } catch (error) {
    setSaveState("bad", error.message);
  } finally {
    saveTimer = null;
  }
}

// --- small bits of chrome -------------------------------------------------

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

function setSaveState(state_, text) {
  ui.saveState.dataset.state = state_;
  ui.saveState.textContent = text;
  ui.saveState.hidden = false;
}

function showError(message) {
  ui.error.textContent = message;
  ui.error.hidden = false;
}

function clearError() {
  ui.error.textContent = "";
  ui.error.hidden = true;
}

function setStatus(state_, text) {
  ui.statusDot.dataset.state = state_;
  ui.statusText.textContent = text;
}

/** Exposed for the console, and so a reader can see what the page holds. */
export const inspect = () => ({ ...state });

reportHealth();
restoreFromHash();
