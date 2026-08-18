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
  document: el("document"),
  views: el("views"),
  saveState: el("save-state"),
  download: el("download"),
  runLabel: el("run-label"),
  statusDot: el("status-dot"),
  statusText: el("status-text"),
  theme: el("theme"),
};

// --- theme ----------------------------------------------------------------
//
// Three states, not two: light, dark, and no opinion. Until the toggle is used the
// page follows the system, which is what someone who set a system preference
// already asked for. Pressing it commits to one and remembers that.

const THEME_KEY = "resumelab.theme";

function currentTheme() {
  return (
    document.documentElement.dataset.theme ??
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")
  );
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const dark = theme === "dark";
  ui.theme.textContent = dark ? "☀" : "☾";
  ui.theme.setAttribute("aria-label", `Switch to the ${dark ? "light" : "dark"} theme`);
  ui.theme.title = ui.theme.getAttribute("aria-label");
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Storage refused; the theme still holds for this page.
  }
}

ui.theme.addEventListener("click", () => {
  applyTheme(currentTheme() === "dark" ? "light" : "dark");
});

// The button has to say what it will do before it is ever pressed, and that depends
// on the system preference when nothing was stored.
ui.theme.textContent = currentTheme() === "dark" ? "☀" : "☾";
ui.theme.title = `Switch to the ${currentTheme() === "dark" ? "light" : "dark"} theme`;
ui.theme.setAttribute("aria-label", ui.theme.title);

// Follow the system while it is still the thing being followed.
window
  .matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", (event) => {
    if (document.documentElement.dataset.theme === undefined) {
      ui.theme.textContent = event.matches ? "☀" : "☾";
    }
  });

const state = {
  runId: null,
  /** The resume being edited, as the server will take it back. */
  resume: null,
  /** Whether this run has an edit saved, which is what the preview should show. */
  hasEdit: false,
  /** "preview", "fields", or "document". Null until a run is open. */
  view: null,
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
    showStages(job.stage, job.state);
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
  state.view = null;
  ui.runLabel.textContent = runId;
  ui.views.hidden = false;
  ui.saveState.hidden = true;
  // In the address bar so a reload reopens the run rather than losing it. The run
  // survives this page; without it here, the page's only reference to it would not.
  location.hash = `run=${encodeURIComponent(runId)}`;
  setView("preview");
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

/** Load the resume being edited, once. Both editors work on the same object. */
async function ensureResume() {
  if (state.resume !== null) {
    return true;
  }
  try {
    state.resume = await request(
      `/api/runs/${encodeURIComponent(state.runId)}/${state.hasEdit ? "edit" : "resume"}`,
    );
    return true;
  } catch (error) {
    showError(error.message);
    return false;
  }
}

/**
 * Show one of the three views.
 *
 * The two editors are alternative ways into the same resume object, not two
 * documents: switching between them carries the edits across untouched, because
 * there is only ever one thing being edited.
 */
async function setView(view) {
  if (state.view !== null && state.view !== "preview") {
    // Leaving an editor with a keystroke still in the debounce would show a PDF one
    // edit behind the screen, and would rebuild the other view from stale content.
    await flushSave();
  }
  if (view !== "preview" && !(await ensureResume())) {
    return;
  }

  state.view = view;
  ui.placeholder.hidden = true;
  ui.preview.hidden = view !== "preview";
  ui.editor.hidden = view !== "fields";
  ui.document.hidden = view !== "document";

  if (view === "preview") {
    const url = pdfPath();
    ui.preview.src = url;
    ui.download.href = url;
    ui.download.hidden = false;
  } else if (view === "fields") {
    buildEditor();
  } else {
    buildDocument();
  }

  for (const button of ui.views.querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.view === view));
  }
}

ui.views.addEventListener("click", async (event) => {
  const view = event.target.dataset?.view;
  if (view !== undefined && view !== state.view) {
    await setView(view);
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
        entry(
          project.name || `Project ${index + 1}`,
          [
            grid(PROJECT_FIELDS.map(([label, key]) => field(label, `projects.${index}.${key}`))),
            field("Technologies", `projects.${index}.technologies`, { list: true }),
            bullets(`projects.${index}.bullets`, project.bullets),
          ],
          // Which project leads is a real editorial choice, and the only way to make
          // it today is to retype three of them into each other's fields.
          { reorder: { path: "projects", index, total: resume.projects.length } },
        ),
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

/** One entry card. Given ``reorder``, it can be dragged within its own list. */
function entry(title, children, { reorder = null } = {}) {
  const node = document.createElement("div");
  node.className = "ed-entry";

  const header = document.createElement("div");
  header.className = "ed-entry__header";
  const heading = document.createElement("p");
  heading.className = "ed-entry__title";
  heading.textContent = title;
  header.append(heading);

  if (reorder !== null) {
    header.prepend(dragHandle(node, reorder));
    acceptDrops(node, reorder);
  }

  node.append(header, ...children.flat());
  return node;
}

/** The entry being dragged, or null. Only one can be in flight at a time. */
let dragging = null;

function dragHandle(card, spec) {
  const handle = document.createElement("button");
  handle.type = "button";
  handle.className = "ed-handle";
  handle.textContent = "⠿";
  handle.dataset.entry = `${spec.path}.${spec.index}`;
  handle.title = "Drag to reorder, or focus and use the arrow keys";
  handle.setAttribute("aria-label", `Reorder, currently ${spec.index + 1} of ${spec.total}`);

  // The card becomes draggable only while the handle is held. Marking it draggable
  // outright would take dragging away from the text inside its own fields.
  handle.addEventListener("pointerdown", () => {
    card.draggable = true;
  });
  handle.addEventListener("keydown", (event) => {
    const step = { ArrowUp: -1, ArrowDown: 1 }[event.key];
    if (step === undefined) {
      return;
    }
    event.preventDefault();
    moveEntry(spec.path, spec.index, spec.index + step);
  });

  card.addEventListener("dragstart", (event) => {
    dragging = spec;
    card.classList.add("is-dragging");
    event.dataTransfer.effectAllowed = "move";
    // Firefox starts no drag at all unless the transfer carries something.
    event.dataTransfer.setData("text/plain", String(spec.index));
  });
  card.addEventListener("dragend", () => {
    card.draggable = false;
    dragging = null;
    card.classList.remove("is-dragging");
    clearDropMarks();
  });

  return handle;
}

function acceptDrops(card, spec) {
  card.addEventListener("dragover", (event) => {
    if (dragging === null || dragging.path !== spec.path) {
      return;
    }
    // Without this the browser refuses the drop, silently.
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    card.dataset.drop = pastHalfway(event, card) ? "after" : "before";
  });

  card.addEventListener("dragleave", () => {
    delete card.dataset.drop;
  });

  card.addEventListener("drop", (event) => {
    if (dragging === null || dragging.path !== spec.path) {
      return;
    }
    event.preventDefault();
    const from = dragging.index;
    dragging = null;
    // An insertion point, not a destination: dropping below the third card means
    // "fourth position", which is only index 3 once the dragged card is gone.
    const insertAt = spec.index + (pastHalfway(event, card) ? 1 : 0);
    moveEntry(spec.path, from, insertAt > from ? insertAt - 1 : insertAt);
  });
}

function pastHalfway(event, card) {
  const box = card.getBoundingClientRect();
  return event.clientY > box.top + box.height / 2;
}

/** Move an entry within its list, then redraw and save. */
function moveEntry(path, from, to) {
  const items = readPath(state.resume, path);
  if (to < 0 || to >= items.length || to === from) {
    return;
  }
  const [moved] = items.splice(from, 1);
  items.splice(to, 0, moved);
  rebuildAndSave();
  // The editor was rebuilt, so the handle that was focused no longer exists.
  // Following the entry is what makes reordering by keyboard usable at all.
  ui.editor.querySelector(`.ed-handle[data-entry="${path}.${to}"]`)?.focus();
}

function clearDropMarks() {
  for (const node of ui.editor.querySelectorAll("[data-drop]")) {
    delete node.dataset.drop;
  }
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

/** Adding or removing changes every path after it, so the view is rebuilt. */
function rebuildAndSave() {
  if (state.view === "document") {
    buildDocument();
  } else {
    buildEditor();
  }
  scheduleSave();
}

/**
 * Split a list field.
 *
 * Either separator is accepted, because the two editors show these differently: the
 * fields editor lists them by comma, the document draws them the way the page does.
 * Blanks are dropped rather than saved as empty entries.
 */
function asList(raw) {
  return raw
    .split(/[,·]/)
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

// --- the document ---------------------------------------------------------
//
// A facsimile of the page, laid out to look like the resume and editable in place.
// Every editable run is bound to a field of the same resume the other editor works
// on, so what leaves here is structured content and never markup. The PDF is still
// drawn by the renderer; this is a way of typing, not a second document.

function buildDocument() {
  const resume = state.resume;
  const page = document.createElement("article");
  page.className = "doc__page";

  page.append(
    docHeader(resume),
    docSection("Summary", [bound("summary", { tag: "p", className: "doc__body" })]),
    docSection(
      "Education",
      resume.education.flatMap((_entry, index) => docEducation(index)),
    ),
    docSection(
      "Experience",
      resume.experiences.flatMap((role, index) => docExperience(role, index)),
    ),
    docSection(
      "Projects",
      resume.projects.flatMap((project, index) => docProject(project, index)),
    ),
    docSection("Skills", [
      bound("skills", { tag: "p", className: "doc__body", list: true, separator: " · " }),
    ]),
  );
  if (resume.achievements.length > 0) {
    page.append(docSection("Achievements", [docBullets("achievements", resume.achievements)]));
  }

  // The caveat is the panel's own, not the page's, so it survives every rebuild.
  ui.document.replaceChildren(ui.document.querySelector(".doc__caveat"), page);
}

/** An editable run bound to one field of the resume. */
function bound(path, { tag = "span", className = "", list = false, separator = ", ", hint = "" } = {}) {
  const node = document.createElement(tag);
  node.className = `doc__edit ${className}`.trim();
  node.contentEditable = "true";
  node.dataset.path = path;
  node.dataset.list = String(list);
  node.dataset.separator = separator;
  if (hint) {
    node.dataset.hint = hint;
  }

  const value = readPath(state.resume, path);
  node.textContent = list ? (value ?? []).join(separator) : (value ?? "");

  node.addEventListener("input", onDocumentInput);
  node.addEventListener("paste", onDocumentPaste);
  node.addEventListener("keydown", onDocumentKeydown);
  return node;
}

/** Text the page draws but the resume does not hold, such as a label or a comma. */
function fixed(text, className = "") {
  const node = document.createElement("span");
  node.className = `doc__fixed ${className}`.trim();
  node.textContent = text;
  return node;
}

function onDocumentInput(event) {
  const node = event.currentTarget;
  // textContent, never innerHTML: whatever the browser did to the markup, the field
  // this is bound to is a string, and that is all that is read back out.
  const raw = node.textContent;
  writePath(state.resume, node.dataset.path, node.dataset.list === "true" ? asList(raw) : raw);
  scheduleSave();
}

/** Paste as plain text. A pasted heading would otherwise arrive with its markup. */
function onDocumentPaste(event) {
  event.preventDefault();
  const text = event.clipboardData.getData("text/plain").replace(/\s+/g, " ").trim();
  // Deprecated, and still the only insertion that the browser's own undo can see.
  document.execCommand("insertText", false, text);
}

function onDocumentKeydown(event) {
  const node = event.currentTarget;
  const bullet = /^(.*\.bullets|achievements)\.(\d+)$/.exec(node.dataset.path);

  if (event.key === "Enter") {
    // Never let contenteditable invent markup: a field holds a line, not a document.
    event.preventDefault();
    if (bullet !== null) {
      const [, listPath, index] = bullet;
      const at = Number(index) + 1;
      readPath(state.resume, listPath).splice(at, 0, "");
      rebuildAndSave();
      focusBound(`${listPath}.${at}`);
    }
    return;
  }

  // Backspace at the start of an empty bullet removes it, the way a list behaves in
  // any editor. Only when empty, so it can never eat text that was typed.
  if (event.key === "Backspace" && bullet !== null && node.textContent === "") {
    event.preventDefault();
    const [, listPath, index] = bullet;
    const items = readPath(state.resume, listPath);
    if (items.length > 1) {
      items.splice(Number(index), 1);
      rebuildAndSave();
      focusBound(`${listPath}.${Math.max(Number(index) - 1, 0)}`);
    }
  }
}

/** Put the caret at the end of a field, after the view was rebuilt under it. */
function focusBound(path) {
  const node = ui.document.querySelector(`[data-path="${path}"]`);
  if (node === null) {
    return;
  }
  node.focus();
  const range = document.createRange();
  range.selectNodeContents(node);
  range.collapse(false);
  const selection = window.getSelection();
  selection.removeAllRanges();
  selection.addRange(range);
}

function docHeader(resume) {
  const header = document.createElement("header");
  header.className = "doc__headerblock";

  const contact = document.createElement("p");
  contact.className = "doc__contact";
  const fields = [
    ["personal.location", "Location"],
    ["personal.email", "Email"],
    ["personal.phone", "Phone"],
    ["personal.linkedin", "LinkedIn"],
    ["personal.github", "GitHub"],
  ];
  fields.forEach(([path, hint], index) => {
    if (index > 0) {
      contact.append(fixed(" · "));
    }
    contact.append(bound(path, { hint }));
  });

  header.append(bound("personal.name", { tag: "h1", className: "doc__name" }), contact);
  return header;
}

function docSection(title, children) {
  const section = document.createElement("section");
  section.className = "doc__section";
  const heading = document.createElement("h2");
  heading.className = "doc__heading";
  heading.textContent = title.toUpperCase();
  section.append(heading, ...children);
  return section;
}

/** A line with content on the left and a date or a figure on the right. */
function docRow(left, right, className = "") {
  const row = document.createElement("div");
  row.className = `doc__row ${className}`.trim();
  const leading = document.createElement("div");
  leading.className = "doc__rowleft";
  leading.append(...left);
  row.append(leading, right);
  return row;
}

/** Two fields, drawn as the one range the page shows. */
function docDates(prefix) {
  const range = document.createElement("div");
  range.className = "doc__right";
  range.append(
    bound(`${prefix}.start_date`, { hint: "Start" }),
    fixed(" – "),
    bound(`${prefix}.end_date`, { hint: "End" }),
  );
  return range;
}

function docEducation(index) {
  const at = `education.${index}`;
  const gpa = document.createElement("div");
  gpa.className = "doc__right doc__detail";
  gpa.append(fixed("GPA: "), bound(`${at}.gpa`, { hint: "GPA" }));

  return [
    docRow(
      [
        bound(`${at}.institution`, { className: "doc__strong" }),
        fixed(" · "),
        bound(`${at}.location`, { hint: "Location" }),
      ],
      docDates(at),
    ),
    docRow(
      [
        bound(`${at}.degree`, { className: "doc__em" }),
        fixed(" "),
        bound(`${at}.field`, { className: "doc__em", hint: "Field" }),
      ],
      gpa,
      "doc__detail",
    ),
  ];
}

function docExperience(role, index) {
  const at = `experiences.${index}`;
  return [
    docRow(
      [
        bound(`${at}.title`, { className: "doc__strong" }),
        fixed(", "),
        bound(`${at}.company`, { className: "doc__em" }),
      ],
      docDates(at),
    ),
    docBullets(`${at}.bullets`, role.bullets),
  ];
}

function docProject(project, index) {
  const at = `projects.${index}`;
  const date = document.createElement("div");
  date.className = "doc__right";
  date.append(bound(`${at}.date`, { hint: "Date" }));

  return [
    docRow(
      [
        bound(`${at}.name`, { className: "doc__strong" }),
        fixed(" - "),
        bound(`${at}.subtitle`, { hint: "Subtitle" }),
        fixed(" / "),
        bound(`${at}.technologies`, { className: "doc__em", list: true, hint: "Technologies" }),
      ],
      date,
    ),
    docBullets(`${at}.bullets`, project.bullets),
  ];
}

function docBullets(path, values) {
  const list = document.createElement("ul");
  list.className = "doc__bullets";
  values.forEach((_value, index) =>
    list.append(bound(`${path}.${index}`, { tag: "li", className: "doc__bulletline" })),
  );
  return list;
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

function showStages(current, runState = "running") {
  const reached = STAGES.findIndex(([name]) => name === current);
  ui.stages.hidden = false;
  ui.stages.replaceChildren(
    ...STAGES.map(([name, label], index) => {
      const item = document.createElement("li");
      item.className = "stages__item";
      item.textContent = label;
      item.dataset.state = stageState(index, reached, runState);
      item.dataset.stage = name;
      return item;
    }),
  );
}

/**
 * How one stage should read, given how far the run got and how it ended.
 *
 * A finished run has no current stage. The last one reported is where the run was
 * when it stopped, which for a completed run means it finished there rather than
 * that it is still going: leaving it marked current is a spinner that never stops.
 */
function stageState(index, reached, runState) {
  if (runState === "completed") {
    return "done";
  }
  if (index < reached) {
    return "done";
  }
  if (index !== reached) {
    return "pending";
  }
  return runState === "failed" ? "failed" : "current";
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
