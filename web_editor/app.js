import { EditorApi } from "./api.js";
import { EditorStore } from "./store.js";
import { EditorCanvas } from "./canvas.js";
import { PreviewController } from "./preview.js";
import { STAGES, SCREEN, ALIGNMENT_LABELS, recordLabel, fieldsFor, applyFields, textFieldTarget, updateTextField } from "./screens.js";

const config = window.TETOLATE_EDITOR_V2;
const AVAILABLE_STAGES = Array.isArray(config.availableStages)
  ? STAGES.filter((stage) => config.availableStages.includes(stage))
  : STAGES;
const api = new EditorApi(config.category, config.jobId);
const store = new EditorStore(config);
const byId = (id) => document.getElementById(id);
const canvas = new EditorCanvas(byId("page-canvas"), store, setStatus, (page, records, signal) => api.textLayers(page, records, signal),
  {select: selectOcrBoxes, has: ocrBoxesChecked});
const preview = new PreviewController(store, canvas, api, setStatus);
const mergeSelection = new Set();
let loading = false;
let loadRequest = 0;
let loadController = null;
let currentPreviewRequest = null;
let navigationPage = null;
let navigationRequest = 0;
let savePromise = null;
let resetting = false;
let ocrAreaRunning = false;
let queuedAction = "";
let tooltipAnchor = null;
let tooltipTimer = null;
let lineBreaksTimer = null;
let lineBreaksController = null;
const typesettingViewKey = `tetolate-editor-view:${config.category}:${config.jobId}`;
const TYPESETTING_VIEWS = new Set(["current", "original", "preview", "render"]);
let typesettingView = localStorage.getItem(typesettingViewKey) || "current";
let hideEditorBoxes = false;
if (!TYPESETTING_VIEWS.has(typesettingView)) typesettingView = "current";
const paneWidthKey = `tetolate-editor-panes:${config.category}:${config.jobId}`;
const paneWidths = { record: 288, inspector: 352 };
try {
  const savedPaneWidths = JSON.parse(localStorage.getItem(paneWidthKey) || "null");
  if (Number.isFinite(savedPaneWidths?.record)) paneWidths.record = savedPaneWidths.record;
  if (Number.isFinite(savedPaneWidths?.inspector)) paneWidths.inspector = savedPaneWidths.inspector;
} catch { localStorage.removeItem(paneWidthKey); }

function setStatus(message, error = false) {
  const target = byId("editor-status");
  target.textContent = message || "";
  target.classList.toggle("error", error);
}

function setOcrAreaRunning(running) {
  ocrAreaRunning = !!running;
  canvas.setBusy(ocrAreaRunning);
  byId("page-select").disabled = ocrAreaRunning;
  byId("prev-page").disabled = ocrAreaRunning;
  byId("next-page").disabled = ocrAreaRunning;
  renderStageTabs();
  renderTools();
}

function setQueuedAction(action, message = "") {
  queuedAction = action;
  const busy = !!action;
  for (const [key, label] of [["stage", "Regenerate stage"], ["page", "Regenerate page"],
    ["all-pages", "Regenerate all pages"], ["finish-edits", "Finish Edits"]]) {
    const button = byId(key === "finish-edits" ? key : `regenerate-${key}`);
    button.disabled = busy;
    button.setAttribute("aria-busy", String(busy));
    button.textContent = action === key ? "Queueing..." : label;
  }
  renderTools();
  if (message) setStatus(message);
}

function hideTooltip() {
  clearTimeout(tooltipTimer);
  tooltipTimer = null;
  if (tooltipAnchor) tooltipAnchor.removeAttribute("aria-describedby");
  tooltipAnchor = null;
  byId("editor-tooltip").hidden = true;
}

function scheduleTooltipHide() {
  clearTimeout(tooltipTimer);
  tooltipTimer = setTimeout(hideTooltip, 180);
}

function showTooltip(anchor) {
  const message = anchor?.dataset.tooltip?.trim();
  if (!message) return;
  clearTimeout(tooltipTimer);
  if (tooltipAnchor && tooltipAnchor !== anchor) tooltipAnchor.removeAttribute("aria-describedby");
  tooltipAnchor = anchor;
  anchor.setAttribute("aria-describedby", "editor-tooltip");

  const tooltip = byId("editor-tooltip");
  tooltip.textContent = message;
  tooltip.hidden = false;
  tooltip.style.visibility = "hidden";
  const anchorBounds = anchor.getBoundingClientRect();
  const tooltipBounds = tooltip.getBoundingClientRect();
  const margin = 8;
  const left = Math.max(
    margin,
    Math.min(anchorBounds.left, window.innerWidth - tooltipBounds.width - margin),
  );
  let top = anchorBounds.bottom + margin;
  if (top + tooltipBounds.height > window.innerHeight - margin) {
    top = Math.max(margin, anchorBounds.top - tooltipBounds.height - margin);
  }
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
  tooltip.style.visibility = "visible";
}

document.addEventListener("pointerover", (event) => {
  const anchor = event.target.closest?.("[data-tooltip]");
  if (anchor) showTooltip(anchor);
});
document.addEventListener("pointerout", (event) => {
  const anchor = event.target.closest?.("[data-tooltip]");
  if (anchor && !anchor.contains(event.relatedTarget)) scheduleTooltipHide();
});
document.addEventListener("focusin", (event) => {
  const anchor = event.target.closest?.("[data-tooltip]");
  if (anchor) showTooltip(anchor);
});
document.addEventListener("focusout", (event) => {
  const anchor = event.target.closest?.("[data-tooltip]");
  if (anchor) scheduleTooltipHide();
});
byId("editor-tooltip").addEventListener("pointerenter", () => clearTimeout(tooltipTimer));
byId("editor-tooltip").addEventListener("pointerleave", scheduleTooltipHide);
byId("editor-tooltip").addEventListener("focusin", () => clearTimeout(tooltipTimer));
byId("editor-tooltip").addEventListener("focusout", scheduleTooltipHide);
window.addEventListener("resize", hideTooltip);

function clampPaneWidth(value, pane) {
  const workspace = byId("editor-app").querySelector(".editor-workspace");
  const workspaceWidth = workspace.clientWidth || window.innerWidth;
  const compact = window.matchMedia("(max-width: 1050px)").matches;
  const otherWidth = compact
    ? 0
    : (pane === "record" ? paneWidths.inspector : paneWidths.record);
  const minimumCanvasWidth = compact ? 320 : 384;
  const maximum = Math.max(
    180,
    Math.min(workspaceWidth * 0.42, workspaceWidth - otherWidth - minimumCanvasWidth - 16),
  );
  return Math.round(Math.max(180, Math.min(maximum, value)));
}

function applyPaneWidths() {
  paneWidths.record = clampPaneWidth(paneWidths.record, "record");
  paneWidths.inspector = clampPaneWidth(paneWidths.inspector, "inspector");
  paneWidths.record = clampPaneWidth(paneWidths.record, "record");
  const workspace = byId("editor-app").querySelector(".editor-workspace");
  workspace.style.setProperty("--record-pane-width", `${paneWidths.record}px`);
  workspace.style.setProperty("--inspector-pane-width", `${paneWidths.inspector}px`);
  byId("record-resizer").setAttribute("aria-valuenow", String(paneWidths.record));
  byId("inspector-resizer").setAttribute("aria-valuenow", String(paneWidths.inspector));
}

function savePaneWidths() {
  localStorage.setItem(paneWidthKey, JSON.stringify(paneWidths));
}

function initPaneResizer(id, pane, direction) {
  const handle = byId(id);
  let drag = null;
  handle.addEventListener("pointerdown", (event) => {
    drag = { x: event.clientX, width: paneWidths[pane] };
    handle.classList.add("dragging");
    handle.setPointerCapture?.(event.pointerId);
  });
  handle.addEventListener("pointermove", (event) => {
    if (!drag) return;
    paneWidths[pane] = clampPaneWidth(
      drag.width + (event.clientX - drag.x) * direction,
      pane,
    );
    applyPaneWidths();
  });
  const finish = () => {
    if (!drag) return;
    drag = null;
    handle.classList.remove("dragging");
    savePaneWidths();
  };
  handle.addEventListener("pointerup", finish);
  handle.addEventListener("pointercancel", finish);
  handle.addEventListener("keydown", (event) => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    const movement = event.key === "ArrowRight" ? 16 : -16;
    paneWidths[pane] = clampPaneWidth(paneWidths[pane] + movement * direction, pane);
    applyPaneWidths();
    savePaneWidths();
  });
}

function initEditorHeightResizer() {
  const handle = byId("height-resizer");
  let extraHeight = 0, drag = null, frame = null;
  const resizeBy = (delta) => {
    const previous = extraHeight;
    extraHeight = Math.round(Math.max(0, Math.min(20000, extraHeight + delta)));
    document.body.style.setProperty("--editor-extra-height", `${extraHeight}px`);
    handle.setAttribute("aria-valuenow", String(extraHeight));
    handle.setAttribute("aria-valuetext", `${extraHeight} extra pixels`);
    return extraHeight - previous;
  };
  const extendAtEdge = () => {
    frame = null;
    if (!drag?.growing || drag.y < window.innerHeight - 32) return;
    const added = resizeBy(12);
    if (!added) return;
    window.scrollBy(0, added);
    frame = requestAnimationFrame(extendAtEdge);
  };
  handle.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    drag = {y: event.clientY, growing: false};
    handle.focus({preventScroll: true});
    handle.classList.add("dragging");
    handle.setPointerCapture(event.pointerId);
  });
  handle.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const delta = event.clientY - drag.y;
    drag.y = event.clientY;
    if (delta) drag.growing = delta > 0;
    resizeBy(delta);
    if (frame === null) frame = requestAnimationFrame(extendAtEdge);
  });
  const finish = () => {
    drag = null;
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null;
    handle.classList.remove("dragging");
  };
  for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) handle.addEventListener(event, finish);
  handle.addEventListener("dblclick", () => { finish(); resizeBy(-extraHeight); });
  handle.addEventListener("keydown", (event) => {
    if (!["ArrowUp", "ArrowDown", "Home"].includes(event.key)) return;
    event.preventDefault();
    resizeBy(event.key === "Home" ? -extraHeight : event.key === "ArrowDown" ? 40 : -40);
  });
}

applyPaneWidths();
initPaneResizer("record-resizer", "record", 1);
initPaneResizer("inspector-resizer", "inspector", -1);
initEditorHeightResizer();
window.addEventListener("resize", applyPaneWidths);

function selectedArtifact() {
  return store.selection?.artifact || SCREEN[store.stage].primary;
}

function nextBoxno(records) {
  return records.reduce((largest, record) => Math.max(largest, Number(record.boxno) || 0), -1) + 1;
}

function newRecordId(prefix = "manual") {
  return `${prefix}_${crypto.randomUUID().replaceAll("-", "").slice(0, 16)}`;
}

function imageUrl(kind = "original") {
  if (kind === "original") return store.document?.image.url;
  return `/api/job/${encodeURIComponent(config.category)}/${encodeURIComponent(config.jobId)}/editor/v2/image/${kind}/${store.page}`;
}

function currentStageState() {
  return store.document?.stageStates?.[store.stage] || {};
}

async function loadTypesettingView() {
  preview.hide();
  currentPreviewRequest?.abort();
  if (typesettingView === "original") {
    canvas.load(imageUrl("original"), { showPlacementText: false });
    setStatus("Original page loaded.");
  } else if (typesettingView === "preview") {
    preview.show();
  } else if (typesettingView === "render") {
    canvas.load(imageUrl("final"), { showPlacementText: false });
    setStatus("Last saved render loaded.");
  } else {
    await exactPreview();
  }
}

async function selectTypesettingView(view) {
  if (!TYPESETTING_VIEWS.has(view)) return;
  typesettingView = view;
  localStorage.setItem(typesettingViewKey, view);
  renderTools();
  await loadTypesettingView();
}

async function load(stage = store.stage, page = store.page) {
  const request = ++loadRequest;
  loadController?.abort();
  const controller = new AbortController();
  loadController = controller;
  currentPreviewRequest?.abort();
  preview.hide();
  navigationPage = page;
  loading = true;
  byId("save-state").textContent = "Loading...";
  try {
    const document = await api.load(stage, page, controller.signal);
    if (request !== loadRequest) return;
    store.setDocument(document);
    mergeSelection.clear();
    byId("page-select").value = String(page);
    setStatus(store.dirty ? "Recovered an unsaved browser draft." : "");
    if (stage === "placement") void loadTypesettingView();
    else canvas.load(document.image.url);
    void loadRevisions();
  } catch (error) {
    if (request === loadRequest && error.name !== "AbortError") {
      setStatus(error.message, true);
      byId("save-state").textContent = "Load failed";
    }
  } finally {
    if (request === loadRequest) { loading = false; navigationPage = null; }
  }
}

async function navigate(stage, page) {
  if (!Number.isInteger(page) || page < 0 || page >= config.pageCount || !AVAILABLE_STAGES.includes(stage)) return;
  const request = ++navigationRequest;
  navigationPage = page;
  loadController?.abort();
  ++loadRequest;
  loading = false;
  if (store.dirty) {
    setStatus("Saving changes before navigation...");
    await save();
    if (request !== navigationRequest) return;
    if (store.dirty) { navigationPage = null; return; }
  }
  if (request !== navigationRequest) return;
  await load(stage, page);
}

async function selectPage() {
  const input = byId("page-select");
  if (!input.reportValidity()) return;
  const page = input.valueAsNumber;
  if (page === (navigationPage ?? store.page)) return;
  await navigate(store.stage, page);
  input.value = String(navigationPage ?? store.page);
}

function recordsForList() {
  const result = [];
  for (const artifact of SCREEN[store.stage].artifacts) {
    for (const record of store.records[artifact] || []) result.push({ artifact, record });
  }
  return result;
}

function rawBoxesFor(artifact, record) {
  if (artifact === "ocr_raw") return [record];
  const sources = new Set(record.sourceBoxnos || []);
  return (store.records.ocr_raw || []).filter((raw) => sources.has(raw.boxno));
}

function ocrBoxesChecked(artifact, record) {
  const raw = rawBoxesFor(artifact, record);
  return raw.length > 0 && raw.every((item) => mergeSelection.has(item.recordId));
}

function selectOcrBoxes(hit, extend = false) {
  if (!extend) mergeSelection.clear();
  if (!hit) { store.select(null, null); return; }
  const raw = rawBoxesFor(hit.artifact, hit.record);
  const remove = extend && raw.length > 0 && raw.every((item) => mergeSelection.has(item.recordId));
  for (const item of raw) remove ? mergeSelection.delete(item.recordId) : mergeSelection.add(item.recordId);
  store.select(hit.artifact, hit.record.recordId);
}

function renderStageTabs() {
  for (const button of document.querySelectorAll(".stage-tab")) {
    const stage = button.dataset.stage;
    const state = store.document?.stageStates?.[stage] || {};
    button.classList.toggle("active", stage === store.stage);
    button.classList.toggle("stale", !!state.stale);
    button.classList.toggle("changed", !!state.changed);
    button.classList.toggle("frozen", !!state.frozen);
    button.disabled = ocrAreaRunning;
  }
}

function renderRecordList() {
  const target = byId("record-list");
  const focused = document.activeElement;
  const focusedId = focused?.dataset?.recordId;
  const filter = byId("record-filter").value.trim().toLowerCase();
  target.replaceChildren();
  for (const { artifact, record } of recordsForList()) {
    const displayRecord = ["placements", "ocr_structured"].includes(artifact)
      ? { ...record, englishText: (store.records.translations || []).find((item) => item.boxno === record.boxno)?.englishText || "" }
      : record;
    const label = recordLabel(store.stage, displayRecord, artifact);
    if (filter && !label.toLowerCase().includes(filter)) continue;
    const row = document.createElement("div");
    row.className = "record-row";
    row.classList.toggle("active", store.selection?.recordId === record.recordId ||
      (store.stage === "ocr" && ocrBoxesChecked(artifact, record)));
    const select = document.createElement("button"); select.type = "button";
    select.dataset.recordId = record.recordId;
    select.className = "record-select"; select.textContent = label;
    const meta = document.createElement("span");
    meta.className = "record-meta";
    const protection = store.document?.protection?.recordStates?.[record.recordId];
    meta.textContent = `${artifact.replace("ocr_", "").replace("placements", "placement")}${protection ? ` · ${protection.protected ? "protected" : "unprotected"} ${protection.kind}` : ""}`;
    select.append(meta);
    if (store.stage === "ocr" && artifact === "ocr_raw") {
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.dataset.recordId = record.recordId;
      checkbox.checked = mergeSelection.has(record.recordId);
      checkbox.title = "Select raw box for grouping";
      checkbox.addEventListener("click", (event) => {
        event.stopPropagation();
        checkbox.checked ? mergeSelection.add(record.recordId) : mergeSelection.delete(record.recordId);
        store.select(artifact, record.recordId);
      });
      row.prepend(checkbox);
    }
    select.addEventListener("click", (event) => store.stage === "ocr"
      ? selectOcrBoxes({artifact, record}, event.ctrlKey || event.metaKey || event.shiftKey)
      : store.select(artifact, record.recordId));
    row.append(select);
    target.append(row);
  }
  if (!target.childElementCount) target.textContent = "No records.";
  if (focusedId) [...target.querySelectorAll("button, input")]
    .find((element) => element.dataset.recordId === focusedId && element.type === focused.type)?.focus({preventScroll: true});
}

function sourceRecord(record) {
  return (store.records.ocr_structured || []).find((item) => item.boxno === record?.boxno);
}

function protectionFor(recordId, field) {
  return store.document?.protection?.records?.[recordId]?.[field];
}

function refreshLineBreaks(input, record) {
  clearTimeout(lineBreaksTimer);
  lineBreaksController?.abort();
  if (record.manualLineBreaks?.trim()) return;
  const pageDocument = store.document;
  const page = store.page;
  const translation = (store.records.translations || []).find((item) => item.boxno === record.boxno)?.englishText || "";
  const entry = {...record, englishText: translation};
  lineBreaksTimer = setTimeout(async () => {
    if (!input.isConnected || input.dataset.userEdited) return;
    const controller = new AbortController();
    lineBreaksController = controller;
    try {
      const text = await api.textLayout(page, entry, controller.signal);
      if (!controller.signal.aborted && store.document === pageDocument && input.isConnected && !input.dataset.userEdited) input.value = text;
    } catch (error) {
      if (!controller.signal.aborted && store.document === pageDocument && input.isConnected && !input.dataset.userEdited)
        setStatus(`Could not load automatic line breaks: ${error.message}`, true);
    }
  }, 200);
}

function renderInspector() {
  const form = byId("record-form");
  form.replaceChildren();
  const record = store.selectedRecord();
  const cleanOnly = store.selection?.artifact === "clean_boxes";
  byId("selection-label").textContent = record ? `${cleanOnly ? "Clean box" : "Box"} ${record.boxno ?? "?"}` : "No selection";
  byId("notes-panel").hidden = store.stage !== "text";
  if (store.document) {
    byId("job-notes").value = store.document.translationNotes?.job || "";
    byId("page-notes").value = store.document.translationNotes?.page || "";
  }
  if (!record) return;
  const formTarget = { ...store.selection, stage: store.stage };

  const values = {};
  const regionFields = document.createElement("div");
  regionFields.className = "field-grid";
  const pairedFields = new Map();
  const pairs = {font: "font", fill: "font", stroke: "outline", strokeWidth: "outline", gravity: "size", fontSizeWidthPercent: "size"};
  const source = store.stage === "placement" && !cleanOnly ? sourceRecord(record) : record;
  const displayRecord = store.stage === "text" || (store.stage === "placement" && !cleanOnly)
    ? {...record, text: source?.text || "", englishText: (store.records.translations || []).find((item) => item.boxno === record.boxno)?.englishText || ""}
    : record;
  for (const [name, labelText, type, value, choices] of fieldsFor(store.stage, displayRecord, store.document?.fonts || [], store.selection.artifact)) {
    const isAngle = ["rotation", "skewX", "skewY"].includes(name);
    const wrapper = document.createElement(isAngle ? "div" : "label");
    wrapper.className = "field-row";
    const heading = document.createElement("span"); heading.className = "field-label";
    wrapper.append(heading);
    if (["manualLineBreaks", "fontSizeWidthPercent"].includes(name)) {
      const help = document.createElement("span"); help.className = "field-help"; help.tabIndex = 0;
      help.textContent = labelText;
      help.dataset.tooltip = name === "fontSizeWidthPercent"
        ? "Font size as a percentage of page width. Clear the field to use the automatic size shown here."
        : "Shows the current automatic line breaks as the box and text settings change. Editing this text fixes your chosen line breaks; resizing then shrinks the font rather than adding breaks. Clear the field to return to automatic wrapping. The saved translation is unchanged.";
      heading.append(help);
    } else heading.append(document.createTextNode(labelText));
    let input;
    if (type === "textarea") {
      input = document.createElement("textarea"); input.rows = ["text", "englishText"].includes(name) ? 2 : 4; input.value = value;
    } else if (type === "select") {
      input = document.createElement("select");
      for (const choice of choices || []) {
        const option = document.createElement("option"); option.value = choice;
        option.textContent = name === "gravity" ? ALIGNMENT_LABELS[choice] : (choice || "Auto / backup");
        input.append(option);
      }
      input.value = value;
    } else {
      input = document.createElement("input"); input.type = type;
      if (type === "checkbox") input.checked = !!value;
      else input.value = value;
      if (name === "fontSizeWidthPercent") { input.min = "0.05"; input.max = "100"; input.step = "0.05"; }
      if (name === "lineSpacing") { input.min = "-200"; input.max = "200"; input.step = "1"; }
      if (isAngle) {
        const limit = name === "rotation" ? 180 : 60;
        input.min = String(-limit); input.max = String(limit); input.step = "0.5";
      }
    }
    input.name = name;
    input.setAttribute("aria-label", labelText);
    values[name] = input;
    if (isAngle) {
      const controls = document.createElement("div"); controls.className = "angle-controls";
      const slider = document.createElement("input"); slider.type = "range";
      slider.min = input.min; slider.max = input.max; slider.step = input.step; slider.value = value;
      slider.tabIndex = -1;
      slider.setAttribute("aria-label", labelText);
      input.setAttribute("aria-label", labelText);
      slider.addEventListener("input", () => { input.value = slider.value; });
      slider.addEventListener("change", () => {
        input.value = slider.value;
        applyInspector(values, formTarget, name);
      });
      input.addEventListener("change", () => {
        slider.value = input.value === "" ? "0" : input.value;
        applyInspector(values, formTarget, name);
      });
      controls.append(input, slider); wrapper.append(controls);
    } else {
      input.addEventListener("change", () => applyInspector(values, formTarget, name));
      wrapper.append(input);
    }
    if (name === "manualLineBreaks") {
      input.addEventListener("input", () => { input.dataset.userEdited = "1"; });
      refreshLineBreaks(input, record);
    }

    const overrideField = ["left", "top", "right", "bottom"].includes(name)
      ? (store.stage === "placement" && !cleanOnly ? "placementRegion" : "region")
      : name;
    const fieldTarget = source && (store.stage === "text" || (store.stage === "placement" && ["text", "englishText"].includes(name)))
      ? textFieldTarget(store.records, source, name) : null;
    const protectedValue = protectionFor(fieldTarget ? fieldTarget.recordId : record.recordId, overrideField);
    if (protectedValue !== undefined) {
      const controls = document.createElement("span"); controls.className = "field-protection";
      const toggle = document.createElement("button"); toggle.type = "button";
      toggle.textContent = "P";
      toggle.setAttribute("aria-pressed", String(!!protectedValue));
      toggle.setAttribute("aria-label", `${protectedValue ? "Unprotect" : "Protect"} ${labelText}`);
      toggle.dataset.tooltip = protectedValue
        ? "Allow a later model rerun to replace this saved field."
        : "Keep this saved field unchanged during later model reruns.";
      toggle.addEventListener("click", () => changeProtection("field", record, overrideField, !protectedValue, fieldTarget));
      const revert = document.createElement("button"); revert.type = "button"; revert.textContent = "R";
      revert.className = "reset-field";
      revert.setAttribute("aria-label", `Reset ${labelText} to generated value`);
      revert.dataset.tooltip = "Remove this field override and restore the latest generated value.";
      revert.addEventListener("click", () => changeProtection("revert-field", record, overrideField, false, fieldTarget));
      controls.append(toggle, revert); heading.append(controls);
    }
    if (["left", "top", "right", "bottom"].includes(name)) {
      if (!regionFields.parentNode) form.append(regionFields);
      regionFields.append(wrapper);
    } else if (store.stage === "placement" && pairs[name]) {
      const pair = pairs[name];
      if (!pairedFields.has(pair)) {
        const row = document.createElement("div"); row.className = `field-pair ${pair}-pair`;
        pairedFields.set(pair, row); form.append(row);
      }
      pairedFields.get(pair).append(wrapper);
    } else {
      form.append(wrapper);
    }
  }

  const relatedIds = store.stage === "text"
    ? [...new Set([record.recordId, textFieldTarget(store.records, record, "englishText").recordId].filter(Boolean))]
    : [record.recordId];
  if (cleanOnly) {
    const note = document.createElement("p");
    note.textContent = "LaMa cleans this area without adding text. It can alter artwork. Preview refreshes automatically. Press Delete to remove this box.";
    form.append(note);
  } else if (relatedIds.some((rid) => store.document?.protection?.recordStates?.[rid] || store.document?.protection?.records?.[rid])) {
    const row = document.createElement("div"); row.className = "field-protection";
    const toggle = document.createElement("button"); toggle.type = "button"; toggle.textContent = "P";
    toggle.dataset.tooltip = "Protect or unprotect all saved changes for this record.";
    const fieldStates = relatedIds.flatMap((rid) => Object.values(store.document?.protection?.records?.[rid] || {}));
    const recordStates = relatedIds.map((rid) => store.document?.protection?.recordStates?.[rid]).filter(Boolean);
    const allProtected = (fieldStates.length > 0 || !!recordStates.length)
      && fieldStates.every(Boolean)
      && recordStates.every((state) => state.protected);
    toggle.setAttribute("aria-pressed", String(allProtected));
    toggle.setAttribute("aria-label", `${allProtected ? "Unprotect" : "Protect"} record`);
    toggle.addEventListener("click", () => changeProtection("record", record, null, !allProtected));
    const revert = document.createElement("button"); revert.type = "button"; revert.textContent = "R";
    revert.className = "reset-field"; revert.setAttribute("aria-label", "Reset record to generated values");
    revert.dataset.tooltip = "Remove all saved overrides for this record and restore its generated values.";
    revert.addEventListener("click", () => changeProtection("revert-record", record, null, false));
    row.append(document.createTextNode("Record "), toggle, revert); form.append(row);
  }
}

function applyInspector(inputs, target, changedField) {
  const record = (store.records[target.artifact] || []).find(
    (item) => item.recordId === target.recordId,
  );
  if (!record) return;
  if (target.stage === "text" || (target.stage === "placement" && target.artifact === "placements" && ["text", "englishText"].includes(changedField))) {
    const source = target.stage === "text" ? record : sourceRecord(record);
    if (!source) return;
    const input = inputs[changedField];
    store.mutate((records) => updateTextField(records, source, changedField,
      input.type === "checkbox" ? input.checked : input.value, store.page, newRecordId));
    return;
  }
  const values = {};
  for (const [name, input] of Object.entries(inputs)) {
    if (target.stage === "placement" && ["text", "englishText"].includes(name)) continue;
    if (name === "manualLineBreaks" && changedField !== name) continue;
    values[name] = input.type === "checkbox" ? input.checked : input.value;
    if (input.type === "number" && input.value !== "") values[name] = Number(input.value);
  }
  const updated = applyFields(target.stage, record, values, target.artifact);
  store.mutate((records) => {
    const items = records[target.artifact];
    items[items.indexOf(record)] = updated;
  });
}

function addButton(target, text, handler, title = "", tooltip = "", shortcut = "") {
  const button = document.createElement("button"); button.type = "button";
  button.textContent = shortcut ? `${text} [${shortcut.toUpperCase()}]` : text;
  button.title = title;
  if (tooltip) button.dataset.tooltip = tooltip;
  if (shortcut) button.dataset.shortcut = shortcut.toLowerCase();
  button.addEventListener("click", handler); target.append(button); return button;
}

function renderTools() {
  const target = byId("stage-tools"); target.replaceChildren();
  const canHideBoxes = store.stage !== "placement" || typesettingView === "preview";
  const hideBoxes = canHideBoxes && hideEditorBoxes;
  if (canvas.hideBoxes !== hideBoxes) { canvas.hideBoxes = hideBoxes; canvas.draw(); }
  if (store.stage === "ocr") {
    addButton(target, "Add raw box", () => canvas.startDraw(addRawBox), "Draw a missing OCR box", "Draw one raw OCR box manually.", "z");
    addButton(target, ocrAreaRunning ? "OCR running..." : "OCR area", () => canvas.startDraw(ocrArea), "OCR a selected region", "Draw a region to scan only that area. The result adds raw OCR boxes and merges the new boxes with one another.", "x");
    addButton(target, "Group checked", groupChecked, "Merge checked raw boxes", "Ctrl/Shift-click boxes or check raw rows, then group them. Double-click the page to cycle overlapping boxes.", "c");
    addButton(target, "Split group", splitGroup, "Split the selected merged group", "Delete the selected merged group and create one separate group for each of its source OCR boxes.", "v");
    addButton(target, "Delete", deleteSelected, "Delete selected", "Delete the selected raw box or merged group.", "b");
  } else if (store.stage === "text") {
    addButton(target, "Move earlier", () => moveSelected(-1), "Move earlier in reading order", "Move the selected record one position earlier in the page reading order. The translation pass follows this order.", "z");
    addButton(target, "Move later", () => moveSelected(1), "Move later in reading order", "Move the selected record one position later in the page reading order. The translation pass follows this order.", "x");
    addButton(target, "Reject", deleteSelected, "Reject this OCR group", "Remove the selected structured record from translation, typesetting, and rendering. Use this only for OCR that is not real text.", "c");
    addButton(
      target,
      queuedAction ? "Translation request running..." : "Retranslate page",
      () => retranslate("page"),
      "Request a new VLM translation for this page",
      "Replace all translations on this page with a new VLM translation, then rerun Typesetting and Render.",
      "v",
    );
    addButton(
      target,
      queuedAction ? "Translation request running..." : "Retranslate selected",
      () => retranslate("record"),
      "Request a new VLM translation for the selected record",
      "Replace only the selected translation with a new VLM translation. Other records remain unchanged.",
      "b",
    );
  } else if (store.stage === "placement") {
    const views = [
      ["current", "Current", "Show the exact current draft. This uses the saved render when it is current, or creates a cached ImageMagick preview when needed."],
      ["original", "Original", "Show the original source page without translated text."],
      ["preview", "Preview", "Show ImageMagick text layers first, then clean the background with LaMa. Clean-box edits refresh automatically."],
      ["render", "Last Render", "Show the last page produced by the pipeline Render pass."],
    ];
    const shortcuts = ["z", "x", "c", "v"];
    for (const [index, [view, label, tooltip]] of views.entries()) {
      const button = addButton(target, label, () => selectTypesettingView(view), "", tooltip, shortcuts[index]);
      button.classList.toggle("active", typesettingView === view);
    }
    addButton(target, "Add clean box", () => canvas.startDraw(addCleanBox), "Draw a clean-only area",
      "LaMa cleans this rectangle without adding text or calling the VLM. Preview refreshes automatically.", "b");
  }
  if (canHideBoxes) {
    const label = document.createElement("label"); label.className = "box-toggle";
    const toggle = document.createElement("input"); toggle.type = "checkbox"; toggle.checked = hideEditorBoxes;
    toggle.dataset.shortcut = "h";
    toggle.addEventListener("change", () => {
      hideEditorBoxes = toggle.checked;
      canvas.hideBoxes = hideEditorBoxes;
      canvas.draw();
    });
    label.append(toggle, document.createTextNode(" Hide boxes [H]")); target.append(label);
  }
  if (ocrAreaRunning) {
    for (const button of target.querySelectorAll("button")) button.disabled = true;
    const progress = document.createElement("span");
    progress.className = "tool-progress";
    progress.setAttribute("role", "status");
    progress.textContent = "OCR is processing the selected area.";
    target.append(progress);
  } else if (queuedAction && store.stage === "text") {
    for (const button of target.querySelectorAll("button")) button.disabled = true;
    const progress = document.createElement("span");
    progress.className = "tool-progress";
    progress.setAttribute("role", "status");
    progress.textContent = "The translation request is being added to the job queue.";
    target.append(progress);
  }
}

function addCleanBox(region) {
  store.mutate((records) => {
    const boxes = records.clean_boxes ||= [];
    const record = {page: store.page, boxno: nextBoxno(boxes), recordId: newRecordId("clean"), region};
    boxes.push(record);
    store.selection = {artifact: "clean_boxes", recordId: record.recordId};
  });
  setStatus("Clean box added. Preview refreshes automatically.");
}

function addRawBox(region) {
  store.mutate(() => {
    const records = store.records.ocr_raw || (store.records.ocr_raw = []);
    const record = { page: store.page, boxno: nextBoxno(records), recordId: newRecordId("raw"), region, text: "", ocrSource: "manual" };
    records.push(record); store.selection = { artifact: "ocr_raw", recordId: record.recordId };
    const groups = store.records.ocr_merged ||= [];
    groups.push({page: store.page, boxno: nextBoxno(groups), recordId: newRecordId("group"),
      sourceBoxnos: [record.boxno], sourceTexts: [""], text: "", region: [...region], ocrSource: "manual"});
    mergeSelection.clear(); mergeSelection.add(record.recordId);
  });
}

async function ocrArea(region) {
  const page = store.page;
  setOcrAreaRunning(true);
  setStatus("Running OCR on selected area...");
  try {
    const raw = store.records.ocr_raw || [];
    const result = await api.cropOcr(page, region, nextBoxno(raw));
    store.mutate(() => {
      const rawRecords = result.rawRecords || result.records || [];
      const boxMap = new Map();
      for (const item of rawRecords) {
        const old = item.boxno, boxno = nextBoxno(store.records.ocr_raw || []);
        boxMap.set(old, boxno);
        (store.records.ocr_raw ||= []).push({ ...item, page: store.page, boxno, recordId: newRecordId("raw") });
      }
      for (const item of result.mergedRecords || []) {
        const sourceBoxnos = (item.sourceBoxnos || []).map((boxno) => boxMap.get(boxno)).filter(Number.isInteger);
        (store.records.ocr_merged ||= []).push({ ...item, page: store.page, boxno: nextBoxno(store.records.ocr_merged), sourceBoxnos, recordId: newRecordId("group") });
      }
    });
    setStatus(`OCR added ${(result.rawRecords || result.records || []).length} raw boxes.`);
  } catch (error) { setStatus(error.message, true); }
  finally { setOcrAreaRunning(false); }
}

function unionRegion(records) {
  return [Math.min(...records.map((r) => r.region[0])), Math.min(...records.map((r) => r.region[1])), Math.max(...records.map((r) => r.region[2])), Math.max(...records.map((r) => r.region[3]))];
}

function groupChecked() {
  const raw = (store.records.ocr_raw || []).filter((item) => mergeSelection.has(item.recordId));
  if (!raw.length) return setStatus("Check one or more raw boxes first.", true);
  store.mutate(() => {
    const selectedBoxnos = new Set(raw.map((item) => item.boxno));
    const boxno = nextBoxno(store.records.ocr_merged || []);
    const rawByBoxno = new Map((store.records.ocr_raw || []).map((item) => [item.boxno, item]));
    store.records.ocr_merged = (store.records.ocr_merged || []).filter((group) => {
      group.sourceBoxnos = (group.sourceBoxnos || []).filter((boxno) => !selectedBoxnos.has(boxno));
      const remaining = group.sourceBoxnos.map((source) => rawByBoxno.get(source)).filter(Boolean);
      if (remaining.length) {
        group.sourceTexts = remaining.map((item) => item.text || "");
        group.text = group.sourceTexts.join(""); group.region = unionRegion(remaining);
      }
      return group.sourceBoxnos.length;
    });
    const groups = store.records.ocr_merged;
    const record = { page: store.page, boxno, recordId: newRecordId("group"), sourceBoxnos: raw.map((item) => item.boxno), sourceTexts: raw.map((item) => item.text || ""), text: raw.map((item) => item.text || "").join(""), region: unionRegion(raw) };
    groups.push(record); store.selection = { artifact: "ocr_merged", recordId: record.recordId };
    mergeSelection.clear();
  });
}

function splitGroup() {
  const group = store.selectedRecord();
  if (!group || selectedArtifact() !== "ocr_merged") return setStatus("Select a merged group first.", true);
  const rawByBox = new Map((store.records.ocr_raw || []).map((item) => [item.boxno, item]));
  store.mutate(() => {
    store.records.ocr_merged = (store.records.ocr_merged || []).filter((item) => item.recordId !== group.recordId);
    for (const boxno of group.sourceBoxnos || []) {
      const raw = rawByBox.get(boxno); if (!raw) continue;
      store.records.ocr_merged.push({ page: store.page, boxno: nextBoxno(store.records.ocr_merged), recordId: newRecordId("group"), sourceBoxnos: [boxno], sourceTexts: [raw.text || ""], text: raw.text || "", region: [...raw.region] });
    }
    store.selection = null;
  });
}

function deleteSelected() {
  if (!store.selection) return;
  if (store.stage === "placement") {
    if (store.selection.artifact !== "clean_boxes") return setStatus("Select a clean box to delete it. Reject text in Text & Translation.", true);
    store.deleteSelected();
    setStatus("Clean box deleted. Preview refreshes automatically.");
    return;
  }
  if (store.stage === "text") {
    const record = store.selectedRecord();
    store.mutate((records) => {
      records.ocr_structured = records.ocr_structured.filter((item) => item.boxno !== record.boxno);
      store.selection = null;
    });
    return;
  }
  if (store.stage === "ocr" && store.selection.artifact === "ocr_raw") {
    const raw = store.selectedRecord();
    store.mutate(() => {
      store.records.ocr_raw = (store.records.ocr_raw || []).filter((item) => item.recordId !== raw.recordId);
      store.records.ocr_merged = (store.records.ocr_merged || []).filter((group) => {
        group.sourceBoxnos = (group.sourceBoxnos || []).filter((boxno) => boxno !== raw.boxno);
        const remaining = group.sourceBoxnos.map((boxno) => store.records.ocr_raw.find((item) => item.boxno === boxno)).filter(Boolean);
        group.sourceTexts = remaining.map((item) => item.text || "");
        group.text = group.sourceTexts.join("");
        if (remaining.length) group.region = unionRegion(remaining);
        return group.sourceBoxnos.length;
      });
      store.selection = null;
    });
  } else {
    store.deleteSelected();
  }
}

function moveSelected(delta) {
  const selection = store.selection; if (!selection) return;
  store.mutate(() => {
    const records = store.records[selection.artifact] || [];
    const index = records.findIndex((item) => item.recordId === selection.recordId);
    const target = Math.max(0, Math.min(records.length - 1, index + delta));
    if (index < 0 || index === target) return;
    const [record] = records.splice(index, 1); records.splice(target, 0, record);
  });
}

async function save() {
  if (savePromise) return savePromise;
  if (resetting) return;
  if (!store.document || !store.dirty) return;
  const previous = store.document;
  const snapshot = JSON.stringify([store.records, store.document.translationNotes]);
  byId("save").disabled = true; byId("save-state").textContent = "Saving...";
  savePromise = (async () => {
    try {
      const document = await api.save(store.stage, store.page, {
        baseRevision: store.document.revision,
        deletedOcrGroupIds: store.stage === "ocr" ? (store.records.ocr_removed_groups || []) : undefined,
        textEdits: store.stage === "placement" ? ["ocr_structured", "translations"].flatMap((artifact) => {
          const field = artifact === "translations" ? "englishText" : "text";
          return (store.records[artifact] || []).filter((record) => {
            const previous = (store.document.recordsByArtifact[artifact] || []).find((item) => item.recordId === record.recordId);
            return (record[field] || "") !== (previous?.[field] || "");
          }).map((record) => ({artifact, recordId: record.recordId, boxno: record.boxno, value: record[field] || ""}));
        }) : undefined,
        recordsByArtifact: Object.fromEntries(
          [...SCREEN[store.stage].artifacts, ...(store.stage === "text" ? ["translations"] : [])]
            .map((artifact) => [artifact, store.records[artifact] || []]),
        ),
        translationNotes: store.stage === "text" ? { job: byId("job-notes").value, page: byId("page-notes").value } : undefined,
      });
      if (store.document !== previous) return;
      if (JSON.stringify([store.records, store.document.translationNotes]) === snapshot) store.markSaved(document);
      else {
        store.document.revision = document.revision;
        store.saveDraft();
      }
      setStatus(store.dirty ? "Saved. Newer edits still need saving." : "Saved. Changed fields are protected.");
    } catch (error) { setStatus(error.message, true); }
    finally { byId("save").disabled = false; }
  })();
  try { await savePromise; }
  finally { savePromise = null; }
}

async function resetStage() {
  if (!store.document || loading || resetting || ocrAreaRunning || queuedAction) return;
  const label = currentStageState().label;
  if (!confirm(`Reset ${label} on page ${store.page}? All saved and unsaved edits and protection flags for this stage will be removed. Other stages, pages, and saved translation notes are kept.`)) return;
  resetting = true;
  byId("editor-app").inert = true;
  byId("reset-stage").disabled = true;
  setStatus("Resetting stage...");
  try {
    const document = await api.reset(store.stage, store.page, store.document.revision);
    mergeSelection.clear();
    store.markSaved(document);
    if (store.stage === "placement") await loadTypesettingView();
    else canvas.load(document.image.url);
    await loadRevisions();
    setStatus(config.reviewCheckpoint === "ocr"
      ? "Stage reset. Continue processing when the OCR review is complete."
      : document.rerunFrom ? "Stage reset. Regenerate page to update the output."
        : "Stage reset. No changes need regeneration.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    resetting = false;
    byId("editor-app").inert = false;
    byId("reset-stage").disabled = false;
    byId("reset-stage").focus();
  }
}

async function changeProtection(action, record, field, protectedValue, fieldTarget = null) {
  if (store.dirty) { setStatus("Save the current draft before changing protection.", true); return; }
  try {
    const document = await api.protect(store.stage, store.page, {
      baseRevision: store.document.revision, action, artifact: selectedArtifact(), recordId: record.recordId, field, protected: protectedValue,
      ...(fieldTarget || {}),
    });
    store.setDocument(document); setStatus("Protection updated.");
  } catch (error) { setStatus(error.message, true); }
}

async function toggleFreeze(scope) {
  if (store.dirty) { setStatus("Save the current draft before freezing it.", true); return; }
  const frozen = scope === "stage" ? !!currentStageState().frozen : AVAILABLE_STAGES.every((stage) => store.document.stageStates[stage]?.frozen);
  try {
    const document = await api.protect(store.stage, store.page, { baseRevision: store.document.revision, action: scope === "stage" ? "freeze-stage" : "freeze-page", protected: !frozen });
    store.setDocument(document); setStatus(`${scope === "stage" ? "Stage" : "Page"} ${!frozen ? "protected" : "unprotected"}.`);
  } catch (error) { setStatus(error.message, true); }
}

async function regenerate(action) {
  if (queuedAction || loading || resetting || ocrAreaRunning) return;
  if (action === "stage" && !confirm("Regenerate this stage on this page? Unprotected generated values can be replaced. Later stages stay out of date.")) return;
  if (action === "finish-edits" && !confirm("Apply all saved edits as the job output? This uses LaMa and ImageMagick only. No OCR or VLM calls are made; untranslated boxes stay untranslated. Old CBZ exports will be deleted.")) return;
  setQueuedAction(action, "Saving edits and queueing the action...");
  byId("editor-app").inert = true;
  try {
    if (store.dirty) await save();
    if (store.dirty) { setQueuedAction(""); return; }
    const result = action === "stage" ? await api.regenerate(store.stage, store.page)
      : action === "page" ? await api.regeneratePage(store.page)
        : action === "all-pages" ? await api.regenerateAllPages() : await api.finishEdits();
    window.location.href = result.url;
  } catch (error) { setQueuedAction(""); setStatus(error.message, true); }
  finally { byId("editor-app").inert = false; }
}

async function continueProcessing() {
  if (queuedAction || config.reviewCheckpoint !== "ocr") return;
  if (store.dirty) await save();
  if (store.dirty) return;
  setQueuedAction("continue", "Saving the OCR review checkpoint and queueing Structure...");
  const button = byId("continue-processing");
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  button.textContent = "Queueing...";
  try {
    const result = await api.continueProcessing();
    window.location.href = result.url;
  } catch (error) {
    setQueuedAction("");
    button.disabled = false;
    button.setAttribute("aria-busy", "false");
    button.textContent = "Continue processing";
    setStatus(error.message, true);
  }
}

async function retranslate(scope) {
  if (queuedAction || store.stage !== "text") return;
  const source = scope === "record" ? store.selectedRecord() : null;
  const selected = source ? (store.records.translations || []).find((item) => item.boxno === source.boxno) : null;
  if (scope === "record" && !selected) {
    setStatus("Select one translation record first.", true);
    return;
  }
  const message = scope === "record"
    ? `Replace translation ${selected.boxno} with a new VLM translation?`
    : `Replace every translation on page ${store.page} with new VLM output?`;
  if (!confirm(message)) return;
  const requestRecord = selected
    ? { recordId: selected.recordId, boxno: selected.boxno }
    : null;
  if (store.dirty) await save();
  if (store.dirty) return;
  setQueuedAction(
    "translation",
    scope === "record"
      ? `Adding translation ${selected.boxno} to the VLM queue...`
      : `Adding page ${store.page} to the VLM translation queue...`,
  );
  try {
    const result = await api.retranslate(store.page, requestRecord);
    window.location.href = result.url;
  } catch (error) {
    setQueuedAction("");
    setStatus(error.message, true);
  }
}

async function exactPreview() {
  const document = store.document, page = store.page;
  const controller = new AbortController();
  currentPreviewRequest = controller;
  try {
    const placementState = store.document?.stageStates?.placement || {};
    const renderIsCurrent = !store.dirty && !placementState.pending && !placementState.stale;
    if (renderIsCurrent) {
      canvas.load(imageUrl("final"), { showPlacementText: false });
      setStatus("Saved render loaded. No preview generation was needed.");
      return;
    }
    canvas.load(imageUrl("original"), {showPlacementText: false});
    setStatus("Checking the cleaning mask and preparing Current...");
    const result = await api.preview(
      "placement",
      store.page,
      store.records.placements || [],
      store.records.ocr_structured || [],
      store.records.clean_boxes || [],
      controller.signal,
    );
    if (controller.signal.aborted || store.document !== document || store.page !== page || typesettingView !== "current") return;
    canvas.load(result.url, { showPlacementText: false });
    if (result.cleanedWithLama) {
      setStatus("Current was cleaned with LaMa and rendered with ImageMagick.");
    } else {
      setStatus(result.cached ? "Cached Current preview loaded." : "Current preview rendered with ImageMagick.");
    }
  } catch (error) {
    if (!controller.signal.aborted && store.document === document && error.name !== "AbortError") setStatus(error.message, true);
  }
}

async function loadRevisions() {
  const pageDocument = store.document;
  const target = byId("revision-list"); target.textContent = "Loading...";
  try {
    const data = await api.revisions();
    if (store.document !== pageDocument) return;
    target.replaceChildren();
    for (const item of (data.revisions || []).slice(0, 50)) {
      const row = document.createElement("div"); row.className = "history-row";
      row.append(document.createTextNode(`#${item.revision} ${item.updatedAt ? new Date(item.updatedAt).toLocaleString() : ""}`));
      const button = document.createElement("button"); button.type = "button"; button.textContent = "Restore";
      button.dataset.tooltip = "Replace the current editor state with this saved revision. This does not regenerate pipeline output.";
      button.addEventListener("click", async () => {
        if (!confirm(`Restore editor revision ${item.revision}?`)) return;
        await api.restoreRevision(item.revision); await load(store.stage, store.page);
      });
      row.append(button); target.append(row);
    }
    if (!target.childElementCount) target.textContent = "No saved revisions.";
  } catch (error) { if (store.document === pageDocument) target.textContent = error.message; }
}

function render() {
  if (!store.document) return;
  renderStageTabs(); renderRecordList(); renderInspector(); renderTools();
  byId("undo").disabled = !store.history.length;
  byId("redo").disabled = !store.future.length;
  byId("save").disabled = !store.dirty;
  byId("save-state").textContent = store.dirty ? "Unsaved draft" : `Saved revision ${store.document.revision}`;
  byId("freeze-stage").textContent = currentStageState().frozen ? "Unprotect stage" : "Protect stage";
  byId("freeze-stage").dataset.tooltip = currentStageState().frozen
    ? "Allow later reruns to replace values in this stage on this page. Individual protected edits remain protected."
    : "Protect all current values in this stage on this page. Later reruns keep them until you unprotect the stage.";
  const pageFrozen = AVAILABLE_STAGES.every((stage) => store.document.stageStates[stage]?.frozen);
  byId("freeze-page").textContent = pageFrozen ? "Unprotect page" : "Protect page";
  byId("freeze-page").dataset.tooltip = pageFrozen
    ? "Allow later reruns to replace values on this page. Individual protected edits remain protected."
    : "Protect all editor stages on this page. Later reruns keep their current values until you unprotect the page.";
}

store.addEventListener("change", render);
document.querySelectorAll(".stage-tab").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.stage, store.page)));
byId("page-select").addEventListener("change", selectPage);
byId("page-select").addEventListener("keydown", (event) => {
  if (event.key === "Enter") { event.preventDefault(); void selectPage(); }
});
byId("prev-page").addEventListener("click", () => navigate(store.stage, (navigationPage ?? store.page) - 1));
byId("next-page").addEventListener("click", () => navigate(store.stage, (navigationPage ?? store.page) + 1));
byId("record-filter").addEventListener("input", renderRecordList);
byId("undo").addEventListener("click", () => store.undo());
byId("redo").addEventListener("click", () => store.redo());
byId("save").addEventListener("click", save);
byId("reset-stage").addEventListener("click", resetStage);
byId("freeze-stage").addEventListener("click", () => toggleFreeze("stage"));
byId("freeze-page").addEventListener("click", () => toggleFreeze("page"));
for (const action of ["stage", "page", "all-pages", "finish-edits"]) {
  byId(action === "finish-edits" ? action : `regenerate-${action}`).addEventListener("click", () => regenerate(action));
}
byId("continue-processing").addEventListener("click", continueProcessing);
for (const id of ["job-notes", "page-notes"]) byId(id).addEventListener("input", () => {
  store.document.translationNotes = {
    job: byId("job-notes").value,
    page: byId("page-notes").value,
  };
  store.dirty = true;
  store.saveDraft();
  byId("save").disabled = false;
  byId("save-state").textContent = "Unsaved draft";
});
document.addEventListener("keydown", (event) => {
  if (resetting) return;
  const typing = event.target.isContentEditable ||
    (["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName) && event.target.type !== "checkbox");
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") { event.preventDefault(); save(); return; }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") { event.preventDefault(); event.shiftKey ? store.redo() : store.undo(); return; }
  if (typing) return;
  if (event.ctrlKey || event.metaKey || event.altKey || ocrAreaRunning) return;
  const key = event.key.toLowerCase();
  const stageIndex = Number.parseInt(key, 10) - 1;
  if (stageIndex >= 0 && stageIndex < AVAILABLE_STAGES.length && String(stageIndex + 1) === key) {
    event.preventDefault(); navigate(AVAILABLE_STAGES[stageIndex], store.page); return;
  }
  if (key === "a") {
    event.preventDefault(); navigate(store.stage, (navigationPage ?? store.page) - 1); return;
  }
  if (key === "d") {
    event.preventDefault(); navigate(store.stage, (navigationPage ?? store.page) + 1); return;
  }
  if (event.key === "Delete") {
    event.preventDefault(); deleteSelected(); return;
  }
  const stageAction = byId("stage-tools").querySelector(`[data-shortcut="${key}"]`);
  if (stageAction && !stageAction.disabled) {
    event.preventDefault(); stageAction.click();
  }
});
window.addEventListener("beforeunload", (event) => { if (store.dirty) { event.preventDefault(); event.returnValue = ""; } });

load();
