import { EditorApi } from "./api.js";
import { EditorStore } from "./store.js";
import { EditorCanvas } from "./canvas.js";
import { STAGES, SCREEN, recordLabel, fieldsFor, applyFields } from "./screens.js";

const config = window.TETOLATE_EDITOR_V2;
const AVAILABLE_STAGES = Array.isArray(config.availableStages)
  ? STAGES.filter((stage) => config.availableStages.includes(stage))
  : STAGES;
const api = new EditorApi(config.category, config.jobId);
const store = new EditorStore(config);
const byId = (id) => document.getElementById(id);
const canvas = new EditorCanvas(byId("page-canvas"), store, setStatus);
const mergeSelection = new Set();
let loading = false;
let ocrAreaRunning = false;
let queuedAction = "";
let tooltipAnchor = null;
let tooltipTimer = null;
const typesettingViewKey = `tetolate-editor-view:${config.category}:${config.jobId}`;
const TYPESETTING_VIEWS = new Set(["current", "original", "preview", "render"]);
let typesettingView = localStorage.getItem(typesettingViewKey) || "current";
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
  const regenerateButton = byId("regenerate");
  const regenerateAllButton = byId("regenerate-all");
  const busy = !!action;
  regenerateButton.disabled = busy;
  regenerateAllButton.disabled = busy;
  regenerateButton.setAttribute("aria-busy", String(busy));
  regenerateAllButton.setAttribute("aria-busy", String(busy));
  regenerateButton.textContent = action === "downstream" ? "Queueing regeneration..." : "Regenerate downstream";
  regenerateAllButton.textContent = action === "all" ? "Queueing regeneration..." : "Regenerate all changes";
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
  const anchor = event.target.closest?.("button[data-tooltip]");
  if (anchor) showTooltip(anchor);
});
document.addEventListener("pointerout", (event) => {
  const anchor = event.target.closest?.("button[data-tooltip]");
  if (anchor && !anchor.contains(event.relatedTarget)) scheduleTooltipHide();
});
document.addEventListener("focusin", (event) => {
  const anchor = event.target.closest?.("button[data-tooltip]");
  if (anchor) showTooltip(anchor);
});
document.addEventListener("focusout", (event) => {
  const anchor = event.target.closest?.("button[data-tooltip]");
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

applyPaneWidths();
initPaneResizer("record-resizer", "record", 1);
initPaneResizer("inspector-resizer", "inspector", -1);
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

function downstreamRegenerationTooltip() {
  const definitions = store.document?.stageDefinitions || [];
  const labels = new Map(definitions.map((item) => [item.key, item.label]));
  const current = definitions.find((item) => item.key === store.stage);
  const later = (current?.invalidates || []).map((stage) => labels.get(stage) || stage);
  if (!later.includes("Render")) later.push("Render");
  const passList = later.length ? later.join(", ") : "later stages";
  return `Save this page, then rerun ${passList} for this page. ${current?.label || "The current stage"} is used as input and is not regenerated.`;
}

async function loadTypesettingView() {
  if (typesettingView === "original") {
    canvas.load(imageUrl("original"), { showPlacementText: false });
    setStatus("Original page loaded.");
  } else if (typesettingView === "preview") {
    canvas.load(imageUrl("cleaned"));
    setStatus("Rough typesetting preview loaded.");
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
  if (loading) return;
  loading = true;
  byId("save-state").textContent = "Loading...";
  try {
    const document = await api.load(stage, page);
    store.setDocument(document);
    mergeSelection.clear();
    byId("page-select").value = String(page);
    if (stage === "placement") await loadTypesettingView();
    else canvas.load(document.image.url);
    setStatus(store.dirty ? "Recovered an unsaved browser draft." : "");
    await loadRevisions();
  } catch (error) {
    setStatus(error.message, true);
    byId("save-state").textContent = "Load failed";
  } finally {
    loading = false;
  }
}

async function navigate(stage, page) {
  if (page < 0 || page >= config.pageCount || !AVAILABLE_STAGES.includes(stage)) return;
  if (store.dirty) {
    setStatus("Saving changes before navigation...");
    await save();
    if (store.dirty) return;
  }
  await load(stage, page);
}

function recordsForList() {
  const result = [];
  for (const artifact of SCREEN[store.stage].artifacts) {
    for (const record of store.records[artifact] || []) result.push({ artifact, record });
  }
  return result;
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
  const filter = byId("record-filter").value.trim().toLowerCase();
  target.replaceChildren();
  for (const { artifact, record } of recordsForList()) {
    const displayRecord = artifact === "placements"
      ? { ...record, englishText: (store.records.translations || []).find((item) => item.boxno === record.boxno)?.englishText || "" }
      : record;
    const label = recordLabel(store.stage, displayRecord);
    if (filter && !label.toLowerCase().includes(filter)) continue;
    const row = document.createElement("button");
    row.type = "button";
    row.className = "record-row";
    row.classList.toggle("active", store.selection?.recordId === record.recordId);
    row.textContent = label;
    const meta = document.createElement("span");
    meta.className = "record-meta";
    const protection = store.document?.protection?.recordStates?.[record.recordId];
    meta.textContent = `${artifact.replace("ocr_", "").replace("placements", "placement")}${protection ? ` · ${protection.protected ? "protected" : "unprotected"} ${protection.kind}` : ""}`;
    row.append(meta);
    if (store.stage === "ocr" && artifact === "ocr_raw") {
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = mergeSelection.has(record.recordId);
      checkbox.title = "Select raw box for grouping";
      checkbox.addEventListener("click", (event) => {
        event.stopPropagation();
        checkbox.checked ? mergeSelection.add(record.recordId) : mergeSelection.delete(record.recordId);
      });
      row.prepend(checkbox);
    }
    row.addEventListener("click", () => store.select(artifact, record.recordId));
    target.append(row);
  }
  if (!target.childElementCount) target.textContent = "No records.";
}

function sourceRecord(record) {
  return (store.records.ocr_structured || []).find((item) => item.boxno === record?.boxno);
}

function protectionFor(recordId, field) {
  return store.document?.protection?.records?.[recordId]?.[field];
}

function renderInspector() {
  const form = byId("record-form");
  form.replaceChildren();
  const record = store.selectedRecord();
  byId("selection-label").textContent = record ? `Box ${record.boxno ?? "?"}` : "No selection";
  byId("notes-panel").hidden = store.stage !== "translation";
  if (store.document) {
    byId("job-notes").value = store.document.translationNotes?.job || "";
    byId("page-notes").value = store.document.translationNotes?.page || "";
  }
  if (!record) return;
  const formTarget = { ...store.selection, stage: store.stage };

  if (["translation", "placement"].includes(store.stage)) {
    const source = sourceRecord(record);
    if (source) {
      const label = document.createElement("label");
      label.textContent = "Source text";
      const area = document.createElement("textarea");
      area.rows = 3; area.readOnly = true; area.value = source.text || "";
      label.append(area); form.append(label);
    }
  }

  const values = {};
  for (const [name, labelText, type, value, choices] of fieldsFor(store.stage, record, store.document?.fonts || [])) {
    const wrapper = document.createElement("label");
    wrapper.className = "field-row";
    wrapper.append(document.createTextNode(labelText));
    let input;
    if (type === "textarea") {
      input = document.createElement("textarea"); input.rows = 4; input.value = value;
    } else if (type === "select") {
      input = document.createElement("select");
      for (const choice of choices || []) {
        const option = document.createElement("option"); option.value = choice; option.textContent = choice || "Auto / backup";
        input.append(option);
      }
      input.value = value;
    } else {
      input = document.createElement("input"); input.type = type;
      if (type === "checkbox") input.checked = !!value;
      else input.value = value;
      if (name === "fontSizeWidthPercent") { input.min = "0.05"; input.max = "100"; input.step = "0.05"; }
    }
    input.name = name;
    values[name] = input;
    input.addEventListener("change", () => applyInspector(values, formTarget));
    wrapper.append(input);

    const overrideField = ["left", "top", "right", "bottom"].includes(name)
      ? (store.stage === "placement" ? "placementRegion" : "region")
      : name;
    const protectedValue = protectionFor(record.recordId, overrideField);
    if (protectedValue !== undefined) {
      const controls = document.createElement("span"); controls.className = "field-protection";
      const toggle = document.createElement("button"); toggle.type = "button";
      toggle.textContent = protectedValue ? "Unprotect" : "Protect";
      toggle.dataset.tooltip = protectedValue
        ? "Allow a later model rerun to replace this saved field."
        : "Keep this saved field unchanged during later model reruns.";
      toggle.addEventListener("click", () => changeProtection("field", record, overrideField, !protectedValue));
      const revert = document.createElement("button"); revert.type = "button"; revert.textContent = "Use generated";
      revert.dataset.tooltip = "Remove this field override and restore the latest generated value.";
      revert.addEventListener("click", () => changeProtection("revert-field", record, overrideField, false));
      controls.append(toggle, revert); wrapper.append(controls);
    }
    form.append(wrapper);
  }

  if (store.document?.protection?.recordStates?.[record.recordId] || store.document?.protection?.records?.[record.recordId]) {
    const row = document.createElement("div"); row.className = "field-protection";
    const toggle = document.createElement("button"); toggle.type = "button"; toggle.textContent = "Toggle record protection";
    toggle.dataset.tooltip = "Protect or unprotect all saved changes for this record.";
    const fieldStates = Object.values(store.document?.protection?.records?.[record.recordId] || {});
    const recordState = store.document?.protection?.recordStates?.[record.recordId];
    const allProtected = (fieldStates.length > 0 || !!recordState)
      && fieldStates.every(Boolean)
      && (!recordState || recordState.protected);
    toggle.addEventListener("click", () => changeProtection("record", record, null, !allProtected));
    const revert = document.createElement("button"); revert.type = "button"; revert.textContent = "Revert record";
    revert.dataset.tooltip = "Remove all saved overrides for this record and restore its generated values.";
    revert.addEventListener("click", () => changeProtection("revert-record", record, null, false));
    row.append(toggle, revert); form.append(row);
  }
}

function applyInspector(inputs, target) {
  const record = (store.records[target.artifact] || []).find(
    (item) => item.recordId === target.recordId,
  );
  if (!record) return;
  const values = {};
  for (const [name, input] of Object.entries(inputs)) {
    values[name] = input.type === "checkbox" ? input.checked : input.value;
    if (input.type === "number" && input.value !== "") values[name] = Number(input.value);
  }
  if (target.stage === "erase") values.safeToEraseOriginal = !values.openLettering;
  const updated = applyFields(target.stage, record, values);
  store.updateRecord(target.artifact, target.recordId, updated);
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
  if (store.stage === "ocr") {
    addButton(target, "Add raw box", () => canvas.startDraw(addRawBox), "Draw a missing OCR box", "Draw one raw OCR box manually.", "z");
    addButton(target, ocrAreaRunning ? "OCR running..." : "OCR area", () => canvas.startDraw(ocrArea), "OCR a selected region", "Draw a region to scan only that area. The result adds raw OCR boxes and merges the new boxes with one another.", "x");
    addButton(target, "Group checked", groupChecked, "Merge checked raw boxes", "Create one merged OCR group from the checked raw boxes. Those boxes are removed from their existing groups.", "c");
    addButton(target, "Split group", splitGroup, "Split the selected merged group", "Delete the selected merged group and create one separate group for each of its source OCR boxes.", "v");
    addButton(target, "Delete", deleteSelected, "Delete selected", "Delete the selected raw box or merged group.", "b");
  } else if (store.stage === "structure") {
    addButton(target, "Move earlier", () => moveSelected(-1), "Move earlier in reading order", "Move the selected record one position earlier in the page reading order. The translation pass follows this order.", "z");
    addButton(target, "Move later", () => moveSelected(1), "Move later in reading order", "Move the selected record one position later in the page reading order. The translation pass follows this order.", "x");
    addButton(target, "Reject", deleteSelected, "Reject this OCR group", "Remove the selected structured record from translation, typesetting, and rendering. Use this only for OCR that is not real text.", "c");
  } else if (store.stage === "translation") {
    addButton(
      target,
      queuedAction ? "Translation request running..." : "Retranslate page",
      () => retranslate("page"),
      "Request a new VLM translation for this page",
      "Replace all translations on this page with a new VLM translation, then rerun Typesetting and Render.",
      "z",
    );
    addButton(
      target,
      queuedAction ? "Translation request running..." : "Retranslate selected",
      () => retranslate("record"),
      "Request a new VLM translation for the selected record",
      "Replace only the selected translation with a new VLM translation. Other records remain unchanged.",
      "x",
    );
  } else if (store.stage === "placement") {
    const views = [
      ["current", "Current", "Show the exact current draft. This uses the saved render when it is current, or creates a cached ImageMagick preview when needed."],
      ["original", "Original", "Show the original source page without translated text."],
      ["preview", "Preview", "Show the cleaned page with the fast editable text preview."],
      ["render", "Render", "Show the last page produced by the pipeline Render pass."],
    ];
    const shortcuts = ["z", "x", "c", "v"];
    for (const [index, [view, label, tooltip]] of views.entries()) {
      const button = addButton(target, label, () => selectTypesettingView(view), "", tooltip, shortcuts[index]);
      button.classList.toggle("active", typesettingView === view);
    }
  }
  if (ocrAreaRunning) {
    for (const button of target.querySelectorAll("button")) button.disabled = true;
    const progress = document.createElement("span");
    progress.className = "tool-progress";
    progress.setAttribute("role", "status");
    progress.textContent = "OCR is processing the selected area.";
    target.append(progress);
  } else if (queuedAction && store.stage === "translation") {
    for (const button of target.querySelectorAll("button")) button.disabled = true;
    const progress = document.createElement("span");
    progress.className = "tool-progress";
    progress.setAttribute("role", "status");
    progress.textContent = "The translation request is being added to the job queue.";
    target.append(progress);
  }
}

function addRawBox(region) {
  store.mutate(() => {
    const records = store.records.ocr_raw || (store.records.ocr_raw = []);
    const record = { page: store.page, boxno: nextBoxno(records), recordId: newRecordId("raw"), region, text: "", ocrSource: "manual" };
    records.push(record); store.selection = { artifact: "ocr_raw", recordId: record.recordId };
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
    store.records.ocr_merged = (store.records.ocr_merged || []).filter((group) => {
      group.sourceBoxnos = (group.sourceBoxnos || []).filter((boxno) => !selectedBoxnos.has(boxno));
      return group.sourceBoxnos.length;
    });
    const groups = store.records.ocr_merged;
    const record = { page: store.page, boxno: nextBoxno(groups), recordId: newRecordId("group"), sourceBoxnos: raw.map((item) => item.boxno), sourceTexts: raw.map((item) => item.text || ""), text: raw.map((item) => item.text || "").join(""), region: unionRegion(raw) };
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
  if (["translation", "placement", "erase"].includes(store.stage)) return setStatus("Delete the source record from Structure.", true);
  if (store.stage === "ocr" && store.selection.artifact === "ocr_raw") {
    const raw = store.selectedRecord();
    store.mutate(() => {
      store.records.ocr_raw = (store.records.ocr_raw || []).filter((item) => item.recordId !== raw.recordId);
      store.records.ocr_merged = (store.records.ocr_merged || []).filter((group) => {
        group.sourceBoxnos = (group.sourceBoxnos || []).filter((boxno) => boxno !== raw.boxno);
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
    records.forEach((item, boxno) => { item.boxno = boxno; });
  });
}

async function save() {
  if (!store.document || !store.dirty) return;
  byId("save").disabled = true; byId("save-state").textContent = "Saving...";
  try {
    const document = await api.save(store.stage, store.page, {
      baseRevision: store.document.revision,
      recordsByArtifact: Object.fromEntries(
        SCREEN[store.stage].artifacts.map((artifact) => [artifact, store.records[artifact] || []]),
      ),
      translationNotes: store.stage === "translation" ? { job: byId("job-notes").value, page: byId("page-notes").value } : undefined,
    });
    store.markSaved(document); setStatus("Saved. Changed fields are protected.");
  } catch (error) { setStatus(error.message, true); }
  finally { byId("save").disabled = false; }
}

async function changeProtection(action, record, field, protectedValue) {
  if (store.dirty) { setStatus("Save the current draft before changing protection.", true); return; }
  try {
    const document = await api.protect(store.stage, store.page, {
      baseRevision: store.document.revision, action, artifact: selectedArtifact(), recordId: record.recordId, field, protected: protectedValue,
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

async function regenerate() {
  if (queuedAction) return;
  if (store.dirty) await save();
  if (store.dirty) return;
  setQueuedAction("downstream", "Adding this page regeneration to the job queue...");
  try {
    const result = await api.regenerate(store.stage, store.page);
    window.location.href = result.url;
  } catch (error) { setQueuedAction(""); setStatus(error.message, true); }
}

async function regenerateAll() {
  if (queuedAction) return;
  if (store.dirty) await save();
  if (store.dirty) return;
  setQueuedAction("all", "Adding all changed pages to the job queue...");
  try {
    const result = await api.regenerateChanges();
    window.location.href = result.url;
  } catch (error) { setQueuedAction(""); setStatus(error.message, true); }
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
  if (queuedAction || store.stage !== "translation") return;
  const selected = scope === "record" ? store.selectedRecord() : null;
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
  try {
    const placementState = store.document?.stageStates?.placement || {};
    const renderIsCurrent = !store.dirty && !placementState.pending && !placementState.stale;
    if (renderIsCurrent) {
      canvas.load(imageUrl("final"), { showPlacementText: false });
      setStatus("Saved render loaded. No preview generation was needed.");
      return;
    }
    setStatus("Checking the cleaning mask and preparing Current...");
    const result = await api.preview(
      "placement",
      store.page,
      store.records.placements || [],
      store.records.ocr_structured || [],
    );
    canvas.load(result.url, { showPlacementText: false });
    if (result.cleanedWithLama) {
      setStatus("Current was cleaned with LaMa and rendered with ImageMagick.");
    } else {
      setStatus(result.cached ? "Cached Current preview loaded." : "Current preview rendered with ImageMagick.");
    }
  } catch (error) { setStatus(error.message, true); }
}

async function loadRevisions() {
  const target = byId("revision-list"); target.textContent = "Loading...";
  try {
    const data = await api.revisions(); target.replaceChildren();
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
  } catch (error) { target.textContent = error.message; }
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
  byId("regenerate").dataset.tooltip = downstreamRegenerationTooltip();
}

store.addEventListener("change", render);
document.querySelectorAll(".stage-tab").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.stage, store.page)));
byId("page-select").addEventListener("change", (event) => navigate(store.stage, Number(event.target.value)));
byId("prev-page").addEventListener("click", () => navigate(store.stage, store.page - 1));
byId("next-page").addEventListener("click", () => navigate(store.stage, store.page + 1));
byId("record-filter").addEventListener("input", renderRecordList);
byId("undo").addEventListener("click", () => store.undo());
byId("redo").addEventListener("click", () => store.redo());
byId("save").addEventListener("click", save);
byId("freeze-stage").addEventListener("click", () => toggleFreeze("stage"));
byId("freeze-page").addEventListener("click", () => toggleFreeze("page"));
byId("regenerate").addEventListener("click", regenerate);
byId("regenerate-all").addEventListener("click", regenerateAll);
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
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName);
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
    event.preventDefault(); navigate(store.stage, store.page - 1); return;
  }
  if (key === "d") {
    event.preventDefault(); navigate(store.stage, store.page + 1); return;
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
