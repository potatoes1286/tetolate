// Run with: node tests/test_clean_boxes.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");
const root = path.join(__dirname, "../web_editor");
const read = (name) => fs.readFileSync(path.join(root, name), "utf8");
const app = read("app.js");
const fn = (name) => app.match(new RegExp(`(?:async )?function ${name}\\([^]*?\\n\\}`))[0];
const screens = vm.runInNewContext(read("screens.js").replaceAll("export ", "") + "\n({SCREEN, fieldsFor, applyFields, recordLabel})");

function setup() {
  const storage = new Map(), handlers = {}, status = [], layers = [], labels = [];
  const localStorage = {getItem: (key) => storage.get(key), setItem: (key, value) => storage.set(key, value), removeItem: (key) => storage.delete(key)};
  const Store = vm.runInNewContext(read("store.js").replace("export class", "class") + "\nEditorStore", {EventTarget, Event, structuredClone, localStorage});
  const store = new Store({category: "default", jobId: "test"});
  const document = {stage: "placement", page: 0, revision: 0, recordsByArtifact: {
    placements: [{recordId: "text-one", boxno: 0, placementRegion: [10, 10, 40, 40]}],
    translations: [{boxno: 0, englishText: "Existing text"}], clean_boxes: [], ocr_structured: [],
  }};
  store.setDocument(document);
  let id = 0;
  const context = vm.createContext({store, SCREEN: screens.SCREEN, newRecordId: (prefix) => `${prefix}-${++id}`,
    setStatus: (...args) => status.push(args), selectedArtifact: () => store.selection?.artifact,
    applyFields: screens.applyFields});
  for (const name of ["nextBoxno", "addCleanBox", "deleteSelected", "applyInspector", "recordsForList"])
    vm.runInContext(fn(name), context);
  const ctx = {clearRect() {}, drawImage() {}, strokeRect() {}, fillRect() {}, fillText: (text) => labels.push(text)};
  const element = {width: 200, height: 200, style: {}, getContext: () => ctx,
    getBoundingClientRect: () => ({left: 0, top: 0, width: 200, height: 200}),
    addEventListener: (name, handler) => { handlers[name] = handler; }, setPointerCapture() {}};
  const Canvas = vm.runInNewContext(read("canvas.js").replace(/^import .*\n/, "").replace("export class", "class") + "\nEditorCanvas", {
    SCREEN: screens.SCREEN, Image: class {}, setTimeout, clearTimeout,
  });
  const canvas = new Canvas(element, store, () => {}, async (_page, entries) => { layers.push(entries); return entries.map(() => ({width: 20, height: 20, close() {}})); });
  const event = (x, y) => ({clientX: x, clientY: y, pointerId: 1, preventDefault() {}});
  const drag = (from, to) => { handlers.pointerdown(event(...from)); handlers.pointermove(event(...to)); handlers.pointerup(event(...to)); };
  return {store, context, canvas, drag, storage, localStorage, document, Store, status, layers, labels};
}

test("drawing a clean box creates no OCR, translation or placement record", () => {
  const page = setup(), before = JSON.stringify(page.store.records);
  page.canvas.startDraw(page.context.addCleanBox);
  page.drag([50, 50], [90, 100]);
  assert.equal(page.store.selection.artifact, "clean_boxes");
  assert.equal(page.store.records.placements.length, 1);
  assert.equal(page.store.records.translations.length, 1);
  assert.equal(page.store.records.ocr_structured.length, 0);
  const box = page.store.records.clean_boxes[0];
  assert.deepEqual(Array.from(box.region), [50, 50, 90, 100]);
  assert.deepEqual(Object.keys(box).sort(), ["boxno", "page", "recordId", "region"]);
  page.store.undo(); assert.equal(JSON.stringify(page.store.records), before);
  page.store.redo(); assert.equal(page.store.records.clean_boxes.length, 1);
});

test("clean box drafts survive reload with the same identity", () => {
  const page = setup(); page.context.addCleanBox([50, 50, 90, 100]);
  const restored = new page.Store({category: "default", jobId: "test"});
  restored.setDocument(page.document);
  assert.equal(restored.records.clean_boxes[0].recordId, page.store.records.clean_boxes[0].recordId);
  assert.equal(restored.dirty, true);
});

test("clean box move and resize use region and do not move text", () => {
  const page = setup(); page.context.addCleanBox([50, 50, 90, 100]);
  page.drag([60, 60], [70, 70]);
  let box = page.store.records.clean_boxes[0];
  assert.deepEqual(Array.from(box.region), [60, 60, 100, 110]);
  assert.equal(box.placementRegion, undefined);
  page.drag([100, 110], [120, 135]);
  assert.deepEqual(Array.from(box.region), [60, 60, 120, 135]);
  assert.deepEqual(page.store.records.placements[0].placementRegion, [10, 10, 40, 40]);
  page.store.undo(); assert.deepEqual(Array.from(page.store.records.clean_boxes[0].region), [60, 60, 100, 110]);
});

test("clean inspector has only coordinates and edits region rather than placementRegion", () => {
  const page = setup(); page.context.addCleanBox([50, 50, 90, 100]);
  const box = page.store.selectedRecord();
  assert.deepEqual(Array.from(screens.fieldsFor("placement", box, [], "clean_boxes"), (field) => field[0]), ["left", "top", "right", "bottom"]);
  assert.equal(screens.recordLabel("placement", box, "clean_boxes"), "0  Clean only");
  page.context.applyInspector({left: {type: "number", value: "55"}}, {...page.store.selection, stage: "placement"}, "left");
  assert.deepEqual(Array.from(page.store.records.clean_boxes[0].region), [55, 50, 90, 100]);
  assert.equal(page.store.records.clean_boxes[0].placementRegion, undefined);
});

test("delete removes only clean boxes and remains undoable", () => {
  const page = setup(); page.context.addCleanBox([50, 50, 90, 100]);
  page.context.deleteSelected(); assert.equal(page.store.records.clean_boxes.length, 0);
  page.store.undo(); assert.equal(page.store.records.clean_boxes.length, 1);
  page.store.select("placements", "text-one"); page.context.deleteSelected();
  assert.equal(page.store.records.placements.length, 1);
  assert.match(page.status.at(-1)[0], /Select a clean box/);
});

test("clean boxes are drawn and listed but never request text layers", async () => {
  const page = setup(); page.context.addCleanBox([50, 50, 90, 100]);
  assert.ok(page.context.recordsForList().some((item) => item.artifact === "clean_boxes"));
  page.store.records.placements = [];
  page.canvas.image.complete = true; page.canvas.draw();
  assert.ok(page.labels.includes("C0"));
  assert.equal(page.canvas.layers.size, 0);
  assert.equal(page.canvas.layerTimer, null);
  assert.equal(page.layers.length, 0);
});

test("Current and Preview send clean boxes while Preview keeps cached text layers", async () => {
  const page = setup(); page.context.addCleanBox([50, 50, 90, 100]);
  const requests = [], loads = [], shown = [];
  Object.assign(page.context, {typesettingView: "preview", AbortController, currentPreviewRequest: null,
    preview: {hide() {}, show: () => shown.push(true)},
    api: {preview: async (...args) => { requests.push(args); return {url: "clean-preview", cleanedWithLama: true}; }},
    canvas: {load: (...args) => loads.push(args)}, imageUrl: () => "original"});
  vm.runInContext(fn("exactPreview") + "\n" + fn("loadTypesettingView"), page.context);
  await page.context.loadTypesettingView();
  assert.equal(shown.length, 1);
  assert.equal(requests.length, 0, "Preview uses its independent background controller");
  page.context.typesettingView = "current";
  await page.context.exactPreview();
  assert.equal(requests[0][2].length, 1);
  assert.equal(requests[0][4].length, 1);
  assert.equal(loads[0][1].showPlacementText, false);
});

test("save and preview API serialize clean boxes separately from text", async () => {
  const requests = [];
  const Api = vm.runInNewContext(read("api.js").replace("export class", "class") + "\nEditorApi", {
    fetch: async (url, options) => { requests.push([url, JSON.parse(options.body)]); return {ok: true, json: async () => ({ok: true})}; },
  });
  const page = setup(); page.context.addCleanBox([50, 50, 90, 100]);
  const api = new Api("default", "test");
  await api.preview("placement", 0, [], [], page.store.records.clean_boxes);
  assert.equal(requests[0][1].cleanBoxes.length, 1);
  assert.deepEqual(requests[0][1].records, []);
  Object.assign(page.context, {api: {save: async (_stage, _page, body) => {
    assert.equal(body.recordsByArtifact.clean_boxes.length, 1);
    assert.equal(body.recordsByArtifact.placements.length, 1);
    return page.document;
  }}, resetting: false, byId: () => ({}), loadTypesettingView: async () => {}});
  vm.runInContext(fn("save"), page.context);
  await page.context.save();
  assert.equal(page.store.dirty, false);
});

test("Last Render and current saved output never start cleaning", async () => {
  const page = setup(), requests = [], loads = [];
  const before = JSON.stringify(page.store.snapshot());
  Object.assign(page.context, {typesettingView: "render", AbortController, currentPreviewRequest: null,
    preview: {hide() {}, show() {}}, api: {preview: async (...args) => {
    requests.push(args); return {url: "fresh-render", cached: false, cleanedWithLama: true};
  }}, canvas: {load: (...args) => loads.push(args)}, imageUrl: () => "last-render"});
  vm.runInContext(fn("exactPreview") + "\n" + fn("loadTypesettingView"), page.context);
  await page.context.loadTypesettingView();
  assert.equal(requests.length, 0);
  assert.equal(loads[0][0], "last-render");
  assert.equal(loads[0][1].showPlacementText, false);
  assert.equal(JSON.stringify(page.store.snapshot()), before);
  assert.equal(page.store.dirty, false);
  page.context.typesettingView = "render";
  await page.context.loadTypesettingView();
  assert.equal(requests.length, 0, "Last Render does not run anything");
  assert.equal(loads.at(-1)[0], "last-render");
  page.context.typesettingView = "current";
  await page.context.loadTypesettingView();
  assert.equal(requests.length, 0, "Current keeps its saved-render fast path");
});

test("view controls include Last Render, with no Re-render or clean delete buttons", () => {
  const page = setup(), buttons = [];
  Object.assign(page.context, {typesettingView: "current", hidePreviewBoxes: false, canvas: page.canvas, ocrAreaRunning: false, queuedAction: "",
    byId: () => ({replaceChildren() {}}), selectTypesettingView() {},
    addButton: (_target, text, _handler, _title, _tooltip, shortcut) => {
      buttons.push({text, shortcut}); return {classList: {toggle() {}}};
    }});
  vm.runInContext(fn("renderTools"), page.context);
  page.context.renderTools();
  assert.deepEqual(buttons.map((button) => button.text), ["Current", "Original", "Preview", "Last Render", "Add clean box"]);
  assert.equal(new Set(buttons.map((button) => button.shortcut)).size, buttons.length);
  assert.doesNotMatch(app, /addButton\([^\n]*"Delete clean box"/);
});

test("Preview Hide boxes toggle is reversible and is not applied to other views or stages", () => {
  const page = setup();
  const element = () => ({children: [], listeners: {}, dataset: {}, append(...children) {this.children.push(...children);},
    replaceChildren() {this.children = [];}, addEventListener(name, callback) {this.listeners[name] = callback;}});
  const target = element();
  Object.assign(page.context, {typesettingView: "preview", hidePreviewBoxes: false, canvas: page.canvas,
    groupChecked() {}, splitGroup() {},
    ocrAreaRunning: false, queuedAction: "", byId: () => target,
    document: {createElement: element, createTextNode: (text) => ({text})},
    addButton: () => ({classList: {toggle() {}}})});
  vm.runInContext(fn("renderTools"), page.context);
  const before = JSON.stringify(page.store.records);
  page.context.renderTools();
  const toggle = target.children[0].children[0];
  assert.equal(toggle.type, "checkbox");
  assert.equal(toggle.checked, false);
  toggle.checked = true; toggle.listeners.change();
  assert.equal(page.canvas.hideBoxes, true);
  page.context.renderTools();
  assert.equal(target.children[0].children[0].checked, true);
  page.context.typesettingView = "current"; page.context.renderTools();
  assert.equal(target.children.length, 0);
  assert.equal(page.canvas.hideBoxes, false);
  page.context.typesettingView = "preview"; page.context.renderTools();
  assert.equal(page.canvas.hideBoxes, true);
  page.store.stage = "ocr"; page.context.renderTools();
  assert.equal(page.canvas.hideBoxes, false);
  assert.equal(JSON.stringify(page.store.records), before);
});

test("H activates only the Preview Hide boxes checkbox and never interrupts typing", () => {
  const page = setup();
  let keydown, visible = true, clicks = 0, prevented = 0;
  const checkbox = {disabled: false, click() { clicks++; }};
  const target = {querySelector: (selector) => visible && selector === '[data-shortcut="h"]' ? checkbox : null};
  Object.assign(page.context, {resetting: false, ocrAreaRunning: false,
    AVAILABLE_STAGES: ["ocr", "text", "placement"], byId: () => target,
    document: {addEventListener: (_event, callback) => { keydown = callback; }}});
  vm.runInContext(app.slice(app.indexOf('document.addEventListener("keydown",'), app.indexOf('window.addEventListener("beforeunload",')), page.context);
  const press = (target = {tagName: "DIV"}, changes = {}) => keydown({key: "h", target,
    preventDefault() {prevented++;}, ...changes});
  press(); press({tagName: "DIV"}, {key: "H"});
  assert.equal(clicks, 2); assert.equal(prevented, 2);
  for (const tagName of ["INPUT", "TEXTAREA", "SELECT"]) press({tagName});
  press({tagName: "DIV", isContentEditable: true});
  for (const modifier of ["ctrlKey", "altKey", "metaKey"]) press(undefined, {[modifier]: true});
  assert.equal(clicks, 2); assert.equal(prevented, 2);
  press({tagName: "INPUT", type: "checkbox"});
  assert.equal(clicks, 3);
  visible = false; press(); assert.equal(clicks, 3);
  assert.match(app, /toggle\.dataset\.shortcut = "h"/);
  assert.match(app, /Hide boxes \[H\]/);
});

test("Delete key still removes clean boxes, but not while typing", () => {
  const page = setup(); page.context.addCleanBox([50, 50, 90, 100]);
  let keydown;
  Object.assign(page.context, {resetting: false, ocrAreaRunning: false, AVAILABLE_STAGES: ["ocr", "text", "placement"],
    document: {addEventListener: (_event, callback) => { keydown = callback; }}});
  const handler = app.slice(app.indexOf('document.addEventListener("keydown",'), app.indexOf('window.addEventListener("beforeunload",'));
  vm.runInContext(handler, page.context);
  keydown({target: {tagName: "INPUT"}, key: "Delete", preventDefault() { assert.fail("typing must not delete a box"); }});
  assert.equal(page.store.records.clean_boxes.length, 1);
  let prevented = false;
  keydown({target: {tagName: "DIV"}, key: "Delete", preventDefault() { prevented = true; }});
  assert.equal(prevented, true);
  assert.equal(page.store.records.clean_boxes.length, 0);
  page.store.undo(); assert.equal(page.store.records.clean_boxes.length, 1);
});

test("background API sends only cleaning records and an abort signal", async () => {
  let body, url, signal;
  const Api = vm.runInNewContext(read("api.js").replace("export class", "class") + "\nEditorApi", {
    fetch: async (target, options) => { url = target; signal = options.signal; body = JSON.parse(options.body); return {ok: true, json: async () => ({ok: true})}; },
  });
  const controller = new AbortController();
  await new Api("default", "test").cleanPreview(0, [], [], controller.signal);
  assert.match(url, /pages\/0\/clean-preview$/);
  assert.deepEqual(Object.keys(body).sort(), ["cleanBoxes", "cleanRecords"]);
  assert.equal(signal, controller.signal);
});
