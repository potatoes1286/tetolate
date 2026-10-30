// Run with: node tests/test_ocr_editor.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");
const root = path.join(__dirname, "../web_editor");
const app = fs.readFileSync(path.join(root, "app.js"), "utf8");
const screens = vm.runInNewContext(fs.readFileSync(path.join(root, "screens.js"), "utf8").replaceAll("export ", "") + "\n({SCREEN, recordLabel})");
const canvasSource = fs.readFileSync(path.join(root, "canvas.js"), "utf8").replace(/^import .*\n/, "").replace("export class", "class") + "\nEditorCanvas";
const functionSource = (name) => app.match(new RegExp(`function ${name}\\([^]*?\\n\\}`))[0];

function setup() {
  const storage = new Map(), handlers = {}, mergeSelection = new Set(), status = [];
  const Store = vm.runInNewContext(fs.readFileSync(path.join(root, "store.js"), "utf8").replace("export class", "class") + "\nEditorStore", {
    EventTarget, Event, structuredClone,
    localStorage: {getItem: (key) => storage.get(key), setItem: (key, value) => storage.set(key, value)},
  });
  const store = new Store({category: "default", jobId: "test"});
  const raw = [0, 1, 2].map((boxno) => ({boxno, recordId: `raw-${boxno}`, text: String(boxno),
    region: [10 + boxno * 50, 10, 40 + boxno * 50, 40]}));
  store.setDocument({stage: "ocr", page: 0, revision: 0, recordsByArtifact: {ocr_raw: raw,
    ocr_merged: raw.map((item) => ({...item, recordId: `group-${item.boxno}`, sourceBoxnos: [item.boxno], sourceTexts: [item.text]}))}});
  let id = 0;
  const context = vm.createContext({store, mergeSelection, newRecordId: (prefix) => `${prefix}-manual-${++id}`,
    setStatus: (...args) => status.push(args), SCREEN: screens.SCREEN, recordLabel: screens.recordLabel});
  for (const name of ["nextBoxno", "rawBoxesFor", "ocrBoxesChecked", "selectOcrBoxes", "addRawBox", "unionRegion", "groupChecked", "splitGroup", "deleteSelected", "recordsForList"])
    vm.runInContext(functionSource(name), context);
  const canvas = {width: 200, height: 200, style: {}, getContext: () => ({}),
    getBoundingClientRect: () => ({left: 0, top: 0, width: 200, height: 200}),
    addEventListener: (name, handler) => { handlers[name] = handler; }, setPointerCapture() {}};
  const Canvas = vm.runInNewContext(canvasSource, {SCREEN: screens.SCREEN, Image: class {}});
  const editor = new Canvas(canvas, store, () => {}, null, {select: context.selectOcrBoxes, has: context.ocrBoxesChecked});
  // Pointer tests need selection events but do not need to paint a page.
  editor.draw = () => {};
  const click = (x, y, modifiers = {}) => {
    const event = {clientX: x, clientY: y, pointerId: 1, preventDefault() {}, ...modifiers};
    handlers.pointerdown(event); handlers.pointerup(event);
  };
  const doubleClick = (x, y) => {
    click(x, y); click(x, y);
    handlers.dblclick({clientX: x, clientY: y, preventDefault() {}});
  };
  return {store, context, editor, mergeSelection, status, click, doubleClick, handlers};
}

test("new OCR boxes use reserved numbers across deletion, undo and recovered drafts", () => {
  const page = setup();
  page.store.document.nextBoxnos = {ocr_raw: 10, ocr_merged: 20};
  page.context.addRawBox([10, 70, 30, 90]);
  assert.equal(page.store.records.ocr_raw.at(-1).boxno, 10);
  assert.equal(page.store.records.ocr_merged.at(-1).boxno, 20);
  page.store.undo();
  page.context.addRawBox([10, 80, 30, 100]);
  assert.equal(page.store.records.ocr_raw.at(-1).boxno, 11);
  page.store.undo();
  const recovered = new page.store.constructor(page.store.config);
  recovered.setDocument(structuredClone(page.store.document));
  assert.equal(recovered.dirty, true);
  const context = vm.createContext({store: recovered});
  vm.runInContext(functionSource("nextBoxno"), context);
  assert.equal(context.nextBoxno(recovered.records.ocr_raw, "ocr_raw"), 12);
  assert.equal(context.nextBoxno(recovered.records.ocr_merged, "ocr_merged"), 22);
});

test("deleting part of an OCR group rebuilds its text and region, and undo restores it", () => {
  const page = setup();
  page.mergeSelection.add("raw-0"); page.mergeSelection.add("raw-1");
  page.context.groupChecked();
  const group = page.store.records.ocr_merged.at(-1);
  page.store.select("ocr_raw", "raw-0");
  page.context.deleteSelected();
  const remaining = page.store.records.ocr_merged.find((item) => item.recordId === group.recordId);
  assert.deepEqual(Array.from(remaining.sourceBoxnos), [1]);
  assert.deepEqual(Array.from(remaining.sourceTexts), ["1"]);
  assert.equal(remaining.text, "1");
  assert.deepEqual(Array.from(remaining.region), [60, 10, 90, 40]);
  page.store.undo();
  const restored = page.store.records.ocr_merged.find((item) => item.recordId === group.recordId);
  assert.equal(restored.text, "01");
  assert.deepEqual(Array.from(restored.region), [10, 10, 90, 40]);
});

test("new manual boxes get a singleton group in the same undoable edit", () => {
  const page = setup();
  page.context.addRawBox([160, 80, 190, 120]);
  const raw = page.store.records.ocr_raw.at(-1), group = page.store.records.ocr_merged.at(-1);
  assert.deepEqual(Array.from(group.sourceBoxnos), [raw.boxno]);
  assert.deepEqual(Array.from(group.region), Array.from(raw.region));
  assert.equal(page.mergeSelection.has(raw.recordId), true);
  page.store.undo();
  assert.equal(page.store.records.ocr_raw.length, 3);
  assert.equal(page.store.records.ocr_merged.length, 3);
  page.store.redo();
  assert.equal(page.store.records.ocr_raw.length, 4);
  assert.equal(page.store.records.ocr_merged.length, 4);
});

test("Ctrl, Shift and Meta select raw boxes through overlapping groups without dragging", () => {
  for (const modifier of ["ctrlKey", "shiftKey", "metaKey"]) {
    const page = setup();
    page.click(20, 20);
    page.click(70, 20, {[modifier]: true});
    assert.deepEqual([...page.mergeSelection], ["raw-0", "raw-1"]);
    assert.equal(page.store.selection.artifact, "ocr_raw");
    assert.equal(page.editor.drag, null);
    assert.equal(page.store.dirty, false);
    page.context.groupChecked();
    const group = page.store.records.ocr_merged.at(-1);
    assert.deepEqual(group.sourceBoxnos, [0, 1]);
    assert.deepEqual(Array.from(group.region), [10, 10, 90, 40]);
    assert.ok(page.store.records.ocr_merged.some((item) => item.sourceBoxnos.length === 1 && item.sourceBoxnos[0] === 2));
    assert.equal(page.mergeSelection.size, 0);
  }
});

test("modifier click toggles a checked box off and ordinary click replaces the set", () => {
  const page = setup();
  page.click(20, 20); page.click(70, 20, {ctrlKey: true}); page.click(20, 20, {ctrlKey: true});
  assert.deepEqual([...page.mergeSelection], ["raw-1"]);
  page.click(120, 20);
  assert.deepEqual([...page.mergeSelection], ["raw-2"]);
  page.click(190, 190);
  assert.equal(page.mergeSelection.size, 0);
  assert.equal(page.store.selection, null);
});

test("double-click cycles through all overlaps, including later double-clicks", () => {
  const page = setup();
  page.store.records.ocr_merged.push({...page.store.records.ocr_merged[0], recordId: "group-extra"});
  const expected = ["group-extra", "raw-0", "group-0", "group-extra"];
  page.click(20, 20);
  for (const recordId of expected) {
    page.doubleClick(20, 20);
    assert.equal(page.store.selection.recordId, recordId);
  }
  assert.equal(page.store.history.length, 0, "selection must not move boxes or create edit history");
});

test("grouping part of an existing group recomputes the remainder without reusing IDs", () => {
  const page = setup();
  page.store.records.ocr_merged = [{boxno: 10, recordId: "all", sourceBoxnos: [0, 1, 2], region: [10, 10, 140, 40]}];
  page.mergeSelection.add("raw-0"); page.mergeSelection.add("raw-1");
  page.context.groupChecked();
  assert.deepEqual(page.store.records.ocr_merged[0].sourceBoxnos, [2]);
  assert.deepEqual(Array.from(page.store.records.ocr_merged[0].region), [110, 10, 140, 40]);
  assert.equal(page.store.records.ocr_merged[0].text, "2");
  assert.equal(page.store.records.ocr_merged[1].boxno, 11);
});

test("record checkboxes are outside buttons and feed the same grouping set", () => {
  const page = setup(), elements = {};
  const document = {activeElement: null, createElement: (tag) => new Element(tag)};
  class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.listeners = {}; this.dataset = {}; this.classList = {toggle() {}}; }
    append(...items) { this.children.push(...items); }
    prepend(item) { this.children.unshift(item); }
    replaceChildren() { this.children = []; }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    querySelectorAll() { return this.children.flatMap((child) => [child, ...child.querySelectorAll()]); }
    focus() { document.activeElement = this; }
    get childElementCount() { return this.children.length; }
  }
  Object.assign(page.context, {byId: (id) => elements[id] ||= id === "record-filter" ? {value: ""} : new Element("div"),
    document});
  vm.runInContext(functionSource("renderRecordList") + "\nrenderRecordList()", page.context);
  const rows = elements["record-list"].children.filter((row) => row.children.some((child) => child.type === "checkbox"));
  assert.equal(rows.length, 3);
  for (const row of rows.slice(0, 2)) {
    assert.equal(row.tag, "div");
    const checkbox = row.children.find((child) => child.type === "checkbox");
    document.activeElement = checkbox;
    page.store.addEventListener("change", () => vm.runInContext("renderRecordList()", page.context), {once: true});
    checkbox.checked = true; checkbox.listeners.click({stopPropagation() {}});
    assert.equal(document.activeElement.type, "checkbox");
    assert.equal(document.activeElement.dataset.recordId, checkbox.dataset.recordId);
    assert.notEqual(document.activeElement, checkbox, "keyboard focus follows the rebuilt checkbox");
  }
  assert.deepEqual([...page.mergeSelection], ["raw-0", "raw-1"]);
  page.context.groupChecked();
  assert.deepEqual(page.store.records.ocr_merged.at(-1).sourceBoxnos, [0, 1]);
});

test("busy OCR blocks selection and double-click changes", () => {
  const page = setup(); page.editor.busy = true;
  page.click(20, 20); page.doubleClick(20, 20);
  assert.equal(page.store.selection, null);
  assert.equal(page.mergeSelection.size, 0);
});

test("group deletion records explicit intent in drafts and undo/redo", () => {
  const page = setup(); page.click(20, 20);
  page.store.deleteSelected();
  assert.deepEqual(Array.from(page.store.records.ocr_removed_groups), ["group-0"]);
  page.store.undo();
  assert.equal(page.store.records.ocr_removed_groups, undefined);
  page.store.redo();
  assert.deepEqual(Array.from(page.store.records.ocr_removed_groups), ["group-0"]);
  assert.equal(page.store.records.ocr_merged.some((record) => record.recordId === "group-0"), false);
});
