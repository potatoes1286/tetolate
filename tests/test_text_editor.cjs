// Run with: node tests/test_text_editor.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");
const root = path.join(__dirname, "../web_editor");
const source = fs.readFileSync(path.join(root, "screens.js"), "utf8");
const screens = vm.runInNewContext(source.replaceAll("export ", "") +
  "\n({STAGES, SCREEN, fieldsFor, applyFields, textFieldTarget, updateTextField, recordLabel})");
const app = fs.readFileSync(path.join(root, "app.js"), "utf8");

function records() {
  return {ocr_structured: [{recordId: "source-0", boxno: 0, text: "Source", openLettering: false, safeToEraseOriginal: true},
                          {recordId: "source-1", boxno: 1, text: "Second"}],
    translations: [{recordId: "translation-0", boxno: 0, englishText: "English"}],
    alt_placement: [{recordId: "alt-0", boxno: 0}], placements: [{boxno: 0}]};
}

test("three tabs share a single source-record list", () => {
  assert.deepEqual(Array.from(screens.STAGES), ["ocr", "text", "placement"]);
  assert.deepEqual(Array.from(screens.SCREEN.text.artifacts), ["ocr_structured"]);
  assert.equal(screens.SCREEN.structure, undefined);
  assert.equal(screens.SCREEN.erase, undefined);
  assert.equal(screens.SCREEN.translation, undefined);
  const fields = Array.from(screens.fieldsFor("text", {}), (field) => field[0]);
  assert.deepEqual(fields, ["text", "sfx", "openLettering", "safeToEraseOriginal", "altPlacementReason", "englishText"]);
});

test("English edits only change the associated translation", () => {
  const data = records();
  const original = JSON.stringify(data.ocr_structured);
  screens.updateTextField(data, data.ocr_structured[0], "englishText", "Changed", 0, () => "new");
  assert.equal(data.translations[0].englishText, "Changed");
  assert.equal(JSON.stringify(data.ocr_structured), original);
  const target = screens.textFieldTarget(data, data.ocr_structured[0], "englishText");
  assert.equal(target.recordId, "translation-0");
  assert.equal(target.artifact, "translations");
  assert.equal(target.pipelineStage, "translation");
});

test("source edits do not change translation or erase fields", () => {
  const data = records();
  screens.updateTextField(data, data.ocr_structured[0], "text", "Corrected", 0, () => "new");
  assert.equal(data.ocr_structured[0].text, "Corrected");
  assert.equal(data.ocr_structured[0].openLettering, false);
  assert.equal(data.translations[0].englishText, "English");
  assert.equal(screens.textFieldTarget(data, data.ocr_structured[0], "text").pipelineStage, "structure");
});

test("erase flags remain consistent in either direction", () => {
  const data = records();
  screens.updateTextField(data, data.ocr_structured[0], "openLettering", true, 0, () => "new");
  assert.equal(data.ocr_structured[0].safeToEraseOriginal, false);
  screens.updateTextField(data, data.ocr_structured[0], "safeToEraseOriginal", true, 0, () => "new");
  assert.equal(data.ocr_structured[0].openLettering, false);
  assert.equal(screens.textFieldTarget(data, data.ocr_structured[0], "openLettering").pipelineStage, "erase");
});

test("a missing translation gets one record with a stable identity", () => {
  const data = records();
  for (const text of ["First", "Second"])
    screens.updateTextField(data, data.ocr_structured[1], "englishText", text, 0, () => "new-id");
  assert.equal(data.translations.length, 2);
  assert.equal(data.translations[1].recordId, "new-id");
  assert.equal(data.translations[1].englishText, "Second");
});

test("reordering source records does not change box numbers or translations", () => {
  const data = records();
  const move = app.match(/function moveSelected\(delta\) \{[\s\S]*?\n\}/)[0];
  const store = {records: data, selection: {artifact: "ocr_structured", recordId: "source-1"},
    mutate: (callback) => callback(data)};
  vm.runInNewContext(move + "\nmoveSelected(-1)", {store});
  assert.deepEqual(data.ocr_structured.map((record) => record.boxno), [1, 0]);
  assert.equal(data.translations[0].boxno, 0);
  assert.equal(data.translations[0].englishText, "English");
});

test("reject removes only the source, preserving descendant data for reset", () => {
  const data = records();
  const remove = app.match(/function deleteSelected\(\) \{[\s\S]*?\n\}/)[0];
  const store = {stage: "text", records: data, selection: {artifact: "ocr_structured", recordId: "source-0"},
    selectedRecord: () => data.ocr_structured[0], mutate: (callback) => callback(data)};
  vm.runInNewContext(remove + "\ndeleteSelected()", {store});
  assert.deepEqual(data.ocr_structured.map((record) => record.boxno), [1]);
  assert.equal(data.translations.length, 1);
  assert.equal(data.translations[0].englishText, "English");
  assert.equal(data.alt_placement.length, 1);
  assert.equal(data.placements.length, 1);
});

test("record labels show both source and English text", () => {
  assert.equal(screens.recordLabel("text", {boxno: 0, text: "Source", englishText: "English"}), "0  Source → English");
});

test("transforms are manual Typesetting fields, default to zero, and clear at zero or blank", () => {
  for (const field of ["rotation", "skewX", "skewY"]) {
    for (const stage of ["ocr", "text"]) assert.ok(!screens.fieldsFor(stage, {}).some(([name]) => name === field));
    assert.equal(screens.fieldsFor("placement", {}).find(([name]) => name === field)[3], 0);
    const record = screens.applyFields("placement", {}, {[field]: -12.5});
    assert.equal(record[field], -12.5);
    for (const value of [0, ""]) assert.equal(screens.applyFields("placement", record, {[field]: value})[field], 0);
  }
});

test("line spacing is a manual Typesetting field and clears to default at zero or blank", () => {
  for (const stage of ["ocr", "text"]) assert.ok(!screens.fieldsFor(stage, {}).some(([name]) => name === "lineSpacing"));
  assert.equal(screens.fieldsFor("placement", {}).find(([name]) => name === "lineSpacing")[3], 0);
  const record = screens.applyFields("placement", {}, {lineSpacing: -4});
  assert.equal(record.lineSpacing, -4);
  for (const value of [0, ""]) assert.equal(screens.applyFields("placement", record, {lineSpacing: value}).lineSpacing, 0);
});

test("OCR and Typesetting coordinates share one row with independent number inputs", () => {
  const render = app.match(/function renderInspector\(\) \{[\s\S]*?\n\}/)[0];
  class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.dataset = {}; this.listeners = {}; }
    append(...children) { for (const child of children) { child.parentNode = this; this.children.push(child); } }
    replaceChildren() { this.children = []; }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    setAttribute(name, value) { this[name] = value; }
  }
  for (const stage of ["ocr", "placement"]) {
    const elements = {};
    const byId = (id) => elements[id] ||= new Element("div");
    const record = {recordId: "box", boxno: 0, region: [262, 62, 366, 246], placementRegion: [270, 70, 370, 250]};
    const changes = [];
    vm.runInNewContext(render + "\nrenderInspector()", {
      byId, store: {stage, selection: {artifact: stage === "ocr" ? "ocr_raw" : "placements"},
        selectedRecord: () => record, document: {fonts: []}},
      document: {createElement: (tag) => new Element(tag), createTextNode: (text) => ({text})},
      fieldsFor: screens.fieldsFor, ALIGNMENT_LABELS: {}, sourceRecord: () => null,
      protectionFor: () => undefined, applyInspector: (_values, target, field) => changes.push([target.stage, field]),
      refreshLineBreaks: () => {},
    });
    const form = elements["record-form"];
    const row = form.children.find((child) => child.className === "field-grid");
    assert.ok(row, stage);
    assert.equal(row.children.length, 4);
    const inputs = row.children.map((label) => label.children.find((child) => child.tag === "input"));
    assert.deepEqual(inputs.map((input) => input.name), ["left", "top", "right", "bottom"]);
    assert.deepEqual(inputs.map((input) => input.value), stage === "ocr" ? record.region : record.placementRegion);
    for (const input of inputs) { assert.equal(input.type, "number"); input.listeners.change(); }
    assert.deepEqual(changes, ["left", "top", "right", "bottom"].map((name) => [stage, name]));
    assert.ok(form.children.some((child) => child.tag === "label"), "other fields stay outside coordinate row");
    if (stage === "placement") {
      const children = (element) => [element, ...(element.children || []).flatMap(children)];
      const help = children(form).find((child) => child.className === "field-help");
      assert.equal(help.tabIndex, 0);
      assert.match(help.dataset.tooltip, /Editing this text fixes your chosen line breaks/);
      const manual = children(form).find((child) => child.name === "manualLineBreaks");
      manual.listeners.input();
      assert.equal(manual.dataset.userEdited, "1");
    }
    if (stage === "placement") for (const name of ["rotation", "skewX", "skewY"]) {
      const children = (element) => [element, ...(element.children || []).flatMap(children)];
      const input = children(form).find((child) => child.name === name);
      assert.equal(input.type, "number");
      assert.equal(input.value, 0);
      assert.equal(input.step, "0.5");
      assert.equal(input.min, name === "rotation" ? "-180" : "-60");
      assert.equal(input.max, name === "rotation" ? "180" : "60");
      input.listeners.change();
      assert.deepEqual(changes.at(-1), [stage, name]);
      const slider = input.parentNode.children.find((child) => child.type === "range");
      assert.ok(slider, "slider is beside the number input");
      assert.equal(slider.min, input.min); assert.equal(slider.max, input.max); assert.equal(slider.step, input.step);
      assert.equal(slider.tabIndex, -1, "Tab skips the slider, not the number input");
      assert.ok(slider["aria-label"]); assert.ok(input["aria-label"]);
      slider.value = "-12.5"; slider.listeners.input();
      assert.equal(input.value, "-12.5");
      slider.listeners.change();
      assert.deepEqual(changes.at(-1), [stage, name]);
      input.value = "0"; input.listeners.change();
      assert.equal(slider.value, "0");
    }
  }
  const css = fs.readFileSync(path.join(root, "editor.css"), "utf8");
  assert.match(css, /\.field-grid\s*\{[^}]*grid-template-columns:\s*repeat\(4, minmax\(0, 1fr\)\)/);
});

test("zero and blank angles replace old values in the store, drafts, undo and redo", () => {
  const storeSource = fs.readFileSync(path.join(root, "store.js"), "utf8");
  const storage = new Map();
  const Store = vm.runInNewContext(storeSource.replace("export class", "class") + "\nEditorStore", {
    EventTarget, Event, structuredClone,
    localStorage: {getItem: (key) => storage.get(key), setItem: (key, value) => storage.set(key, value)},
  });
  const apply = app.match(/function applyInspector\(inputs, target, changedField\) \{[\s\S]*?\n\}/)[0];
  for (const name of ["rotation", "skewX", "skewY"]) for (const value of ["0", ""]) {
    const store = new Store({category: "default", jobId: "12345678"});
    storage.clear();
    store.setDocument({stage: "placement", page: 0, revision: 1,
      recordsByArtifact: {placements: [{recordId: "box", [name]: 15}]}});
    vm.runInNewContext(apply + `\napplyInspector(inputs, target, ${JSON.stringify(name)})`, {
      store, applyFields: screens.applyFields,
      inputs: {[name]: {type: "number", value}}, target: {stage: "placement", artifact: "placements", recordId: "box"},
    });
    assert.equal(store.records.placements[0][name], 0);
    assert.equal(JSON.parse(storage.get(store.draftKey())).records.placements[0][name], 0);
    store.undo(); assert.equal(store.records.placements[0][name], 15);
    store.redo(); assert.equal(store.records.placements[0][name], 0);
  }
});

test("clearing font size restores auto in the record, draft, reload, undo and redo", () => {
  const storage = new Map();
  const Store = vm.runInNewContext(fs.readFileSync(path.join(root, "store.js"), "utf8").replace("export class", "class") + "\nEditorStore", {
    EventTarget, Event, structuredClone,
    localStorage: {getItem: (key) => storage.get(key), setItem: (key, value) => storage.set(key, value)},
  });
  const config = {category: "default", jobId: "12345678"};
  const store = new Store(config);
  const document = {stage: "placement", page: 0, revision: 1,
    recordsByArtifact: {placements: [{recordId: "box", fontSizeWidthPercent: 1.65, font: "ComicNeue-Bold.ttf", rotation: 12}]}};
  store.setDocument(document);
  const apply = app.match(/function applyInspector\(inputs, target, changedField\) \{[\s\S]*?\n\}/)[0];
  const context = vm.createContext({store, applyFields: screens.applyFields,
    inputs: {fontSizeWidthPercent: {type: "number", value: ""}},
    target: {stage: "placement", artifact: "placements", recordId: "box"}});
  vm.runInContext(apply + '\napplyInspector(inputs, target, "fontSizeWidthPercent")', context);
  const isAuto = (record) => assert.equal(Object.hasOwn(record, "fontSizeWidthPercent"), false);
  isAuto(store.records.placements[0]);
  assert.equal(store.records.placements[0].font, "ComicNeue-Bold.ttf");
  assert.equal(store.records.placements[0].rotation, 12);
  isAuto(JSON.parse(storage.get(store.draftKey())).records.placements[0]);
  const reloaded = new Store(config); reloaded.setDocument(document);
  isAuto(reloaded.records.placements[0]);
  store.undo(); assert.equal(store.records.placements[0].fontSizeWidthPercent, 1.65);
  store.redo(); isAuto(store.records.placements[0]);
  context.inputs.fontSizeWidthPercent.value = "2.25";
  vm.runInContext('applyInspector(inputs, target, "fontSizeWidthPercent")', context);
  assert.equal(store.records.placements[0].fontSizeWidthPercent, 2.25);
});
