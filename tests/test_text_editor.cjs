// Run with: node tests/test_text_editor.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");
const root = path.join(__dirname, "../web_editor");
const source = fs.readFileSync(path.join(root, "screens.js"), "utf8");
const screens = vm.runInNewContext(source.replaceAll("export ", "") +
  "\n({STAGES, SCREEN, fieldsFor, textFieldTarget, updateTextField, recordLabel})");
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

test("OCR and Typesetting coordinates share one row with independent number inputs", () => {
  const render = app.match(/function renderInspector\(\) \{[\s\S]*?\n\}/)[0];
  class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.dataset = {}; this.listeners = {}; }
    append(...children) { for (const child of children) { child.parentNode = this; this.children.push(child); } }
    replaceChildren() { this.children = []; }
    addEventListener(name, callback) { this.listeners[name] = callback; }
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
  }
  const css = fs.readFileSync(path.join(root, "editor.css"), "utf8");
  assert.match(css, /\.field-grid\s*\{[^}]*grid-template-columns:\s*repeat\(4, minmax\(0, 1fr\)\)/);
});
