const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");
const root = path.join(__dirname, "../web_editor");
const app = fs.readFileSync(path.join(root, "app.js"), "utf8");
const fn = (name) => app.match(new RegExp(`function ${name}\\([^]*?\\n\\}`))[0];
const screens = vm.runInNewContext(fs.readFileSync(path.join(root, "screens.js"), "utf8").replaceAll("export ", "") + "\n({fieldsFor, applyFields})");

function setup() {
  const timers = new Map(), pending = [], status = []; let timer = 0;
  const store = {document: {}, page: 2, records: {translations: [{boxno: 0, englishText: "Current translation"}]}};
  const context = vm.createContext({store, AbortController, lineBreaksTimer: null, lineBreaksController: null,
    setTimeout: (callback, delay) => { assert.equal(delay, 200); timers.set(++timer, callback); return timer; },
    clearTimeout: (id) => timers.delete(id), setStatus: (...args) => status.push(args),
    api: {textLayout: (page, entry, signal) => new Promise((resolve, reject) => {pending.push({page, entry, signal, resolve, reject});})}});
  vm.runInContext(fn("refreshLineBreaks"), context);
  const input = {isConnected: true, dataset: {}, value: "Initial\nbreaks"};
  const record = {recordId: "box", boxno: 0, placementRegion: [0, 0, 300, 200]};
  const advance = () => { const callbacks = [...timers.values()]; timers.clear(); return callbacks.map((callback) => callback()); };
  return {context, store, pending, status, timers, input, record, advance};
}

test("the field displays automatic layout until a manual override exists", () => {
  const field = (record) => screens.fieldsFor("placement", record).find(([name]) => name === "manualLineBreaks");
  assert.equal(field({_renderedText: "One\nTwo"})[3], "One\nTwo");
  assert.match(field({})[1], /automatic/);
  assert.equal(field({_renderedText: "Old", manualLineBreaks: "Chosen\nlines"})[3], "Chosen\nlines");
  assert.match(field({manualLineBreaks: "Chosen"})[1], /fixed/);
});

test("geometry changes debounce and refresh displayed breaks without changing the record", async () => {
  const page = setup();
  page.context.refreshLineBreaks(page.input, page.record);
  page.record.placementRegion = [20, 20, 150, 220];
  page.context.refreshLineBreaks(page.input, page.record);
  assert.equal(page.timers.size, 1);
  const [finished] = page.advance();
  assert.deepEqual(Array.from(page.pending[0].entry.placementRegion), [20, 20, 150, 220]);
  assert.equal(page.pending[0].entry.englishText, "Current translation");
  page.pending[0].resolve("Updated\nline breaks"); await finished;
  assert.equal(page.input.value, "Updated\nline breaks");
  assert.equal(Object.hasOwn(page.record, "manualLineBreaks"), false);
});

test("late layout results cannot replace active typing, another selection, or a new page", async () => {
  for (const invalidation of ["typing", "selection", "page"]) {
    const page = setup(); page.context.refreshLineBreaks(page.input, page.record);
    const [finished] = page.advance();
    if (invalidation === "typing") { page.input.dataset.userEdited = "1"; page.input.value = "Typed"; }
    if (invalidation === "selection") page.input.isConnected = false;
    if (invalidation === "page") page.store.document = {};
    page.pending[0].resolve("Stale"); await finished;
    assert.notEqual(page.input.value, "Stale");
  }
});

test("manual mode cancels automatic work; clearing the override starts it again", async () => {
  const page = setup(); page.context.refreshLineBreaks(page.input, page.record);
  const [first] = page.advance();
  page.record.manualLineBreaks = "Chosen\nbreaks";
  page.context.refreshLineBreaks(page.input, page.record);
  assert.equal(page.pending[0].signal.aborted, true);
  assert.equal(page.timers.size, 0);
  page.pending[0].resolve("Old auto"); await first;
  page.record.manualLineBreaks = "";
  page.context.refreshLineBreaks(page.input, page.record);
  const [second] = page.advance();
  page.pending[1].resolve("Automatic\nagain"); await second;
  assert.equal(page.input.value, "Automatic\nagain");
});

test("unrelated field edits do not fix the displayed automatic breaks", () => {
  const page = setup();
  page.store.records.placements = [page.record];
  page.store.mutate = (callback) => callback(page.store.records);
  Object.assign(page.context, {applyFields: screens.applyFields,
    inputs: {manualLineBreaks: {type: "textarea", value: "Displayed\nautomatic"}, left: {type: "number", value: "25"}},
    target: {stage: "placement", artifact: "placements", recordId: "box"}});
  vm.runInContext(fn("applyInspector"), page.context);
  vm.runInContext('applyInspector(inputs, target, "left")', page.context);
  assert.equal(Object.hasOwn(page.store.records.placements[0], "manualLineBreaks"), false);
  vm.runInContext('applyInspector(inputs, target, "manualLineBreaks")', page.context);
  assert.equal(page.store.records.placements[0].manualLineBreaks, "Displayed\nautomatic");
  page.context.inputs.left.value = "40";
  vm.runInContext('applyInspector(inputs, target, "left")', page.context);
  assert.equal(page.store.records.placements[0].manualLineBreaks, "Displayed\nautomatic");
  page.context.inputs.manualLineBreaks.value = "";
  vm.runInContext('applyInspector(inputs, target, "manualLineBreaks")', page.context);
  assert.equal(page.store.records.placements[0].manualLineBreaks, "");
});

test("layout errors are readable, but aborted requests do not show errors", async () => {
  for (const aborted of [false, true]) {
    const page = setup(); page.context.refreshLineBreaks(page.input, page.record);
    const [finished] = page.advance();
    if (aborted) page.context.lineBreaksController.abort();
    page.pending[0].reject(new Error("Endpoint unavailable")); await finished;
    assert.equal(page.status.length, aborted ? 0 : 1);
    if (!aborted) assert.match(page.status[0][0], /automatic line breaks.*Endpoint unavailable/);
  }
});

test("layout API returns text without decoding images or calling preview rendering", async () => {
  const requests = [];
  const source = fs.readFileSync(path.join(root, "api.js"), "utf8").replace("export class", "class") + "\nEditorApi";
  const Api = vm.runInNewContext(source, {fetch: async (...args) => {requests.push(args); return {ok: true, json: async () => ({text: "One\nTwo"})};}});
  const api = new Api("default", "test"), controller = new AbortController();
  assert.equal(await api.textLayout(2, {placementRegion: [0, 0, 100, 200]}, controller.signal), "One\nTwo");
  assert.match(requests[0][0], /pages\/2\/text-layout$/);
  assert.equal(requests[0][1].signal, controller.signal);
  assert.deepEqual(JSON.parse(requests[0][1].body), {record: {placementRegion: [0, 0, 100, 200]}});
});
