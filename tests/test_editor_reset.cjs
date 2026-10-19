// Run with: node tests/test_editor_reset.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");
const root = path.join(__dirname, "../web_editor");
const appSource = fs.readFileSync(path.join(root, "app.js"), "utf8");
const resetSource = appSource.match(/async function resetStage\(\) \{[\s\S]*?\n\}/)[0];
const storeSource = fs.readFileSync(path.join(root, "store.js"), "utf8");

function setup({confirm = true, error = null, stage = "text", checkpoint = null, rerunFrom = null} = {}) {
  const storage = new Map();
  const localStorage = {
    getItem: (key) => storage.get(key) || null,
    setItem: (key, value) => storage.set(key, value),
    removeItem: (key) => storage.delete(key),
  };
  const Store = vm.runInNewContext(storeSource.replace("export class", "class") + "\nEditorStore", {
    EventTarget, Event, structuredClone, localStorage,
  });
  const store = new Store({category: "default", jobId: "12345678"});
  const generated = {
    stage, page: 2, revision: 4, rerunFrom, image: {url: "/original/2"},
    recordsByArtifact: {translations: [{recordId: "one", englishText: "Generated"}]},
    translationNotes: {job: "Saved notes", page: "Saved page notes"},
  };
  store.setDocument(structuredClone(generated));
  store.records.translations[0].englishText = "Draft";
  store.dirty = true;
  store.history.push(store.snapshot());
  store.future.push(store.snapshot());
  store.saveDraft();
  localStorage.setItem(store.draftKey(3, stage), "Other page draft");
  const elements = {"editor-app": {inert: false}, "reset-stage": {disabled: false, focus() {}}};
  const calls = [];
  const context = vm.createContext({
    store, config: {reviewCheckpoint: checkpoint}, loading: false, resetting: false,
    ocrAreaRunning: false, queuedAction: "", mergeSelection: new Set(["one"]),
    confirm: (message) => { calls.push(["confirm", message]); return confirm; },
    currentStageState: () => ({label: "Test stage"}),
    byId: (id) => elements[id],
    setStatus: (...args) => calls.push(["status", ...args]),
    api: {reset: async (...args) => {
      calls.push(["reset", ...args]);
      assert.equal(elements["editor-app"].inert, true);
      assert.equal(elements["reset-stage"].disabled, true);
      if (error) throw new Error(error);
      return {...structuredClone(generated), revision: 5};
    }},
    canvas: {load: (...args) => calls.push(["canvas", ...args])},
    loadTypesettingView: async () => calls.push(["preview"]),
    loadRevisions: async () => calls.push(["history"]),
  });
  vm.runInContext(resetSource, context);
  return {context, store, storage, calls, elements};
}

test("reset clears the current draft and undo history, then loads generated records", async () => {
  const page = setup();
  await page.context.resetStage();
  assert.deepEqual(page.calls.find(([type]) => type === "reset"), ["reset", "text", 2, 4]);
  assert.match(page.calls[0][1], /Reset Test stage on page 2/);
  assert.equal(page.store.records.translations[0].englishText, "Generated");
  assert.equal(page.store.document.translationNotes.job, "Saved notes");
  assert.equal(page.store.dirty, false);
  assert.equal(page.store.history.length, 0);
  assert.equal(page.store.future.length, 0);
  assert.equal(page.storage.has(page.store.draftKey()), false);
  assert.equal(page.storage.get(page.store.draftKey(3, "text")), "Other page draft");
  assert.equal(page.context.mergeSelection.size, 0);
  assert.equal(page.elements["editor-app"].inert, false);
  assert.equal(page.elements["reset-stage"].disabled, false);
  assert.ok(page.calls.some(([type]) => type === "history"));
  assert.match(page.calls.at(-1)[1], /No changes need regeneration/);
});

test("reset still reports regeneration when other pending changes remain", async () => {
  const page = setup({rerunFrom: "placements"});
  await page.context.resetStage();
  assert.match(page.calls.at(-1)[1], /Regenerate downstream/);
});

test("cancel keeps all edits without a request", async () => {
  const page = setup({confirm: false});
  await page.context.resetStage();
  assert.equal(page.calls.length, 1);
  assert.equal(page.store.records.translations[0].englishText, "Draft");
  assert.equal(page.store.dirty, true);
  assert.equal(page.storage.has(page.store.draftKey()), true);
});

test("a failed reset keeps the draft and re-enables the editor", async () => {
  const page = setup({error: "Editor revision is stale."});
  await page.context.resetStage();
  assert.equal(page.store.records.translations[0].englishText, "Draft");
  assert.equal(page.store.dirty, true);
  assert.equal(page.storage.has(page.store.draftKey()), true);
  assert.equal(page.elements["editor-app"].inert, false);
  assert.equal(page.elements["reset-stage"].disabled, false);
  assert.deepEqual(page.calls.at(-1), ["status", "Editor revision is stale.", true]);
});

for (const [key, value] of [["loading", true], ["resetting", true], ["ocrAreaRunning", true], ["queuedAction", "all"]]) {
  test(`does not reset while ${key}`, async () => {
    const page = setup();
    page.context[key] = value;
    await page.context.resetStage();
    assert.equal(page.calls.length, 0);
    assert.equal(page.store.dirty, true);
  });
}

test("typesetting reset refreshes its preview", async () => {
  const page = setup({stage: "placement"});
  await page.context.resetStage();
  assert.ok(page.calls.some(([type]) => type === "preview"));
});

test("OCR checkpoint reset shows the continue action", async () => {
  const page = setup({stage: "ocr", checkpoint: "ocr"});
  await page.context.resetStage();
  assert.match(page.calls.at(-1)[1], /Continue processing/);
});

test("API sends a revision-checked POST to the current page and stage", async () => {
  const apiSource = fs.readFileSync(path.join(root, "api.js"), "utf8");
  const calls = [];
  const Api = vm.runInNewContext(apiSource.replace("export class", "class") + "\nEditorApi", {
    fetch: async (...args) => { calls.push(args); return {ok: true, json: async () => ({revision: 5})}; },
  });
  const result = await new Api("default", "12345678").reset("text", 2, 4);
  assert.equal(result.revision, 5);
  assert.equal(calls[0][0], "/api/job/default/12345678/editor/v2/pages/2/stages/text/reset");
  assert.equal(calls[0][1].method, "POST");
  assert.deepEqual(JSON.parse(calls[0][1].body), {baseRevision: 4});
});
