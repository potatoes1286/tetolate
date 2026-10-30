const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");
const app = fs.readFileSync(path.join(__dirname, "../web_editor/app.js"), "utf8");
const source = (name) => app.match(new RegExp(`(?:async )?function ${name}\\([^]*?\\n\\}`))[0];
const settle = () => new Promise((resolve) => setImmediate(resolve));

function setup({dirty = false, confirm = true, error = null, save = null} = {}) {
  const elements = new Map(), calls = [], status = [];
  const store = {dirty, stage: "text", page: 2};
  const context = vm.createContext({store, queuedAction: "", loading: false, resetting: false, ocrAreaRunning: false,
    window: {location: {href: ""}}, confirm: () => confirm, renderTools() {},
    byId: (id) => {
      if (!elements.has(id)) elements.set(id, {disabled: false, setAttribute() {}, dataset: {}});
      return elements.get(id);
    },
    setStatus: (...args) => status.push(args),
    save: async () => {calls.push(["save"]); if (save) await save(); else store.dirty = false;},
    api: Object.fromEntries(["regenerate", "regeneratePage", "regenerateAllPages", "finishEdits"].map((name) => [name, async (...args) => {
      calls.push([name, ...args]); if (error) throw new Error(error); return {url: "/job/default/test"};
    }])),
  });
  for (const name of ["setQueuedAction", "regenerate"]) vm.runInContext(source(name), context);
  return {context, elements, store, calls, status};
}

for (const [action, expected] of [["stage", ["regenerate", "text", 2]], ["page", ["regeneratePage", 2]],
  ["all-pages", ["regenerateAllPages"]], ["finish-edits", ["finishEdits"]]]) {
  test(`${action} saves the draft and uses its exact scope`, async () => {
    const page = setup({dirty: true}); await page.context.regenerate(action);
    assert.deepEqual(page.calls, [["save"], expected]);
    assert.equal(page.context.window.location.href, "/job/default/test");
    assert.equal(page.elements.get("editor-app").inert, false);
    for (const id of ["regenerate-stage", "regenerate-page", "regenerate-all-pages", "finish-edits"])
      assert.equal(page.elements.get(id).disabled, true);
  });
}

test("failed saves and cancelled confirmation never queue a rerun", async () => {
  const page = setup({dirty: true, save: async () => {}});
  await page.context.regenerate("page");
  assert.deepEqual(page.calls, [["save"]]); assert.equal(page.context.queuedAction, "");
  for (const action of ["stage", "finish-edits"]) {
    const cancelled = setup({confirm: false}); await cancelled.context.regenerate(action);
    assert.equal(cancelled.calls.length, 0);
  }
});

test("double clicks during Save queue only one action", async () => {
  let release;
  const page = setup({dirty: true, save: () => new Promise((resolve) => {release = resolve;})});
  const first = page.context.regenerate("page"); await settle();
  await page.context.regenerate("all-pages"); assert.deepEqual(page.calls, [["save"]]);
  page.store.dirty = false; release(); await first;
  assert.deepEqual(page.calls, [["save"], ["regeneratePage", 2]]);
});

test("queue errors restore controls and display the message", async () => {
  const page = setup({error: "No saved changes"}); await page.context.regenerate("page");
  assert.equal(page.context.queuedAction, "");
  assert.equal(page.elements.get("editor-app").inert, false);
  assert.equal(page.elements.get("finish-edits").disabled, false);
  assert.deepEqual(page.status.at(-1), ["No saved changes", true]);
});

test("API actions use separate stage, page, all-pages and finish routes", async () => {
  const requests = [];
  const apiSource = fs.readFileSync(path.join(__dirname, "../web_editor/api.js"), "utf8");
  const Api = vm.runInNewContext(apiSource.replace("export class", "class") + "\nEditorApi", {
    fetch: async (url, options) => {requests.push([url, options.method]); return {ok: true, json: async () => ({})};},
  });
  const api = new Api("default", "test");
  await api.regenerate("text", 2); await api.regeneratePage(2); await api.regenerateAllPages(); await api.finishEdits();
  assert.deepEqual(requests, ["pages/2/stages/text/regenerate", "pages/2/regenerate", "regenerate-all-pages", "finish-edits"]
    .map((suffix) => [`/api/job/default/test/editor/v2/${suffix}`, "POST"]));
});
