// Run with: node tests/test_preview_flow.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");
const root = path.join(__dirname, "../web_editor");
const read = (name) => fs.readFileSync(path.join(root, name), "utf8");
const app = read("app.js");
const fn = (name) => app.match(new RegExp(`(?:async )?function ${name}\\([^]*?\\n\\}`))[0];
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return {promise, resolve, reject}; };
const settle = async () => { for (let index = 0; index < 12; index++) await Promise.resolve(); };

function documentFor(page = 3) {
  return {page, stage: "placement", revision: 1, image: {url: `original-${page}`}, stageStates: {},
    recordsByArtifact: {placements: [{recordId: `text-${page}`, boxno: 0, placementRegion: [10, 10, 30, 30]}],
      translations: [{boxno: 0, englishText: "Hello"}], ocr_structured: [{region: [10, 10, 30, 30], openLettering: false}], clean_boxes: []}};
}

function setup({text = null, automatic = true} = {}) {
  const timers = new Map(), events = [], requests = [], loads = [], statuses = [], storage = new Map();
  let timer = 0;
  const localStorage = {getItem: (key) => storage.get(key), setItem: (key, value) => storage.set(key, value), removeItem: (key) => storage.delete(key)};
  const Store = vm.runInNewContext(read("store.js").replace("export class", "class") + "\nEditorStore", {Event, EventTarget, structuredClone, localStorage});
  const store = new Store({category: "default", jobId: "test"}); store.setDocument(documentFor());
  const canvas = {load: async (url, options) => { loads.push([url, options]); events.push(`image:${url}`); return true; },
    prepareTextLayers: async () => { events.push("text:start"); if (text) await text.promise; events.push("text:done"); }};
  const api = {cleanPreview: (page, records, boxes, signal) => {
    events.push(`clean:${page}`);
    const result = deferred(); requests.push({page, records, boxes, signal, ...result});
    if (automatic) result.resolve({url: `clean-${page}-${requests.length}`});
    return result.promise;
  }};
  const status = (...args) => statuses.push(args);
  const Preview = vm.runInNewContext(read("preview.js").replace("export class", "class") + "\nPreviewController", {
    AbortController, structuredClone,
    setTimeout: (callback, delay) => { assert.equal(delay, 250); timers.set(++timer, callback); return timer; },
    clearTimeout: (id) => timers.delete(id),
  });
  const preview = new Preview(store, canvas, api, status);
  const advance = async () => { for (const [id, callback] of [...timers]) { timers.delete(id); callback(); } await settle(); };
  return {store, canvas, api, preview, requests, loads, statuses, events, timers, advance, status};
}

test("ImageMagick text finishes before LaMa starts; background swaps preserve layers", async () => {
  const text = deferred(), page = setup({text});
  page.preview.show(); await settle();
  assert.deepEqual(page.events, ["image:original-3", "text:start"]);
  assert.equal(page.requests.length, 0);
  text.resolve(); await settle();
  assert.deepEqual(page.events, ["image:original-3", "text:start", "text:done", "clean:3", "image:clean-3-1"]);
  assert.equal(page.loads[1][1], undefined, "cleaned background keeps text layers enabled");
});

test("Preview shows the saved cleaned page and text while updated cleaning is pending", async () => {
  const page = setup({automatic: false});
  page.store.document.image.cleanedUrl = "saved-cleaned-3";
  page.preview.show(); await settle();
  assert.deepEqual(page.events, ["image:saved-cleaned-3", "text:start", "text:done", "clean:3"]);
  assert.equal(page.loads.at(-1)[0], "saved-cleaned-3");
  page.store.mutate((records) => records.clean_boxes.push({recordId: "clean", boxno: 0, region: [40, 40, 60, 60]}));
  await page.advance();
  assert.equal(page.loads.at(-1)[0], "saved-cleaned-3", "keep the saved cleaned page during edits too");
  page.requests[1].resolve({url: "updated-cleaned-3"}); await settle();
  assert.equal(page.loads.at(-1)[0], "updated-cleaned-3");
  page.requests[0].resolve({url: "old-cleaned-3"}); await settle();
  assert.equal(page.loads.at(-1)[0], "updated-cleaned-3");
});

test("Preview uses the original page when no saved cleaned page exists", async () => {
  const page = setup({automatic: false});
  page.store.document.image.cleanedUrl = null;
  page.preview.show(); await settle();
  assert.equal(page.loads.at(-1)[0], "original-3");
  assert.equal(page.requests.length, 1);
});

test("add, move, resize, delete, undo and redo automatically refresh only cleaning", async () => {
  const page = setup(); page.preview.show(); await settle();
  const box = {recordId: "clean", boxno: 0, region: [50, 50, 80, 80]};
  page.store.mutate((records) => records.clean_boxes.push(box));
  assert.equal(page.requests.length, 1); await page.advance(); assert.equal(page.requests.length, 2);
  for (let x = 51; x <= 60; x++) page.store.mutate((records) => { records.clean_boxes[0].region[0] = x; });
  assert.equal(page.timers.size, 1); await page.advance();
  assert.equal(page.requests.length, 3); assert.equal(page.requests[2].boxes[0].region[0], 60);
  page.store.mutate((records) => { records.clean_boxes[0].region[2] = 95; }); await page.advance();
  assert.equal(page.requests.at(-1).boxes[0].region[2], 95);
  page.store.undo(); await page.advance(); assert.equal(page.requests.at(-1).boxes[0].region[2], 80);
  page.store.redo(); await page.advance(); assert.equal(page.requests.at(-1).boxes[0].region[2], 95);
  page.store.mutate((records) => { records.clean_boxes = []; }); await page.advance();
  assert.equal(page.requests.at(-1).boxes.length, 0);
  const count = page.requests.length;
  page.store.mutate((records) => { records.placements[0].fontSizeWidthPercent = 2; records.translations[0].englishText = "Changed"; });
  await page.advance(); assert.equal(page.requests.length, count, "text/style edits do not invoke LaMa");
});

test("switching to Original cancels cleaning and ignores stale success and errors", async () => {
  for (const fail of [false, true]) {
    const page = setup({automatic: false}); page.preview.show(); await settle();
    page.preview.hide(); await page.canvas.load("original");
    assert.equal(page.requests[0].signal.aborted, true);
    const count = page.statuses.length;
    fail ? page.requests[0].reject(new Error("Old page failed")) : page.requests[0].resolve({url: "stale"});
    await settle(); assert.equal(page.loads.at(-1)[0], "original"); assert.equal(page.statuses.length, count);
  }
});

test("a late mask result cannot overwrite a newer edit on the same page", async () => {
  const page = setup({automatic: false}); page.preview.show(); await settle();
  page.store.mutate((records) => { records.clean_boxes.push({recordId: "clean", boxno: 0, region: [40, 40, 60, 60]}); });
  await page.advance(); assert.equal(page.requests[0].signal.aborted, true);
  page.requests[1].resolve({url: "latest-mask"}); await settle();
  page.requests[0].resolve({url: "old-mask"}); await settle();
  assert.equal(page.loads.at(-1)[0], "latest-mask");
});

test("save/reset document replacement refreshes the active background", async () => {
  const page = setup({automatic: false}); page.preview.show(); await settle();
  page.store.setDocument(documentFor()); await page.advance();
  assert.equal(page.requests.length, 2);
  page.requests[0].resolve({url: "before-reset"}); page.requests[1].resolve({url: "after-reset"}); await settle();
  assert.equal(page.loads.at(-1)[0], "after-reset");
});

test("source erase/region changes refresh cleaning but selection does not", async () => {
  const page = setup(); page.preview.show(); await settle();
  page.store.select("placements", "text-3"); await page.advance(); assert.equal(page.requests.length, 1);
  page.store.mutate((records) => { records.ocr_structured[0].openLettering = true; }); await page.advance();
  assert.equal(page.requests[1].records.length, 0);
});

function navigationSetup() {
  const page = setup({automatic: false}), pendingLoads = [];
  page.api.load = (stage, number, signal) => { const request = deferred(); pendingLoads.push({stage, number, signal, ...request}); return request.promise; };
  const elements = new Map();
  const context = vm.createContext({store: page.store, api: page.api, canvas: page.canvas, preview: page.preview,
    loading: false, loadRequest: 0, loadController: null, currentPreviewRequest: null, navigationPage: null,
    typesettingView: "preview", AbortController, mergeSelection: new Set(), config: {pageCount: 10},
    AVAILABLE_STAGES: ["ocr", "text", "placement"], setStatus: page.status,
    imageUrl: () => "original", loadRevisions: async () => {},
    byId: (id) => { if (!elements.has(id)) elements.set(id, {}); return elements.get(id); }});
  for (const name of ["load", "navigate", "loadTypesettingView", "exactPreview"]) vm.runInContext(fn(name), context);
  return {...page, context, pendingLoads};
}

test("page 3 → 4 → 5 navigation finishes while page 4 LaMa is still pending", async () => {
  const page = navigationSetup();
  const fourth = page.context.load("placement", 4); page.pendingLoads[0].resolve(documentFor(4));
  await fourth; await settle();
  assert.equal(page.context.loading, false);
  assert.equal(page.requests.length, 1, "LaMa for page 4 is still pending");
  const fifth = page.context.navigate("placement", 5); page.pendingLoads[1].resolve(documentFor(5));
  await fifth; await settle(); assert.equal(page.store.page, 5);
  page.requests[1].resolve({url: "clean-5"}); await settle();
  const count = page.statuses.length;
  page.requests[0].resolve({url: "clean-4-late"}); await settle();
  assert.equal(page.loads.at(-1)[0], "clean-5"); assert.equal(page.statuses.length, count);
});

test("rapid next-page keys use the requested page and late page fetches are discarded", async () => {
  const page = navigationSetup(); let keydown;
  Object.assign(page.context, {resetting: false, ocrAreaRunning: false,
    document: {addEventListener: (_event, callback) => { keydown = callback; }}});
  vm.runInContext(app.slice(app.indexOf('document.addEventListener("keydown",'), app.indexOf('window.addEventListener("beforeunload",')), page.context);
  keydown({target: {tagName: "DIV"}, key: "d", preventDefault() {}});
  keydown({target: {tagName: "DIV"}, key: "d", preventDefault() {}});
  assert.deepEqual(page.pendingLoads.map((item) => item.number), [4, 5]);
  assert.equal(page.pendingLoads[0].signal.aborted, true);
  page.pendingLoads[1].resolve(documentFor(5)); await settle(); assert.equal(page.store.page, 5);
  page.pendingLoads[0].resolve(documentFor(4)); await settle(); assert.equal(page.store.page, 5);
});

test("Current also shows the new source page while its exact rendering is pending", async () => {
  const page = navigationSetup(), renders = [];
  page.context.typesettingView = "current";
  page.context.imageUrl = () => `original-${page.store.page}`;
  page.api.preview = (_stage, number) => { const request = deferred(); renders.push({number, ...request}); return request.promise; };
  const changed = documentFor(4); changed.stageStates.placement = {pending: true};
  const fourth = page.context.load("placement", 4); page.pendingLoads[0].resolve(changed); await fourth;
  assert.equal(page.loads.at(-1)[0], "original-4");
  const changed5 = documentFor(5); changed5.stageStates.placement = {pending: true};
  const fifth = page.context.load("placement", 5); page.pendingLoads[1].resolve(changed5); await fifth;
  assert.equal(page.loads.at(-1)[0], "original-5");
  renders[0].resolve({url: "old-render-4"}); await settle();
  assert.equal(page.loads.at(-1)[0], "original-5");
  renders[1].resolve({url: "current-5"}); await settle();
  assert.equal(page.loads.at(-1)[0], "current-5");
});

test("a late exact Current response cannot replace Original or another page", async () => {
  const page = navigationSetup(), result = deferred();
  page.context.typesettingView = "current"; page.store.dirty = true;
  page.api.preview = () => result.promise;
  const current = page.context.exactPreview();
  page.context.typesettingView = "original"; await page.context.loadTypesettingView();
  const count = page.statuses.length;
  result.resolve({url: "old-current"}); await current;
  assert.equal(page.loads.at(-1)[0], "original"); assert.equal(page.statuses.length, count);
});

test("text-render failures remain visible and do not start LaMa", async () => {
  const page = setup(); page.canvas.prepareTextLayers = async () => { throw new Error("ImageMagick failed"); };
  page.preview.show(); await settle();
  assert.equal(page.requests.length, 0);
  assert.match(page.statuses.at(-1)[0], /ImageMagick failed/);
});

test("saved revisions use the browser document and ignore old page responses", async () => {
  const page = navigationSetup(), pending = [];
  const element = () => ({children: [], append(...children) { this.children.push(...children); },
    replaceChildren() { this.children = []; }, addEventListener() {}, dataset: {},
    get childElementCount() { return this.children.length; }});
  const target = element();
  page.context.document = {createElement: element, createTextNode: (text) => ({text})};
  page.context.byId = () => target;
  page.api.revisions = () => { const result = deferred(); pending.push(result); return result.promise; };
  vm.runInContext(fn("loadRevisions"), page.context);
  const first = page.context.loadRevisions();
  pending[0].resolve({revisions: [{revision: 12}]}); await first;
  assert.equal(target.children.length, 1);
  assert.equal(target.children[0].children[0].text, "#12 ");
  assert.equal(target.children[0].children[1].textContent, "Restore");
  const stale = page.context.loadRevisions();
  page.store.setDocument(documentFor(5));
  const current = page.context.loadRevisions();
  pending[2].resolve({revisions: [{revision: 13}]}); await current;
  pending[1].resolve({revisions: [{revision: 11}]}); await stale;
  assert.equal(target.children[0].children[0].text, "#13 ");
});
