const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../web_pages.py"), "utf8");
const script = source.match(/JOB_OPTIONS_SCRIPT = """\s*<script>([\s\S]*?)<\/script>\s*"""/)[1];
const settle = () => new Promise((resolve) => setImmediate(resolve));
const field = (name, value, type = "text", valid = true) => ({name, value: String(value), type, checked: value, checkValidity: () => valid});

function setup(fetch = async () => ({ok: true})) {
  const events = {}, timers = new Map(), requests = [], navigations = [], status = {textContent: ""};
  let timer = 0;
  const form = {dataset: {jobOptions: "/api/v1/jobs/test/settings?category=default"}, elements: [], isConnected: true,
    querySelector: (selector) => selector === "details" ? {addEventListener: (name, callback) => {events[name] = callback;}} : status,
    addEventListener: (name, callback) => {events[name] = callback;}};
  vm.runInNewContext(script, {
    document: {querySelector: () => form, addEventListener: (name, callback) => {events[name] = callback;}},
    window: {addEventListener: (name, callback) => {events[name] = callback;}, location: {assign: (url) => navigations.push(url)}},
    fetch: async (url, options) => {requests.push([url, {...options, body: JSON.parse(options.body)}]); return fetch(url, options);},
    setTimeout: (callback, delay) => {assert.equal(delay, 250); timers.set(++timer, callback); return timer;},
    clearTimeout: (id) => timers.delete(id),
  });
  return {form, events, requests, status, timers, navigations,
    change: (input) => events.change({target: input}), type: (input) => events.input({target: input}),
    flush: async () => {for (const [id, callback] of [...timers]) {timers.delete(id); callback();} await settle();},
    click: async () => {let prevented = false; const link = {href: "/job/default/test/edit", hasAttribute: () => false};
      await events.click({target: {closest: () => link}, preventDefault() {prevented = true;}}); return prevented;},
    unload: () => {let prevented = false; events.beforeunload({preventDefault() {prevented = true;}}); return prevented;},
  };
}

test("job endpoint, workers, false flags and zero tokens save without submitting", async () => {
  const page = setup();
  for (const input of [field("paddleocr_vl_server_url", "http://localhost:18081/v1"), field("ocr_page_workers", 3, "number"),
    field("enable_proofreading", false, "checkbox"), field("thinking_budget_tokens", 0, "number"), field("vlm_model", "", "select-one", false)]) page.change(input);
  await settle();
  assert.deepEqual(page.requests.map(([, request]) => request.body), [
    {paddleocr_vl_server_url: "http://localhost:18081/v1"}, {ocr_page_workers: 3}, {proofread_translations: false}, {thinking_budget_tokens: 0}, {vlm_model: ""},
  ]);
  assert.equal(page.status.textContent, "Advanced options saved.");
  assert.equal(page.unload(), false);
});

test("typing is debounced and clicking Edit waits for the saved endpoint", async () => {
  let release;
  const page = setup(() => new Promise((resolve) => {release = () => resolve({ok: true});}));
  const input = field("paddleocr_vl_server_url", "http://localhost:18081/v1");
  page.type(input);
  assert.equal(page.requests.length, 0);
  const click = page.click(); await settle();
  assert.equal(page.requests.length, 1);
  assert.equal(page.navigations.length, 0);
  release(); assert.equal(await click, true);
  assert.deepEqual(page.navigations, ["/job/default/test/edit"]);
});

test("failed saves block Edit, show the server message and allow a retry", async () => {
  let fail = true;
  const page = setup(async () => ({ok: !fail, status: 400, text: async () => '{"error":{"message":"Invalid OCR endpoint."}}'}));
  const input = field("paddleocr_vl_server_url", "bad");
  page.change(input); await settle();
  assert.equal(await page.click(), true); assert.equal(page.navigations.length, 0);
  assert.match(page.status.textContent, /Invalid OCR endpoint/);
  assert.equal(page.unload(), true);
  fail = false; input.value = "http://localhost:18081/v1"; page.change(input); await settle();
  assert.equal(page.status.textContent, "Advanced options saved.");
  assert.equal(page.unload(), false);
});

test("Edit also waits for input changed while an earlier save is pending", async () => {
  const releases = [];
  const page = setup(() => new Promise((resolve) => {releases.push(() => resolve({ok: true}));}));
  page.type(field("lama_workers", 2, "number"));
  const click = page.click(); await settle();
  page.type(field("lama_workers", 3, "number"));
  releases[0](); await settle();
  assert.equal(page.requests.length, 2);
  assert.deepEqual(page.requests[1][1].body, {lama_workers: 3});
  assert.equal(page.navigations.length, 0);
  releases[1](); await click;
  assert.equal(page.navigations.length, 1);
});

test("saves are ordered and late responses cannot clear newer validation errors", async () => {
  let release;
  const page = setup(() => new Promise((resolve) => {release = () => resolve({ok: true});}));
  page.change(field("lama_workers", 2, "number"));
  page.change(field("imagemagick_workers", 4, "number")); await settle();
  assert.equal(page.requests.length, 1);
  page.change(field("lama_workers", "", "number", false));
  release(); await settle(); assert.equal(page.requests.length, 2);
  release(); await settle(); assert.match(page.status.textContent, /Not saved/);
});

test("auth keys save privately, blank retains them, and removal is explicit", async () => {
  const page = setup();
  page.change(field("paddleocr_vl_auth_token", "test-private-key", "password"));
  page.change(field("vlm_auth_token", "", "password"));
  page.change(field("clear_vlm_auth_token", true, "checkbox")); await settle();
  assert.deepEqual(page.requests.map(([, request]) => request.body), [{paddleocr_vl_auth_token: "test-private-key"}, {clear_vlm_auth_token: true}]);
  for (const [url, request] of page.requests) {
    assert.equal(url, "/api/v1/jobs/test/settings?category=default");
    assert.equal(request.credentials, "same-origin");
    assert.equal(request.method, "PATCH");
    assert.equal(request.keepalive, undefined, "do not send credentials during page unload");
  }
});

test("Save retries fields, while beforeunload warns for pending changes", async () => {
  const page = setup(); page.form.elements = [field("ocr_page_workers", 6, "number")];
  let prevented = false;
  page.events.submit({preventDefault() {prevented = true;}}); await settle();
  assert.equal(prevented, true); assert.deepEqual(page.requests[0][1].body, {ocr_page_workers: 6});
  page.type(field("lama_workers", 7, "number"));
  assert.equal(page.unload(), true); await settle(); assert.equal(page.unload(), false);
});

test("clearing an unsaved key keeps the saved key and dismisses its failed edit", async () => {
  const page = setup(async () => ({ok: false, status: 400, text: async () => '{"error":{"message":"Invalid key."}}'}));
  const input = field("paddleocr_vl_auth_token", "bad", "password");
  page.change(input); await settle(); assert.equal(page.unload(), true);
  input.value = ""; page.change(input); await settle();
  assert.equal(page.unload(), false);
  assert.equal(page.requests.length, 1);
});

test("detached forms do not block navigation or unload", async () => {
  const page = setup(); page.type(field("lama_workers", 4, "number")); page.form.isConnected = false;
  assert.equal(await page.click(), false); assert.equal(page.unload(), false);
});
