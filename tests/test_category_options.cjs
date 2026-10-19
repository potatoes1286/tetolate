// Run with: node tests/test_category_options.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../web_pages.py"), "utf8");
const script = source.match(/CATEGORY_OPTIONS_SCRIPT = """\s*<script>([\s\S]*?)<\/script>\s*"""/)[1];
const settle = () => new Promise((resolve) => setImmediate(resolve));
const input = (name, value, type = "number", valid = true) => ({name, value: String(value), type,
  checked: value, checkValidity: () => valid});

function setup(fetch = async () => ({ok: true})) {
  const events = {}, requests = [], timers = new Map(), status = {textContent: ""};
  let nextTimer = 0;
  const form = {dataset: {categoryOptions: "shorties"},
    querySelector: (selector) => selector === "details"
      ? {addEventListener: (name, handler) => { events[name] = handler; }} : status,
    addEventListener: (name, handler) => { events[name] = handler; }};
  vm.runInNewContext(script, {
    document: {querySelector: () => form},
    window: {addEventListener: (name, handler) => { events[name] = handler; }},
    fetch: async (url, options) => { requests.push([url, {...options, body: JSON.parse(options.body)}]); return fetch(url, options); },
    setTimeout: (callback, delay) => { assert.equal(delay, 250); timers.set(++nextTimer, callback); return nextTimer; },
    clearTimeout: (id) => timers.delete(id),
  });
  return {status, requests, timers,
    change: (target) => events.change({target}),
    type: (target) => events.input({target}),
    flush: async () => { for (const [id, callback] of [...timers]) { timers.delete(id); callback(); } await settle(); },
    submit: () => events.submit(),
    unload: () => { let prevented = false; events.beforeunload({preventDefault() { prevented = true; }}); return prevented; },
  };
}

test("all three worker counts save without an upload or model", async () => {
  const page = setup();
  for (const [name, value] of [["ocr_page_workers", 4], ["lama_workers", 2], ["imagemagick_workers", 8]]) page.change(input(name, value));
  await settle();
  assert.deepEqual(page.requests.map(([, request]) => request.body), [{ocr_page_workers: 4}, {lama_workers: 2}, {imagemagick_workers: 8}]);
  for (const [url, request] of page.requests) {
    assert.equal(url, "/api/v1/categories/shorties/settings");
    assert.equal(request.method, "PATCH");
    assert.equal(request.credentials, "same-origin");
    assert.equal(request.keepalive, true);
  }
  assert.equal(page.status.textContent, "Advanced options saved.");
  assert.equal(page.unload(), false);
});

test("false flags, zero thinking tokens, empty notes and model are saved explicitly", async () => {
  const page = setup();
  for (const field of [input("enable_proofreading", false, "checkbox"), input("thinking_budget_tokens", 0),
    input("translation_notes", "", "textarea"), input("vlm_model", "", "select-one", false)]) page.change(field);
  await settle();
  assert.deepEqual(page.requests.map(([, request]) => request.body),
    [{proofread_translations: false}, {thinking_budget_tokens: 0}, {translation_notes: ""}, {vlm_model: ""}]);
});

test("typing is debounced and change does not submit the same value twice", async () => {
  const page = setup(), field = input("imagemagick_workers", 1);
  for (const value of [1, 2, 3]) { field.value = String(value); page.type(field); }
  assert.equal(page.timers.size, 1);
  assert.equal(page.requests.length, 0);
  await page.flush();
  page.change(field);
  await settle();
  assert.deepEqual(page.requests.map(([, request]) => request.body), [{imagemagick_workers: 3}]);
});

test("requests run in order and do not overwrite unrelated fields", async () => {
  let release;
  const page = setup(() => new Promise((resolve) => { release = () => resolve({ok: true}); }));
  page.change(input("lama_workers", 2));
  page.change(input("lama_workers", 6));
  await settle();
  assert.equal(page.requests.length, 1);
  assert.equal(page.unload(), true);
  release(); await settle();
  assert.equal(page.requests.length, 2);
  assert.deepEqual(page.requests[1][1].body, {lama_workers: 6});
  release(); await settle();
  assert.equal(page.unload(), false);
});

test("failed saves stay visible across unrelated successes and can be retried", async () => {
  let fail = true;
  const page = setup((_url, request) => ({ok: !fail || !request.body.includes("lama_workers"), status: 400}));
  page.change(input("lama_workers", 2));
  page.change(input("imagemagick_workers", 3));
  await settle();
  assert.match(page.status.textContent, /not saved/);
  assert.equal(page.unload(), true);
  fail = false;
  page.change(input("lama_workers", 2));
  await settle();
  assert.equal(page.status.textContent, "Advanced options saved.");
  assert.equal(page.unload(), false);
});

test("invalid current input is not cleared by an older successful save", async () => {
  let release;
  const page = setup(() => new Promise((resolve) => { release = () => resolve({ok: true}); }));
  page.change(input("lama_workers", 4));
  await settle();
  page.change(input("lama_workers", ""));
  release(); await settle();
  assert.equal(page.requests.length, 1);
  assert.match(page.status.textContent, /not saved/);
  page.change(input("lama_workers", 4));
  await settle();
  release(); await settle();
  assert.equal(page.status.textContent, "Advanced options saved.");
});

test("invalid values and passwords never get sent as category defaults", async () => {
  const page = setup();
  page.change(input("ocr_page_workers", 0, "number", false));
  page.change(input("vlm_auth_token", "private-key", "password"));
  page.type(input("paddleocr_vl_auth_token", "private-ocr-key", "password"));
  page.change(input("series_id", "series", "select-one"));
  await page.flush();
  assert.equal(page.requests.length, 0);
  assert.match(page.status.textContent, /Not saved/);
});

test("reload flushes pending input and warns until the save completes", async () => {
  const page = setup();
  page.type(input("lama_workers", 7));
  assert.equal(page.unload(), true);
  await settle();
  assert.deepEqual(page.requests[0][1].body, {lama_workers: 7});
  assert.equal(page.unload(), false);
  page.type(input("lama_workers", 8));
  page.submit();
  assert.equal(page.unload(), false, "a valid upload also saves its options");
  await settle();
});

test("large notes do not hit the browser keepalive request size limit", async () => {
  const page = setup();
  const notes = "Notes ".repeat(10000);
  page.change(input("translation_notes", notes, "textarea"));
  await settle();
  assert.equal(page.requests[0][1].keepalive, false);
  assert.deepEqual(page.requests[0][1].body, {translation_notes: notes});
  assert.equal(page.status.textContent, "Advanced options saved.");
});

test("a duplicate change does not hide an in-flight save failure", async () => {
  let release;
  const page = setup(() => new Promise((resolve) => { release = () => resolve({ok: false, status: 500}); }));
  const field = input("lama_workers", 4);
  page.change(field);
  await settle();
  page.change(field);
  release(); await settle();
  assert.equal(page.requests.length, 1);
  assert.match(page.status.textContent, /not saved/);
  assert.equal(page.unload(), true);
});

test("unchanged input after a completed save does not leave a Saving message", async () => {
  const page = setup(), field = input("lama_workers", 4);
  page.change(field); await settle();
  page.type(field); await page.flush();
  assert.equal(page.requests.length, 1);
  assert.equal(page.status.textContent, "Advanced options saved.");
});
