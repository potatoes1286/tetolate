const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../web_editor/api.js"), "utf8");

for (const [body, message] of [
  ['{"detail":"Could not connect to the OCR endpoint."}', "Could not connect to the OCR endpoint."],
  ['{"error":{"message":"Settings were not saved."}}', "Settings were not saved."],
  ["Internal Server Error", "Internal Server Error"],
  ["<html>Error</html>", "Internal Server Error"],
  ["", "Internal Server Error"],
]) {
  test(`failed OCR response is read once: ${body || "empty"}`, async () => {
    const response = new Response(body, {status: 500, statusText: "Internal Server Error"});
    const Api = vm.runInNewContext(source.replace("export class", "class") + "\nEditorApi", {fetch: async () => response});
    const api = new Api("category", "job");
    await assert.rejects(api.cropOcr(0, [0, 0, 10, 10], 1), (error) => error.message === message && error.status === 500);
    assert.equal(response.bodyUsed, true);
  });
}

test("successful OCR response still returns records", async () => {
  const Api = vm.runInNewContext(source.replace("export class", "class") + "\nEditorApi", {
    fetch: async () => new Response('{"rawRecords":[{"text":"Hello"}]}', {status: 200}),
  });
  assert.equal((await new Api("category", "job").cropOcr(0, [0, 0, 10, 10], 1)).rawRecords[0].text, "Hello");
});

const settle = async () => { for (let index = 0; index < 20; index++) await Promise.resolve(); };
function cleaningApi() {
  const requests = [];
  const Api = vm.runInNewContext(source.replace("export class", "class") + "\nEditorApi", {
    fetch: (url, options) => new Promise((resolve, reject) => {
      requests.push({url, options, body: JSON.parse(options.body), reject,
        finish: (data) => resolve({ok: true, json: async () => data}),
        fail: () => resolve({ok: false, status: 500, statusText: "Cleaning failed", text: async () => "Cleaning failed"})});
    }),
  });
  return {api: new Api("category", "job"), requests};
}

test("Current preview sends an immutable copy of draft translations", async () => {
  const {api, requests} = cleaningApi();
  const translations = [{boxno: 0, englishText: "Draft translation"}];
  const pending = api.preview("placement", 0, [], [], [], translations);
  translations[0].englishText = "Later edit";
  assert.equal(requests[0].body.translations[0].englishText, "Draft translation");
  requests[0].finish({url: "current-draft"});
  assert.equal((await pending).url, "current-draft");
});

test("a burst keeps the active request and only the newest immutable waiting mask", async () => {
  const {api, requests} = cleaningApi();
  const first = api.cleanPreview(0, [], [{region: [10, 10, 30, 30]}]);
  const skipped = [];
  for (let index = 1; index < 4; index++) {
    const boxes = [{region: [index * 10, 20, 80, 90]}];
    const pending = api.cleanPreview(index, [], boxes);
    boxes[0].region[0] = 999;
    skipped.push(pending.catch((error) => error));
  }
  assert.equal(requests.length, 1);
  requests[0].finish({url: "first"});
  assert.equal((await first).url, "first"); await settle();
  assert.equal(requests.length, 2);
  assert.match(requests[1].url, /pages\/3\/clean-preview$/);
  assert.equal(requests[1].body.cleanBoxes[0].region[0], 30);
  requests[1].finish({url: "last"});
  const results = await Promise.all(skipped);
  assert.deepEqual(results.map((value) => value.name || value.url), ["AbortError", "AbortError", "last"]);
});

test("caller abort hides an active result without releasing the network slot", async () => {
  const {api, requests} = cleaningApi(), controller = new AbortController();
  const first = api.cleanPreview(0, [], [], controller.signal);
  const rejected = assert.rejects(first, {name: "AbortError"});
  controller.abort(); await rejected;
  const next = api.cleanPreview(1, [], []);
  assert.equal(requests.length, 1);
  assert.equal(requests[0].options.signal, undefined);
  requests[0].finish({url: "discarded"}); await settle();
  assert.equal(requests.length, 2);
  requests[1].finish({url: "current"});
  assert.equal((await next).url, "current");
});

test("Preview and Current share one cleaning slot", async () => {
  for (const currentFirst of [false, true]) {
    const {api, requests} = cleaningApi();
    const current = () => api.preview("placement", 2, [{boxno: 0}], [], [], []);
    const preview = () => api.cleanPreview(3, [], []);
    const first = currentFirst ? current() : preview();
    const second = currentFirst ? preview() : current();
    assert.equal(requests.length, 1);
    requests[0].finish({url: "first"}); await first; await settle();
    assert.equal(requests.length, 2);
    assert.match(requests[1].url, currentFirst ? /clean-preview$/ : /stages\/placement\/preview$/);
    requests[1].finish({url: "second"}); await second;
  }
});

test("hiding Preview cancels the waiting draft and does not send it later", async () => {
  const {api, requests} = cleaningApi(), controller = new AbortController();
  const first = api.cleanPreview(0, [], []);
  const pending = api.cleanPreview(1, [], [], controller.signal);
  const rejected = assert.rejects(pending, {name: "AbortError"});
  controller.abort(); await rejected;
  requests[0].finish({url: "first"}); await first; await settle();
  assert.equal(requests.length, 1);
  assert.equal(api.pendingCleaning, null);
  assert.equal(api.cleaning, false);
});

test("an already aborted request neither sends work nor replaces a valid waiting draft", async () => {
  const {api, requests} = cleaningApi(), controller = new AbortController();
  const first = api.cleanPreview(0, [], []);
  const waiting = api.cleanPreview(1, [], []);
  controller.abort();
  await assert.rejects(api.cleanPreview(2, [], [], controller.signal), {name: "AbortError"});
  requests[0].finish({url: "first"}); await first; await settle();
  assert.equal(requests.length, 2);
  assert.match(requests[1].url, /pages\/1\/clean-preview$/);
  requests[1].finish({url: "waiting"}); await waiting;
});

test("HTTP and network errors release the slot and start the newest draft", async () => {
  for (const network of [false, true]) {
    const {api, requests} = cleaningApi();
    const first = api.cleanPreview(0, [], []);
    const failed = assert.rejects(first, /failed/i);
    const waiting = api.cleanPreview(1, [], []);
    network ? requests[0].reject(new Error("Network failed")) : requests[0].fail();
    await failed; await settle();
    assert.equal(requests.length, 2);
    requests[1].finish({url: "recovered"});
    assert.equal((await waiting).url, "recovered");
    await settle(); assert.equal(api.cleaning, false);
  }
});

test("cleaning queues are separate for separate editor instances", async () => {
  const first = cleaningApi(), second = cleaningApi();
  const a = first.api.cleanPreview(0, [], []), b = second.api.cleanPreview(1, [], []);
  assert.equal(first.requests.length, 1); assert.equal(second.requests.length, 1);
  first.requests[0].finish({url: "a"}); second.requests[0].finish({url: "b"});
  assert.equal((await a).url, "a"); assert.equal((await b).url, "b");
});
