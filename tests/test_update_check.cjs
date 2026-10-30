// Run with: node tests/test_update_check.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");

// Execute the same inline script that base_page serves, without Python or a server.
const source = fs.readFileSync(path.join(__dirname, "../web_pages.py"), "utf8");
const script = source.match(/UPDATE_CHECK_SCRIPT = r"""\s*<script>([\s\S]*?)<\/script>\s*"""/)[1];

function setup(fetch, version = "1.0.0") {
  const attributes = new Map();
  let onClick;
  let onTimeout;
  let timerCleared = false;
  const check = {
    dataset: {version},
    addEventListener: (type, handler) => { assert.equal(type, "click"); onClick = handler; },
    setAttribute: (name, value) => attributes.set(name, value),
    removeAttribute: (name) => attributes.delete(name),
  };
  const status = {
    textContent: "",
    replaceChildren: (...children) => { status.children = children; },
  };
  vm.runInNewContext(script, {
    document: {
      getElementById: (id) => id === "check-updates" ? check : status,
      createElement: (tag) => { assert.equal(tag, "a"); return {}; },
    },
    fetch, AbortController,
    setTimeout: (callback, delay) => { assert.equal(delay, 10000); onTimeout = callback; return 1; },
    clearTimeout: (id) => { assert.equal(id, 1); timerCleared = true; },
  });
  return {
    status, attributes,
    click: () => onClick({preventDefault() {}}),
    timeout: () => onTimeout(),
    timerCleared: () => timerCleared,
  };
}

function response(tag, options = {}) {
  return {ok: true, status: 200, json: async () => ({tag_name: tag, ...options})};
}

test("checks only on click, sends no credentials, and links a newer release", async () => {
  let calls = 0;
  const page = setup(async (url, options) => {
    calls++;
    assert.equal(url, "https://api.github.com/repos/potatoes1286/tetolate/releases/latest");
    assert.equal(options.credentials, "omit");
    assert.equal(options.cache, "no-store");
    assert.equal(options.headers.Accept, "application/vnd.github+json");
    return response("v1.1.0", {name: "<script>not trusted</script>", html_url: "javascript:bad()"});
  });
  assert.equal(calls, 0);
  await page.click();
  assert.equal(calls, 1);
  const [label, link] = page.status.children;
  assert.equal(label, "Newer update: ");
  assert.equal(link.textContent, "v1.1.0");
  assert.equal(link.href, "https://github.com/potatoes1286/tetolate/releases/tag/v1.1.0");
  assert.equal(link.rel, "noopener noreferrer");
  assert.equal(page.attributes.has("aria-disabled"), false);
  assert.equal(page.timerCleared(), true);
});

for (const [current, latest, newer] of [
  ["1.0.0", "v1.0.0", false], ["1.0.0", "v0.9.9", false],
  ["1.9.0", "v1.10.0", true], ["1.10.0", "v1.9.0", false],
  ["1.0.9", "1.0.10", true], ["1.9.9", "v2.0.0", true],
]) {
  test(`compares ${current} with ${latest} numerically`, async () => {
    const page = setup(async () => response(latest), current);
    await page.click();
    if (newer) assert.equal(page.status.children[0], "Newer update: ");
    else assert.equal(page.status.textContent, "Up to date.");
  });
}

test("reports no published releases", async () => {
  const page = setup(async () => ({ok: false, status: 404}));
  await page.click();
  assert.equal(page.status.textContent, "No published releases found.");
  assert.equal(page.timerCleared(), true);
});

for (const [name, fetch] of [
  ["network failure", async () => { throw new Error("offline"); }],
  ["rate limit", async () => ({ok: false, status: 403})],
  ["server failure", async () => ({ok: false, status: 503})],
  ["invalid JSON", async () => ({ok: true, json: async () => { throw new Error("JSON"); }})],
  ["invalid tag", async () => response("<script>bad</script>")],
  ["missing tag", async () => response(undefined)],
  ["null payload", async () => ({ok: true, json: async () => null})],
  ["prerelease", async () => response("v2.0.0", {prerelease: true})],
  ["draft", async () => response("v2.0.0", {draft: true})],
]) {
  test(`handles ${name} and permits retry`, async () => {
    let calls = 0;
    const page = setup(async (...args) => ++calls === 1 ? fetch(...args) : response("v1.0.0"));
    await page.click();
    assert.equal(page.status.textContent, "Could not check for updates. Try again.");
    assert.equal(page.attributes.has("aria-disabled"), false);
    assert.equal(page.timerCleared(), true);
    await page.click();
    assert.equal(page.status.textContent, "Up to date.");
  });
}

test("ignores repeated clicks while busy and stops after a timeout", async () => {
  let calls = 0;
  const page = setup((_url, {signal}) => {
    calls++;
    return new Promise((_resolve, reject) => signal.addEventListener("abort", () => reject(new Error("timeout"))));
  });
  const pending = page.click();
  assert.equal(page.status.textContent, "Checking...");
  assert.equal(page.attributes.get("aria-disabled"), "true");
  await page.click();
  assert.equal(calls, 1);
  page.timeout();
  await pending;
  assert.equal(page.status.textContent, "Could not check for updates. Try again.");
  assert.equal(page.attributes.has("aria-disabled"), false);
  assert.equal(page.timerCleared(), true);
});
