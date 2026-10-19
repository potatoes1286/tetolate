// Run with: node tests/test_editor_canvas.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");
const root = path.join(__dirname, "../web_editor");
const screens = vm.runInNewContext(fs.readFileSync(path.join(root, "screens.js"), "utf8").replaceAll("export ", "") + "\n({SCREEN, ALIGNMENT_LABELS, fieldsFor})");
const source = fs.readFileSync(path.join(root, "canvas.js"), "utf8").replace(/^import .*\n/, "").replace("export class", "class") + "\nEditorCanvas";

function setup({failure = false, render = null, view = true} = {}) {
  const requests = [], images = [], status = [], timers = new Map();
  let nextTimer = 0, closed = 0;
  const ctx = {clearRect() {}, drawImage: (...args) => images.push(args), strokeRect() {}, fillRect() {}, fillText() {}};
  const canvas = {width: 200, height: 200, style: {}, getContext: () => ctx,
    addEventListener() {}, getBoundingClientRect: () => ({width: 200, height: 200})};
  const store = {stage: "placement", page: 0, document: {defaultFont: "Default.ttf"}, addEventListener() {},
    records: {translations: [{boxno: 0, englishText: "Hello"}],
      placements: [{recordId: "zero", boxno: 0, font: "Comic.ttf", gravity: "center", placementRegion: [20, 20, 180, 180], fontSizeWidthPercent: 10}]}};
  const Canvas = vm.runInNewContext(source, {SCREEN: screens.SCREEN, Image: class {}, AbortController,
    setTimeout: (callback, delay) => { assert.equal(delay, 200); timers.set(++nextTimer, callback); return nextTimer; },
    clearTimeout: (id) => timers.delete(id)});
  const editor = new Canvas(canvas, store, (...args) => status.push(args), async (page, entries, signal) => {
    requests.push([page, structuredClone(entries), signal]);
    if (failure) throw new Error("Missing ImageMagick");
    return render ? render(page, entries) : entries.map((entry) => ({entry: structuredClone(entry), width: entry.placementRegion[2] + 16, height: entry.placementRegion[3] + 16, close() { closed++; }}));
  });
  editor.image.complete = true;
  editor.showPlacementText = view;
  const advance = async () => {
    for (const [id, callback] of [...timers]) { timers.delete(id); await callback(); }
  };
  return {editor, store, canvas, requests, images, status, timers, advance, closed: () => closed};
}

test("alignment labels are plain directions and renderer values stay unchanged", () => {
  assert.deepEqual(Object.values(screens.ALIGNMENT_LABELS), ["Up", "Center", "Down", "Left", "Right", "Up left", "Up right", "Down left", "Down right"]);
  const field = screens.fieldsFor("placement", {gravity: "west"}).find(([name]) => name === "gravity");
  assert.equal(field[3], "west");
  assert.equal(screens.ALIGNMENT_LABELS[field[3]], "Left");
});

test("prepareTextLayers starts Magick immediately and waits for it, then reuses cached layers", async () => {
  let finish;
  const page = setup({render: () => new Promise((resolve) => { finish = resolve; })});
  let ready = false;
  const prepared = page.editor.prepareTextLayers().then(() => { ready = true; });
  assert.equal(page.requests.length, 1);
  assert.equal(ready, false);
  finish([{width: 50, height: 50, close() {}}]); await prepared;
  assert.equal(ready, true);
  await page.editor.prepareTextLayers(); assert.equal(page.requests.length, 1);
});

test("a previous page's pending Magick batch does not block the new page", async () => {
  let finish, closed = 0;
  const page = setup({render: (number, entries) => number === 0
    ? new Promise((resolve) => { finish = resolve; })
    : entries.map(() => ({width: 50, height: 50, close() {}}))});
  page.editor.draw(); const previous = page.advance();
  assert.equal(page.requests.length, 1);
  page.store.page = 1; page.store.document = {...page.store.document};
  page.editor.draw(); await page.advance();
  assert.equal(page.requests[0][2].aborted, true);
  assert.equal(page.requests.length, 2);
  assert.equal(page.requests[1][0], 1);
  finish([{width: 50, height: 50, close() { closed++; }}]); await previous;
  assert.equal(closed, 1);
  assert.equal(page.editor.renderingLayers, false);
});

test("late source-image loads cannot paint an older page", async () => {
  const page = setup(); let painted = 0;
  page.editor.draw = () => { painted++; };
  const old = page.editor.load("page-4"), oldImage = page.editor.image;
  const current = page.editor.load("page-5"), currentImage = page.editor.image;
  oldImage.onload(); assert.equal(await old, false); assert.equal(painted, 0);
  currentImage.naturalWidth = 200; currentImage.naturalHeight = 300;
  currentImage.onload(); assert.equal(await current, true); assert.equal(painted, 1);
  assert.equal(page.canvas.height, 300);
});

test("Preview debounces rendering and draws exact layer pixels with bleed, without scaling", async () => {
  const page = setup();
  page.editor.draw();
  page.editor.draw();
  assert.equal(page.requests.length, 0);
  assert.equal(page.timers.size, 1);
  await page.advance();
  assert.equal(page.requests.length, 1);
  assert.deepEqual(page.requests[0][1][0].placementRegion, [0, 0, 160, 160]);
  assert.equal(page.requests[0][1][0].englishText, "Hello");
  const [bitmap, x, y] = page.images.at(-1);
  assert.equal(bitmap.width, 176);
  assert.equal(x, 12);
  assert.equal(y, 12);
  assert.equal(page.images.at(-1).length, 3, "no canvas text stretch or resampling");
  page.editor.draw();
  await page.advance();
  assert.equal(page.requests.length, 1);
});

test("moving a box immediately reuses the existing bitmap", async () => {
  const page = setup();
  page.editor.draw();
  await page.advance();
  page.store.records.placements[0].placementRegion = [30, 30, 190, 190];
  page.editor.draw();
  assert.equal(page.requests.length, 1);
  assert.equal(page.timers.size, 0);
  assert.deepEqual(page.images.at(-1).slice(1), [22, 22]);
});

test("reloading the document refreshes font and server-default caches", async () => {
  const page = setup();
  page.editor.draw();
  await page.advance();
  page.store.document = {...page.store.document};
  page.editor.draw();
  await page.advance();
  assert.equal(page.requests.length, 2);
  assert.equal(page.closed(), 1);
});

test("rapid resizing cancels obsolete work and renders only the last size", async () => {
  const page = setup();
  for (const right of [180, 160, 150]) {
    page.store.records.placements[0].placementRegion[2] = right;
    page.editor.draw();
  }
  assert.equal(page.timers.size, 1);
  await page.advance();
  assert.equal(page.requests.length, 1);
  assert.equal(page.requests[0][1][0].placementRegion[2], 130);
});

test("font, outline, text, wrapping and alignment changes render new layers", async () => {
  const page = setup();
  page.editor.draw();
  await page.advance();
  const patches = [{font: "Other.ttf"}, {strokeWidth: 0}, {manualLineBreaks: "One\nTwo"},
    {fontSizeWidthPercent: 50}, {rotation: 25}, {skewX: -15}, {skewY: 10},
    ...Object.keys(screens.ALIGNMENT_LABELS).map((gravity) => ({gravity}))];
  for (const patch of patches) {
    Object.assign(page.store.records.placements[0], patch);
    page.editor.draw();
    await page.advance();
    for (const [key, value] of Object.entries(patch)) {
      assert.equal(page.images.at(-1)[0].entry[key === "manualLineBreaks" ? "englishText" : key], value);
    }
  }
  assert.equal(page.requests.length, patches.length, "returning to Center reuses the matching cached layer");
});

test("clipping and half-pixel rounding match the renderer's pixel grid", async () => {
  const page = setup();
  page.store.records.placements[0].placementRegion = [-10, 20.5, 180.5, 230];
  page.editor.draw();
  await page.advance();
  assert.deepEqual(page.requests[0][1][0].placementRegion, [0, 0, 180, 180]);
  assert.deepEqual(page.images.at(-1).slice(1), [-8, 12]);
});

test("expanded transformed layers keep the box center and remain cached during movement", async () => {
  const page = setup({render: (_page, entries) => entries.map(() => ({width: 244, height: 302, close() {}}))});
  Object.assign(page.store.records.placements[0], {rotation: 35, skewX: 20, skewY: -10});
  page.editor.draw();
  await page.advance();
  const entry = page.requests[0][1][0];
  assert.deepEqual([entry.rotation, entry.skewX, entry.skewY], [35, 20, -10]);
  assert.deepEqual(page.images.at(-1).slice(1), [-22, -51]);
  page.store.records.placements[0].placementRegion = [30, 30, 190, 190];
  page.editor.draw();
  await page.advance();
  assert.equal(page.requests.length, 1);
  assert.deepEqual(page.images.at(-1).slice(1), [-12, -41]);
});

test("Original and Current do not request editable text layers", async () => {
  const page = setup({view: false});
  page.editor.draw();
  await page.advance();
  assert.equal(page.requests.length, 0);
});

test("switching from Preview to Current cancels work before the new background loads", async () => {
  const page = setup();
  page.editor.draw();
  page.editor.load("/current", {showPlacementText: false});
  await page.advance();
  assert.equal(page.requests.length, 0);
});

test("deletion or leaving Typesetting cancels queued renders", async () => {
  for (const leave of [false, true]) {
    const page = setup();
    page.editor.draw();
    if (leave) page.store.stage = "ocr";
    else page.store.records.placements = [];
    page.editor.draw();
    await page.advance();
    assert.equal(page.requests.length, 0);
  }
});

test("late renders from a previous page are discarded and decoded bitmaps released", async () => {
  let resolve;
  const page = setup({render: () => new Promise((done) => {resolve = done;})});
  page.editor.draw();
  const pending = page.advance();
  await Promise.resolve();
  page.store.page = 1;
  page.store.stage = "ocr";
  page.editor.draw();
  let closed = false;
  resolve([{close() {closed = true;}}]);
  await pending;
  assert.equal(closed, true);
  assert.equal(page.images.some(([image]) => image.close), false);
});

test("render failures are visible, but are not retried on every redraw", async () => {
  const page = setup({failure: true});
  page.editor.draw();
  await page.advance();
  page.editor.draw();
  await page.advance();
  assert.equal(page.requests.length, 1);
  assert.match(page.status.at(-1)[0], /Could not render preview text: Missing ImageMagick/);
  assert.equal(page.status.at(-1)[1], true);
});

test("all missing boxes share one batch and obsolete results are released", async () => {
  let finish;
  const page = setup({render: () => new Promise((resolve) => {finish = resolve;})});
  page.store.records.placements.push({...page.store.records.placements[0], boxno: 1, font: "Other.ttf"});
  page.store.records.translations.push({boxno: 1, englishText: "Second"});
  page.editor.draw();
  const pending = Promise.all([...page.timers.values()].map((callback) => callback()));
  page.timers.clear();
  await Promise.resolve();
  assert.equal(page.requests.length, 1);
  assert.equal(page.requests[0][1].length, 2);
  page.store.records.placements.pop();
  page.editor.draw();
  let closed = 0;
  finish([{width: 176, height: 176, close() {closed++;}}, {close() {closed++;}}]);
  await pending;
  assert.equal(page.requests.length, 1);
  assert.equal(closed, 1);
});

test("layer cache is bounded and evicted bitmaps are released", async () => {
  const page = setup();
  for (let index = 0; index < 130; index++) {
    page.store.records.translations[0].englishText = `Text ${index}`;
    page.editor.draw();
    await page.advance();
  }
  assert.equal(page.editor.layers.size, 128);
  assert.equal(page.closed(), 2);
});

test("API posts one placement batch and decodes all PNGs", async () => {
  const requests = [];
  const apiSource = fs.readFileSync(path.join(root, "api.js"), "utf8").replace("export class", "class") + "\nEditorApi";
  const Api = vm.runInNewContext(apiSource, {fetch: async (url, options) => {
    requests.push([url, options]);
    return {ok: true, json: async () => ({layers: [btoa("PNG1"), btoa("PNG2")]})};
  }, Blob, atob, createImageBitmap: async (blob) => {assert.equal(blob.type, "image/png"); return blob.text();}});
  const records = [{englishText: "Hello"}, {englishText: "Second"}];
  const result = await new Api("default", "12345678").textLayers(2, records);
  assert.deepEqual(Array.from(result), ["PNG1", "PNG2"]);
  assert.match(requests[0][0], /pages\/2\/text-layers$/);
  assert.equal(requests[0][1].method, "POST");
  assert.deepEqual(JSON.parse(requests[0][1].body), {records});
});

test("only newly uncached boxes are sent after an edit", async () => {
  const page = setup();
  page.store.records.placements.push({...page.store.records.placements[0], boxno: 1, font: "Other.ttf"});
  page.store.records.translations.push({boxno: 1, englishText: "Second"});
  page.editor.draw();
  await page.advance();
  assert.equal(page.requests[0][1].length, 2);
  page.store.records.translations[1].englishText = "Changed";
  page.editor.draw();
  await page.advance();
  assert.equal(page.requests[1][1].length, 1);
  assert.equal(page.requests[1][1][0].englishText, "Changed");
});

test("identical text layers are requested once and reused at separate positions", async () => {
  const page = setup();
  page.store.records.placements.push({...page.store.records.placements[0], boxno: 1, placementRegion: [30, 30, 190, 190]});
  page.store.records.translations.push({boxno: 1, englishText: "Hello"});
  page.editor.draw();
  await page.advance();
  assert.equal(page.requests[0][1].length, 1);
  const rendered = page.images.filter(([image]) => image.entry);
  assert.deepEqual(rendered.map((item) => item.slice(1)), [[12, 12], [22, 22]]);
  assert.equal(rendered[0][0], rendered[1][0]);
});

test("large pages split into bounded batches without rendering cached boxes again", async () => {
  const page = setup();
  const template = page.store.records.placements[0];
  page.store.records.placements = Array.from({length: 300}, (_, boxno) => ({...template, boxno}));
  page.store.records.translations = Array.from({length: 300}, (_, boxno) => ({boxno, englishText: `Box ${boxno}`}));
  page.editor.draw();
  await page.advance();
  await page.advance();
  assert.deepEqual(page.requests.map((request) => request[1].length), [256, 44]);
  assert.equal(page.timers.size, 0);
});

test("edits during a batch wait, then send only the latest state", async () => {
  let finish;
  const page = setup({render: () => new Promise((resolve) => {finish = resolve;})});
  page.editor.draw();
  const first = page.advance();
  page.store.records.translations[0].englishText = "Old edit";
  page.editor.draw();
  page.store.records.translations[0].englishText = "Latest edit";
  page.editor.draw();
  assert.equal(page.requests.length, 1);
  finish([{close() {}}]);
  await first;
  const second = page.advance();
  assert.equal(page.requests.length, 2);
  assert.equal(page.requests[1][1][0].englishText, "Latest edit");
  finish([{width: 176, height: 176, close() {}}]);
  await second;
});

test("a partial PNG decode failure releases successful bitmaps", async () => {
  let closed = 0;
  const apiSource = fs.readFileSync(path.join(root, "api.js"), "utf8").replace("export class", "class") + "\nEditorApi";
  const Api = vm.runInNewContext(apiSource, {fetch: async () => ({ok: true, json: async () => ({layers: [btoa("OK"), btoa("BAD")]})}),
    Blob, atob, createImageBitmap: async (blob) => {
      if (await blob.text() === "BAD") throw new Error("bad PNG");
      return {close() {closed++;}};
    }});
  await assert.rejects(new Api("default", "12345678").textLayers(0, [{}, {}]), /bad PNG/);
  assert.equal(closed, 1);
});
