const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");
const vm = require("node:vm");

const app = fs.readFileSync(path.join(__dirname, "../web_editor/app.js"), "utf8");
const source = app.match(/function initEditorHeightResizer\([^]*?\n\}/)[0];

function setup() {
  const handlers = {}, attributes = {}, properties = {}, frames = new Map(), scrolls = [], classes = new Set();
  let nextFrame = 0, focused = false, captured = null;
  const handle = {addEventListener: (event, callback) => {handlers[event] = callback;},
    setAttribute: (name, value) => {attributes[name] = value;},
    setPointerCapture: (id) => {captured = id;}, focus: (options) => {focused = options.preventScroll;},
    classList: {add: (name) => classes.add(name), remove: (name) => classes.delete(name)}};
  const context = {byId: () => handle, document: {body: {style: {setProperty: (name, value) => {properties[name] = value;}}}},
    window: {innerHeight: 900, scrollBy: (_x, y) => scrolls.push(y)},
    requestAnimationFrame: (callback) => {frames.set(++nextFrame, callback); return nextFrame;},
    cancelAnimationFrame: (id) => frames.delete(id)};
  vm.runInNewContext(source + "\ninitEditorHeightResizer();", context);
  const pointer = (event, y, button = 0) => handlers[event]({clientY: y, button, pointerId: 1, preventDefault() {}});
  const key = (key) => {let prevented = false; handlers.keydown({key, preventDefault: () => {prevented = true;}}); return prevented;};
  const advance = () => {for (const [id, callback] of [...frames]) {frames.delete(id); callback();}};
  const height = () => Number.parseInt(properties["--editor-extra-height"] || "0", 10);
  return {handlers, pointer, key, advance, height, frames, classes, attributes, scrolls, focused: () => focused, captured: () => captured};
}

test("dragging the bottom edge expands and contracts only editor height", () => {
  const ui = setup();
  ui.pointer("pointerdown", 600);
  assert.equal(ui.focused(), true); assert.equal(ui.captured(), 1);
  ui.pointer("pointermove", 720); assert.equal(ui.height(), 120);
  assert.equal(ui.attributes["aria-valuenow"], "120");
  ui.pointer("pointermove", 660); assert.equal(ui.height(), 60);
  ui.pointer("pointermove", 500); assert.equal(ui.height(), 0);
  ui.pointer("pointerup", 500);
  assert.equal(ui.classes.has("dragging"), false);
  assert.equal(ui.frames.size, 0);
  ui.pointer("pointermove", 700); assert.equal(ui.height(), 0);
});

test("holding a downward drag at the window edge extends the editor and scrolls the document", () => {
  const ui = setup();
  ui.pointer("pointerdown", 860); ui.advance(); assert.equal(ui.height(), 0);
  ui.pointer("pointermove", 880); ui.advance();
  assert.equal(ui.height(), 32); assert.deepEqual(ui.scrolls, [12]);
  ui.advance(); assert.equal(ui.height(), 44); assert.deepEqual(ui.scrolls, [12, 12]);
  ui.pointer("pointermove", 800); ui.advance();
  assert.equal(ui.height(), 0); assert.equal(ui.frames.size, 0);
});

test("pointer release, cancellation and capture loss stop automatic extension", () => {
  for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) {
    const ui = setup(); ui.pointer("pointerdown", 860); ui.pointer("pointermove", 880); ui.advance();
    ui.pointer(event, 880); const height = ui.height(); ui.advance();
    assert.equal(ui.height(), height); assert.equal(ui.frames.size, 0);
    assert.equal(ui.classes.has("dragging"), false);
  }
});

test("arrow keys resize, Home resets, and unrelated keys are not intercepted", () => {
  const ui = setup();
  assert.equal(ui.key("ArrowDown"), true); assert.equal(ui.height(), 40);
  ui.key("ArrowDown"); ui.key("ArrowUp"); assert.equal(ui.height(), 40);
  assert.equal(ui.attributes["aria-valuetext"], "40 extra pixels");
  ui.key("Home"); assert.equal(ui.height(), 0);
  ui.key("ArrowUp"); assert.equal(ui.height(), 0);
  assert.equal(ui.key("h"), false);
});

test("double-click restores default height and stops an active drag", () => {
  const ui = setup(); ui.pointer("pointerdown", 860); ui.pointer("pointermove", 880); ui.advance();
  ui.handlers.dblclick(); assert.equal(ui.height(), 0); assert.equal(ui.frames.size, 0);
  ui.pointer("pointermove", 890); assert.equal(ui.height(), 0);
});

test("non-primary presses are ignored and height stays within its accessible bounds", () => {
  const ui = setup(); ui.pointer("pointerdown", 600, 2); ui.pointer("pointermove", 650);
  assert.equal(ui.height(), 0); assert.equal(ui.captured(), null);
  ui.pointer("pointerdown", 0); ui.pointer("pointermove", 25000); ui.advance();
  assert.equal(ui.height(), 20000); assert.equal(ui.frames.size, 0);
  ui.key("ArrowDown"); assert.equal(ui.height(), 20000);
});
