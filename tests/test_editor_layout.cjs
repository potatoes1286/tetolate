const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {test} = require("node:test");

const css = fs.readFileSync(path.join(__dirname, "../web_editor/editor.css"), "utf8");

test("desktop editor height is bounded and its panes scroll rather than expand the document", () => {
  assert.match(css, /body\s*\{[^}]*height:\s*calc\(100dvh \+ var\(--editor-extra-height, 0px\)\)[^}]*display:\s*flex/);
  assert.match(css, /\.editor-app\s*\{[^}]*flex:\s*1;[^}]*min-height:\s*0/);
  assert.match(css, /\.editor-workspace\s*\{[^}]*min-height:\s*0;[^}]*grid-template-rows:\s*minmax\(0, 1fr\)/);
  assert.match(css, /\.record-pane, \.inspector-pane\s*\{[^}]*min-height:\s*0;[^}]*overflow:\s*auto/);
  assert.match(css, /\.canvas-wrap\s*\{[^}]*min-height:\s*0;[^}]*overflow:\s*auto/);
});

test("compact layouts reserve inspector space even when there is no selected record", () => {
  const tablet = css.slice(css.indexOf("@media (max-width: 1050px)"), css.indexOf("@media (max-width: 700px)"));
  assert.match(tablet, /grid-template-rows:\s*minmax\(0, 3fr\) minmax\(0, 2fr\)/);
  const phone = css.slice(css.indexOf("@media (max-width: 700px)"));
  assert.match(phone, /body\s*\{[^}]*height:\s*auto/);
  assert.match(phone, /grid-template-rows:\s*7rem calc\(max\(30rem, 65dvh\) \+ var\(--editor-extra-height, 0px\)\) minmax\(20rem, 50dvh\)/);
});

test("stage actions stay on one full-width row and field protection states remain visible", () => {
  assert.match(css, /\.stage-tools\s*\{[^}]*flex-wrap:\s*nowrap;[^}]*overflow-x:\s*auto/);
  assert.match(css, /\.stage-tools > \*\s*\{[^}]*flex-shrink:\s*0;[^}]*white-space:\s*nowrap/);
  assert.match(css, /\.field-pair\s*\{[^}]*grid-template-columns:\s*repeat\(2, minmax\(0, 1fr\)\)/);
  assert.match(css, /button\[aria-pressed="true"\]\s*\{[^}]*background:/);
  assert.match(css, /\.reset-field:hover, \.reset-field:focus-visible\s*\{[^}]*background:/);
});
