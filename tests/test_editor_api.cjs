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
