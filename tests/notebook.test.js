import test from "node:test";
import assert from "node:assert/strict";
import { deflateSync } from "node:zlib";
import { GET } from "../api/notebook.js";
import { createKaggleUrl, publicNotebookBase, MAX_NOTEBOOK_PAYLOAD_BYTES, MAX_NOTEBOOK_URL_LENGTH } from "../src/utils/notebook.js";
import { cnnCatalog, transformerCatalog, unetCatalog, odCatalog, mlCatalog, lossCatalog } from "../src/data/catalogs.js";
import { imageDatasets, csvDatasets } from "../src/data/datasets.js";
import { generateTorchCode, generateTransformerCode, generateUnetCode, generateODCode, generateSklearnCode } from "../src/utils/codeGen.js";

const fixtures = [
  ["cnn", cnnCatalog, ["input", "conv", "resnet", "pool", "gap", "head"], generateTorchCode],
  ["transformer", transformerCatalog, ["t_input", "patch_embed", "pos_embed", "cls_token", "transformer_block", "t_cls_pool", "t_head"], generateTransformerCode],
  ["unet", unetCatalog, ["u_input", "u_enc", "u_enc", "u_bottleneck", "u_dec", "u_dec", "u_seg_head"], generateUnetCode],
  ["od", odCatalog, ["od_input", "cspdarknet", "panet", "yolo_head"], generateODCode],
  ["ml", mlCatalog, ["tabular_input", "standard_scaler", "random_forest"], generateSklearnCode],
];

for (const [mode, catalog, kinds, generate] of fixtures) {
  test(`${mode}: browser compression → Kaggle source URL → API preserves the full generated code`, async () => {
    const graph = kinds.map((kind, index) => ({ ...structuredClone(catalog.find(b => b.kind === kind)), id: `block-${index}`, attention: [] }));
    const dataset = mode === "ml" ? csvDatasets[0] : imageDatasets.find(d => d.compatible.includes(mode));
    const settings = { epochs: 17, batch_size: 16, lr: 0.002, optimizer: "Adam", visualizations: ["plot_loss"] };
    const code = mode === "ml" ? generate(graph, lossCatalog.ml[0], settings, dataset)
      : generate(graph, dataset, ["Resize", "Normalize"], lossCatalog[mode][0], settings, "None");
    const { notebookUrl, kaggleUrl } = await createKaggleUrl({ mode, datasetName: dataset.name, code }, "https://forge.example");
    assert.ok(kaggleUrl.length <= MAX_NOTEBOOK_URL_LENGTH);
    assert.equal(new URL(kaggleUrl).searchParams.get("src"), notebookUrl);
    assert.equal(new URL(notebookUrl).pathname, "/api/notebook.ipynb");
    const response = GET(new Request(notebookUrl));
    assert.equal(response.status, 200);
    assert.equal(response.headers.get("Content-Disposition"), `attachment; filename="model_${mode}.ipynb"`);
    const notebook = await response.json();
    assert.equal(notebook.nbformat, 4);
    assert.equal(notebook.metadata.kernelspec.name, "python3");
    assert.equal(notebook.cells[1].source, code);
    assert.equal(notebook.cells[1].execution_count, null);
    assert.deepEqual(notebook.cells[1].outputs, []);
    assert.ok(notebook.cells[0].source.includes(dataset.name));
  });
}

test("Unicode, newlines and special characters survive import", async () => {
  const payload = { mode: "cnn", datasetName: "细胞 🧬", code: '# α & ? / + =\nprint("🧬")\n' };
  const { notebookUrl } = await createKaggleUrl(payload, "https://forge.example");
  assert.equal((await GET(new Request(notebookUrl)).json()).cells[1].source, payload.code);
});

test("invalid, truncated, oversized, and decompression-bomb links return 400", () => {
  const encode = value => deflateSync(value).toString("base64url");
  const invalid = ["", "not-deflate", "%", "a".repeat(MAX_NOTEBOOK_URL_LENGTH + 1),
    encode("not JSON"), encode("null"), encode(JSON.stringify({ mode: "cnn", code: "" })),
    encode("x".repeat(MAX_NOTEBOOK_PAYLOAD_BYTES + 1))];
  for (const data of invalid) {
    const response = GET(new Request(`https://forge.example/api/notebook.ipynb?data=${encodeURIComponent(data)}`));
    assert.equal(response.status, 400);
    assert.equal(response.headers.get("Cache-Control"), "no-store");
  }
});

test("localhost and non-public URLs show an actionable deployment error", () => {
  for (const base of ["http://localhost:3000", "https://localhost", "https://127.0.0.1", "https://[::1]", "https://192.168.0.1", "https://forge.local", "https://user:pass@forge.example"]) {
    assert.throws(() => publicNotebookBase(base), /public HTTPS deployment/);
  }
  assert.equal(publicNotebookBase("https://forge.example/ignored"), "https://forge.example");
});

test("oversized code is rejected with a manual download option", async () => {
  await assert.rejects(createKaggleUrl({ mode: "cnn", datasetName: "test", code: "x".repeat(MAX_NOTEBOOK_PAYLOAD_BYTES) }, "https://forge.example"), /Download Code/);
});
