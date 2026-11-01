export const MAX_NOTEBOOK_PAYLOAD_BYTES = 128 * 1024;
export const MAX_NOTEBOOK_URL_LENGTH = 7000;
export const notebookModes = ["cnn", "transformer", "unet", "od", "ml"];

export function createNotebook({ code, mode, datasetName }) {
  if (typeof code !== "string" || !code.trim() || !notebookModes.includes(mode) ||
      typeof datasetName !== "string" || datasetName.length > 200) {
    throw new Error("Invalid notebook content.");
  }
  return {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      kernelspec: { display_name: "Python 3", language: "python", name: "python3" },
      language_info: { name: "python", file_extension: ".py", mimetype: "text/x-python" },
    },
    cells: [
      {
        id: "introduction", cell_type: "markdown", metadata: {},
        source: `# Model Forge — ${mode.toUpperCase()}\n\nDataset: ${datasetName}\n\nGenerated from your current model and training settings. Select **Run All** to execute the code. Dataset loaders and advanced block implementations marked as placeholders in the code still need to be supplied.`,
      },
      { id: "model", cell_type: "code", metadata: {}, execution_count: null, outputs: [], source: code },
    ],
  };
}

export function publicNotebookBase(value) {
  const url = new URL(value);
  const host = url.hostname.toLowerCase();
  if (url.protocol !== "https:" || url.username || url.password ||
      host === "localhost" || host.endsWith(".localhost") || host.endsWith(".local") ||
      host.includes(":") || /^\d+\.\d+\.\d+\.\d+$/.test(host)) {
    throw new Error("Kaggle needs a public HTTPS deployment. Open the deployed app to use Run in Kaggle.");
  }
  return url.origin;
}

export async function createKaggleUrl(payload, baseUrl) {
  const base = publicNotebookBase(baseUrl);
  const bytes = new TextEncoder().encode(JSON.stringify(payload));
  if (bytes.length > MAX_NOTEBOOK_PAYLOAD_BYTES) {
    throw new Error("This notebook is too large for a Kaggle link. Use Download Code to import it manually.");
  }
  if (typeof CompressionStream === "undefined") {
    throw new Error("Use a current browser for Run in Kaggle, or use Download Code to import manually.");
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("deflate"));
  const compressed = new Uint8Array(await new Response(stream).arrayBuffer());
  const encoded = btoa(Array.from(compressed, byte => String.fromCharCode(byte)).join(""))
    .replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
  const notebookUrl = new URL("/api/notebook.ipynb", base);
  notebookUrl.searchParams.set("data", encoded);
  const kaggleUrl = new URL("https://www.kaggle.com/kernels/welcome");
  kaggleUrl.searchParams.set("src", notebookUrl.href);
  if (kaggleUrl.href.length > MAX_NOTEBOOK_URL_LENGTH) {
    throw new Error("This notebook is too large for a Kaggle link. Use Download Code to import it manually.");
  }
  return { notebookUrl: notebookUrl.href, kaggleUrl: kaggleUrl.href };
}
