import { inflateSync } from "node:zlib";
import { createNotebook, MAX_NOTEBOOK_PAYLOAD_BYTES, MAX_NOTEBOOK_URL_LENGTH } from "../src/utils/notebook.js";

const headers = {
  "Cache-Control": "no-store",
  "Access-Control-Allow-Origin": "*",
  "X-Content-Type-Options": "nosniff",
};

export function GET(request) {
  const encoded = new URL(request.url).searchParams.get("data");
  if (!encoded || encoded.length > MAX_NOTEBOOK_URL_LENGTH || !/^[A-Za-z0-9_-]+$/.test(encoded)) {
    return Response.json({ error: "Missing or invalid notebook data." }, { status: 400, headers });
  }
  try {
    const json = inflateSync(Buffer.from(encoded, "base64url"), { maxOutputLength: MAX_NOTEBOOK_PAYLOAD_BYTES });
    const payload = JSON.parse(json.toString("utf8"));
    const notebook = createNotebook(payload);
    return Response.json(notebook, {
      headers: { ...headers, "Content-Disposition": `attachment; filename="model_${payload.mode}.ipynb"` },
    });
  } catch {
    return Response.json({ error: "The notebook link is invalid or too large." }, { status: 400, headers });
  }
}
