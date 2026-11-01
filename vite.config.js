import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { GET } from "./api/notebook.js";

function notebookApi() {
  const middleware = async (req, res, next) => {
    const url = new URL(req.url, "http://localhost");
    if (!["/api/notebook", "/api/notebook.ipynb"].includes(url.pathname)) return next();
    if (!["GET", "HEAD"].includes(req.method)) {
      res.writeHead(405, { Allow: "GET, HEAD" });
      res.end();
      return;
    }
    const response = GET(new Request(url));
    res.writeHead(response.status, Object.fromEntries(response.headers));
    res.end(req.method === "HEAD" ? undefined : await response.text());
  };
  return {
    name: "notebook-api",
    configureServer(server) { server.middlewares.use(middleware); },
    configurePreviewServer(server) { server.middlewares.use(middleware); },
  };
}

export default defineConfig({
  plugins: [react(), notebookApi()],
  server: {
    port: 3000,
  },
});
