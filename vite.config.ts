import { defineConfig, type Connect, type Plugin } from "vite";
import jev from "./netlify/functions/jev.ts";

// Serve the Netlify Function locally too, so `bun run dev` and `preview` can reach Jev.
const handle: Connect.NextHandleFunction = async (req, res) => {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  const response = await jev(
    new Request(`http://localhost${req.url}`, {
      method: req.method,
      headers: req.headers as Record<string, string>,
      body: req.method === "POST" ? Buffer.concat(chunks) : undefined,
    }),
  );
  res.statusCode = response.status;
  response.headers.forEach((value, name) => res.setHeader(name, value));
  res.end(Buffer.from(await response.arrayBuffer()));
};
const jevFunction: Plugin = {
  name: "jev-function",
  configureServer: (server) => void server.middlewares.use("/.netlify/functions/jev", handle),
  configurePreviewServer: (server) => void server.middlewares.use("/.netlify/functions/jev", handle),
};

export default defineConfig({
  plugins: [jevFunction],
  // The MuJoCo loader finds mujoco.wasm next to itself; pre-bundling would break that in dev.
  optimizeDeps: { exclude: ["@mujoco/mujoco"] },
  server: { open: false },
  build: { target: "es2022", chunkSizeWarningLimit: 700 },
});
