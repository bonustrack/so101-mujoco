import { defineConfig } from "vite";

export default defineConfig({
  // The MuJoCo loader finds mujoco.wasm next to itself; pre-bundling would break that in dev.
  optimizeDeps: { exclude: ["@mujoco/mujoco"] },
  server: { open: false },
  build: { target: "es2022", chunkSizeWarningLimit: 700 },
});
