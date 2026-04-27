import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During development the frontend runs at :5173 and FastAPI at :8000.
// We proxy /api so fetches in code can stay relative.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
