import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [
    react({
      babel: {
        plugins: [["babel-plugin-react-compiler"]],
      },
    }),
  ],
  server: {
    host: true,
    strictPort: true,
    port: 5173,
    watch: {
      usePolling: true,
    },
    proxy: {
      "/api": {
        target: "http://backend:8000",
        changeOrigin: true,
        secure: false,
        // REMOVED: rewrite: (path) => path.replace(/^\/api/, ""),
        // Now '/api/detect' stays '/api/detect'
      },
    },
  },
});