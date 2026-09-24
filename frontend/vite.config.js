import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/AI_assistant_Chatbot/",
  server: {
    // Vite's default 5173 falls inside a Windows-reserved port block
    // (5148-5247) on some machines, which fails the bind with EACCES.
    // Pin a port outside the ephemeral range and fail loudly if taken.
    port: 3000,
    strictPort: true,
  },
});
