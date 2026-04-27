/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      // Restrained palette: near-black canvas, single warm accent, three nation-coded
      // colors that stay desaturated so they read as data, not decoration.
      colors: {
        bg: {
          DEFAULT: "#0c0d10",
          elev: "#14161b",
          panel: "#191b21",
        },
        line: "#2a2d35",
        ink: {
          DEFAULT: "#e6e7ea",
          dim: "#9ba0aa",
          mute: "#6b7280",
        },
        accent: {
          DEFAULT: "#d39459",
          dim: "#8a6034",
        },
        invader: "#c95757",
        defender: "#5b8eda",
        neutral: "#6c6f7a",
        contested: "#3a3d44",
        warn: "#d3a14b",
        ok: "#6ba36b",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui"],
        mono: ["JetBrains Mono", "ui-monospace", "Menlo"],
      },
      fontSize: {
        xxs: ["0.69rem", { lineHeight: "0.95rem" }],
      },
    },
  },
  plugins: [],
};
