import type { Config } from "tailwindcss";

export default {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
      },
    },
  },
  plugins: [],

  include: ["./src/**/*"],
  compilerOptions: {
    strict: true,
    esModuleInterop: true,
    lib: ["dom", "es2015"],
    jsx: "react",
  },
} satisfies Config;
