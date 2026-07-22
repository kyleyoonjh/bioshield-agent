/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bio: {
          50: "#f0fdf4",
          100: "#dcfce7",
          500: "#22c55e",
          600: "#16a34a",
          700: "#15803d",
          900: "#14532d",
        },
        clinical: {
          50:  "#f0f4ff",
          100: "#dce8ff",
          200: "#b8d0ff",
          300: "#85aff5",
          400: "#5185e8",
          500: "#2d63d4",
          600: "#1d4db8",
          700: "#163b95",
          800: "#1e3a5f",
          900: "#152a47",
          950: "#0d1b2e",
        },
      },
    },
  },
  plugins: [],
};
