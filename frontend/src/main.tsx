import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// Vercel's dashboard doesn't expose client-side console output, so these
// two listeners are the only way to catch errors that happen OUTSIDE
// React's render cycle (a rejected promise nobody awaited/caught, a raw
// runtime error in an event handler) — React render errors are already
// caught by App.tsx's ErrorBoundary, which is a separate mechanism that
// can't see these.
window.addEventListener("error", event => {
  console.error("[window.onerror]", event.message, event.error);
});
window.addEventListener("unhandledrejection", event => {
  console.error("[unhandledrejection]", event.reason);
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <App />
);
