import { useEffect, useState } from "react";

export interface ToastMessage {
  id: number;
  message: string;
  type: "info" | "warning" | "success";
}

let _nextId = 0;
export function makeToast(message: string, type: ToastMessage["type"] = "info"): ToastMessage {
  return { id: _nextId++, message, type };
}

export function ToastContainer({ toasts, onDismiss }: {
  toasts: ToastMessage[];
  onDismiss: (id: number) => void;
}) {
  return (
    <div className="fixed bottom-5 right-5 z-50 flex flex-col gap-2 pointer-events-none">
      {toasts.map(t => (
        <ToastItem key={t.id} toast={t} onDismiss={onDismiss} />
      ))}
    </div>
  );
}

function ToastItem({ toast, onDismiss }: { toast: ToastMessage; onDismiss: (id: number) => void }) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    // slide in
    requestAnimationFrame(() => setVisible(true));
    // auto-dismiss after 4 s
    const t = setTimeout(() => {
      setVisible(false);
      setTimeout(() => onDismiss(toast.id), 300);
    }, 4000);
    return () => clearTimeout(t);
  }, [toast.id, onDismiss]);

  const styleMap = {
    warning: "bg-amber-50 border-amber-300 text-amber-800",
    success: "bg-emerald-50 border-emerald-300 text-emerald-800",
    info:    "bg-blue-50 border-blue-200 text-blue-800",
  };
  const iconMap = { warning: "⚠", success: "✅", info: "ℹ" };

  return (
    <div
      className={`pointer-events-auto flex items-start gap-3 px-4 py-3 rounded-xl shadow-lg border text-sm max-w-sm transition-all duration-300 ${
        visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2"
      } ${styleMap[toast.type]}`}
    >
      <span className="text-base leading-none mt-0.5">{iconMap[toast.type]}</span>
      <span className="flex-1 leading-snug">{toast.message}</span>
      <button
        onClick={() => onDismiss(toast.id)}
        className="text-xs opacity-50 hover:opacity-100 leading-none mt-0.5"
      >✕</button>
    </div>
  );
}
