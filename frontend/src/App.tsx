import { useState, useEffect, useCallback, Component } from "react";
import type { ErrorInfo, ReactNode } from "react";

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[ErrorBoundary]", error, info.componentStack);
  }
  render() {
    if (this.state.error) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-slate-50">
          <div className="text-center p-8 bg-white rounded-xl shadow border border-red-100 max-w-md">
            <p className="text-2xl mb-2">⚠️</p>
            <p className="font-semibold text-slate-700 mb-1">앱 오류가 발생했습니다</p>
            <p className="text-xs text-slate-400 mb-4">{(this.state.error as Error).message}</p>
            <button
              onClick={() => this.setState({ error: null })}
              className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700"
            >
              다시 시도
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

import DrugDiscoveryChatPanel from "./components/DrugDiscoveryChatPanel";
import DemoBanner from "./components/DemoBanner";
import { ToastContainer } from "./components/Toast";
import { checkDemoMode } from "./services/covidService";
import type { ToastMessage } from "./components/Toast";

export default function App() {
  const [demoMode, setDemoMode] = useState(false);
  const [toasts, setToasts]     = useState<ToastMessage[]>([]);

  useEffect(() => {
    checkDemoMode().then(on => setDemoMode(on));
  }, []);

  const dismissToast = useCallback((id: number) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  return (
    <ErrorBoundary>
      <div className="h-screen flex flex-col bg-slate-50 overflow-hidden">

        {/* Header */}
        <header className="flex-shrink-0 bg-gradient-to-r from-blue-900 to-blue-700 shadow-lg z-20">
          <div className="px-5 py-3 flex items-center gap-4">
            <div>
              <h1 className="text-base font-bold tracking-wide text-white leading-none">
                AiRemedy-Agent
              </h1>
              <p className="text-[11px] text-blue-200 mt-0.5">
                MCP-native Autonomous Scientific Design Agent · Bio Computation Engine
              </p>
            </div>

            <div className="ml-auto flex items-center gap-4">
              <span className="text-[11px] font-semibold px-3 py-1 rounded-md bg-blue-600 text-white">
                🧪 신약개발
              </span>

              {/* Status indicator */}
              <span className="flex items-center gap-1.5 text-[11px] text-blue-200">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                Ready
              </span>
            </div>
          </div>
        </header>

        {/* Main */}
        <main className="flex-1 flex overflow-hidden">
          <div className="flex-1 overflow-hidden">
            <DrugDiscoveryChatPanel />
          </div>
        </main>

        <ToastContainer toasts={toasts} onDismiss={dismissToast} />
        {demoMode && <DemoBanner />}
      </div>
    </ErrorBoundary>
  );
}
