import { useCallback, useRef } from "react";
import type { CollectPayload, CollectResponse } from "../types";

export function useAutoCollect() {
  const lastIdRef = useRef<string | null>(null);

  const collect = useCallback(
    async (payload: CollectPayload): Promise<string | null> => {
      try {
        const res = await fetch("/api/v2/collect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) return null;
        const data: CollectResponse = await res.json();
        lastIdRef.current = data.id;
        return data.id;
      } catch {
        console.warn("[autoCollect] background collect failed — continuing");
        return null;
      }
    },
    [],
  );

  return { collect, lastIdRef };
}
