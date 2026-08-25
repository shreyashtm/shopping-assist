"use client";

import type { Stage } from "@/lib/types";

/*
 * Progress reported from the server, not guessed on a timer.
 *
 * A search spends most of its time in one LLM call plus a weather lookup.
 * These labels advance only when the backend says the stage actually changed.
 */
const STAGES: { key: Stage; label: string; detail: string }[] = [
  {
    key: "interpreting",
    label: "Reading your request",
    detail: "Working out what the occasion, timing and conditions actually call for.",
  },
  {
    key: "checking conditions",
    label: "Checking conditions",
    detail: "Looking up weather and elevation for the place and dates you gave.",
  },
  {
    key: "searching",
    label: "Searching the catalogue",
    detail: "Matching against the catalogue, once per need.",
  },
];

export function ThinkingState({ stage }: { stage: Stage | null }) {
  if (stage === "cached") {
    return <p className="animate-rise text-sm text-muted">Recalling an earlier answer…</p>;
  }

  const activeIndex = STAGES.findIndex((s) => s.key === stage);

  return (
    <div className="animate-rise space-y-8">
      <ol className="space-y-4">
        {STAGES.map((s, index) => {
          const done = activeIndex > index;
          const active = activeIndex === index;
          return (
            <li key={s.key} className="flex gap-4">
              <span
                className={`eyebrow mt-0.5 w-6 shrink-0 ${
                  active ? "text-accent" : done ? "text-foreground" : "text-border"
                }`}
              >
                {done ? "✓" : String(index + 1).padStart(2, "0")}
              </span>
              <div className="min-w-0 flex-1">
                <p
                  className={`text-sm ${
                    active ? "font-medium text-foreground" : done ? "text-muted" : "text-border"
                  }`}
                >
                  {s.label}
                </p>
                {active && (
                  <>
                    <p className="editorial mt-1 text-[13px] leading-relaxed text-muted">
                      {s.detail}
                    </p>
                    <div className="mt-2 h-px w-full bg-border">
                      <div className="animate-sweep h-px bg-accent" />
                    </div>
                  </>
                )}
              </div>
            </li>
          );
        })}
      </ol>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="border border-border bg-surface">
            <div className="aspect-square animate-pulse bg-surface-muted" />
            <div className="space-y-2 p-4">
              <div className="h-3 w-2/3 animate-pulse bg-surface-muted" />
              <div className="h-3 w-1/3 animate-pulse bg-surface-muted" />
              <div className="h-10 animate-pulse bg-surface-muted" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
