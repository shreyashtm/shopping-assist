"use client";

import { useState } from "react";

import { ContextStrip } from "@/components/ContextStrip";
import { ContextVariablesPanel } from "@/components/ContextVariablesPanel";
import type { ClarifyingQuestion, ContextVariable, ResolvedContext } from "@/lib/types";

interface Props {
  intro: string;
  context: ResolvedContext;
  contextVariables: ContextVariable[];
  questions: ClarifyingQuestion[];
  onSubmit: (answers: string[]) => void;
  onSkip: () => void;
  busy: boolean;
  /** True when a recommendation is already showing above this panel, so the
   * copy and context blocks here should read as refinement, not as the only
   * thing this turn produced -- and shouldn't repeat what was already shown. */
  refining?: boolean;
}

/**
 * The "ask before recommending" step.
 *
 * Answers are tapped, not typed, which is what keeps this step free: the chip
 * values merge into the interpreted query on the server without a second LLM
 * call. Typing a paragraph would require re-interpretation, so chips are the
 * primary path and the skip button is the escape hatch.
 */
export function ClarifyPanel({
  intro,
  context,
  contextVariables,
  questions,
  onSubmit,
  onSkip,
  busy,
  refining = false,
}: Props) {
  const [picked, setPicked] = useState<Record<string, string>>({});
  const answeredCount = Object.keys(picked).length;

  return (
    <section className="animate-rise space-y-8">
      <div>
        {!refining && <h2 className="display text-2xl sm:text-[28px]">{intro}</h2>}
        <p className="editorial mt-2 text-sm text-muted">
          {refining
            ? "A couple of things would make this more precise:"
            : "A couple of quick things and I can narrow this down properly."}
        </p>
        {!refining && (
          <div className="mt-4 space-y-4">
            <ContextStrip context={context} />
            <ContextVariablesPanel variables={contextVariables} />
          </div>
        )}
      </div>

      <div className="space-y-7">
        {questions.map((q, index) => (
          <div key={q.slot} className="space-y-3">
            <div className="flex items-baseline gap-3 border-t border-border pt-4">
              <span className="eyebrow text-accent">
                {String(index + 1).padStart(2, "0")}
              </span>
              <h3 className="text-sm font-medium">{q.question}</h3>
            </div>
            <div className="flex flex-wrap gap-2">
              {q.options.map((option) => {
                const selected = picked[q.slot] === option.value;
                return (
                  <button
                    key={option.value}
                    disabled={busy}
                    onClick={() =>
                      setPicked((prev) =>
                        // Tapping the selected chip again clears it, so a
                        // mis-tap does not force a restart.
                        prev[q.slot] === option.value
                          ? Object.fromEntries(
                              Object.entries(prev).filter(([k]) => k !== q.slot),
                            )
                          : { ...prev, [q.slot]: option.value },
                      )
                    }
                    className={`border px-4 py-2.5 text-sm transition-colors disabled:opacity-50 ${
                      selected
                        ? "border-accent bg-accent text-accent-contrast"
                        : "border-border bg-surface hover:border-foreground"
                    }`}
                  >
                    {option.label}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-4 border-t border-border pt-5">
        <button
          onClick={() => onSubmit(Object.values(picked))}
          disabled={busy || answeredCount === 0}
          className="bg-accent px-6 py-2.5 text-sm font-medium text-accent-contrast transition-opacity hover:opacity-90 disabled:opacity-40"
        >
          {busy ? "Finding products…" : "Show me what fits"}
        </button>
        <button
          onClick={onSkip}
          disabled={busy}
          className="text-sm text-muted underline underline-offset-4 hover:text-foreground disabled:opacity-50"
        >
          Just show me now
        </button>
        {answeredCount > 0 && answeredCount < questions.length && (
          <span className="text-xs text-muted">
            {answeredCount} of {questions.length} answered — I&rsquo;ll assume the rest
          </span>
        )}
      </div>
    </section>
  );
}
