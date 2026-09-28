"use client";

import type { ClarifyingQuestion } from "@/lib/types";

interface Props {
  questions: ClarifyingQuestion[];
  /** Slot -> tapped value. Controlled from the parent, not local state --
   * `page.tsx` needs to read the current answers the instant the real
   * response arrives, which a self-contained `useState` here couldn't offer. */
  answers: Record<string, string>;
  onAnswer: (slot: string, value: string) => void;
}

/**
 * Best-effort questions shown while the one real LLM call is still running.
 *
 * These are a guess from the request text alone -- see
 * `context_slots.guess_preview_questions` -- not the real interpretation, so
 * the framing has to say that plainly rather than look like the final
 * clarify step arrived early. Tapping one costs nothing extra: if it turns
 * out to match a real question once interpretation returns, `page.tsx`
 * applies it automatically; if not, it is dropped without ceremony.
 */
export function ProvisionalQuestions({ questions, answers, onAnswer }: Props) {
  if (questions.length === 0) return null;

  return (
    <div className="animate-rise space-y-4 border-l-2 border-border pl-4 sm:pl-5">
      <p className="eyebrow text-muted">While I look into this</p>
      <div className="space-y-4">
        {questions.map((q) => (
          <div key={q.slot} className="space-y-2">
            <p className="text-sm text-muted">{q.question}</p>
            <div className="flex flex-wrap gap-2">
              {q.options.map((option) => {
                const selected = answers[q.slot] === option.value;
                return (
                  <button
                    key={option.value}
                    onClick={() => onAnswer(q.slot, option.value)}
                    className={`border px-3.5 py-2 text-xs transition-colors ${
                      selected
                        ? "border-accent bg-accent text-accent-contrast"
                        : "border-border bg-surface text-muted hover:border-foreground hover:text-foreground"
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
    </div>
  );
}
