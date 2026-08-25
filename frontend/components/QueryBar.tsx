"use client";

import { useState } from "react";

const EXAMPLES = [
  "I'm trekking Hampta Pass the last week of October for a week — find me essentials and clothing.",
  "Find me good traditional wear for my friend's wedding in March next year.",
  "I need a premium gifting hamper for my parents' 25th anniversary next month.",
];

interface Props {
  onSubmit: (query: string) => void;
  busy: boolean;
  /** Shown only on an empty thread; once a conversation exists they are noise. */
  showExamples?: boolean;
  placeholder?: string;
}

export function QueryBar({
  onSubmit,
  busy,
  showExamples = false,
  placeholder = "Describe what you're shopping for, the way you'd tell a friend…",
}: Props) {
  const [value, setValue] = useState("");

  function submit() {
    const trimmed = value.trim();
    if (trimmed.length < 3 || busy) return;
    onSubmit(trimmed);
    // Cleared on send: the composer is reused for every turn, so leaving the
    // previous message in it makes the next one start as an edit of the last.
    setValue("");
  }

  return (
    <div className="w-full">
      <div className="border border-foreground bg-surface transition-colors focus-within:border-accent">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            // Enter sends; Shift+Enter is a newline. Matches chat conventions.
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          rows={showExamples ? 3 : 2}
          disabled={busy}
          placeholder={placeholder}
          className="w-full resize-none bg-transparent px-5 pt-4 pb-2 text-base outline-none placeholder:text-muted disabled:opacity-60"
        />
        <div className="flex items-center justify-between gap-3 px-5 pb-4">
          <span className="hidden text-xs text-muted sm:inline">
            Enter to send · Shift+Enter for a new line
          </span>
          <button
            onClick={submit}
            disabled={busy || value.trim().length < 3}
            className="ml-auto bg-accent px-6 py-2.5 text-sm font-medium text-accent-contrast transition-opacity hover:opacity-90 disabled:opacity-40"
          >
            {busy ? "Thinking…" : "Send"}
          </button>
        </div>
      </div>

      {showExamples && (
        <div className="mt-5">
          <p className="eyebrow text-muted">Try</p>
          <div className="mt-2 flex flex-col gap-1.5 sm:flex-row sm:flex-wrap">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                onClick={() => setValue(ex)}
                disabled={busy}
                className="border border-border bg-surface-muted px-3 py-2 text-left text-xs text-muted transition-colors hover:border-accent hover:text-foreground disabled:opacity-50 sm:max-w-[19rem]"
              >
                {ex.length > 58 ? `${ex.slice(0, 58)}…` : ex}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
