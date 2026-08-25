"use client";

import { ClarifyPanel } from "@/components/ClarifyPanel";
import { ResultsView } from "@/components/ResultsView";
import { ThinkingState } from "@/components/ThinkingState";
import type { AssistantTurn as Turn } from "@/lib/thread";

interface Props {
  turn: Turn;
  /** Only the newest turn stays interactive -- see `AskedSummary`. */
  isLast: boolean;
  onAnswer: (turn: Turn, answers: string[]) => void;
  onSkip: (turn: Turn) => void;
  onRetry: (turn: Turn) => void;
}

/**
 * A clarify turn that has already been answered.
 *
 * Rendered read-only rather than as live chips: re-answering a question from
 * halfway up the thread would silently re-run an old search and overwrite
 * newer results, which reads as the app losing its place.
 */
function AskedSummary({ turn }: { turn: Turn }) {
  const questions = turn.response?.questions ?? [];
  return (
    <div className="border-l-2 border-border pl-4">
      <p className="eyebrow text-muted">Asked</p>
      <ul className="mt-1.5 space-y-1">
        {questions.map((q) => (
          <li key={q.slot} className="text-sm text-muted">
            {q.question}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function AssistantTurnView({ turn, isLast, onAnswer, onSkip, onRetry }: Props) {
  if (turn.status === "loading") {
    return (
      <div className="border-l-2 border-accent pl-4 sm:pl-5">
        <ThinkingState stage={turn.stage} />
      </div>
    );
  }

  if (turn.status === "error") {
    return (
      <div className="animate-rise border border-maroon/40 bg-surface px-5 py-5">
        <p className="text-sm font-medium">I couldn&rsquo;t finish that search.</p>
        <p className="mt-1.5 text-sm text-muted">{turn.error}</p>
        <button
          onClick={() => onRetry(turn)}
          className="mt-4 border border-foreground px-5 py-2 text-xs font-medium transition-colors hover:border-accent hover:text-accent"
        >
          Try again
        </button>
      </div>
    );
  }

  const response = turn.response;
  if (!response) return null;

  if (response.mode === "clarify") {
    if (!isLast) return <AskedSummary turn={turn} />;
    const hasGroups = (response.groups ?? []).length > 0;
    return (
      <div className="space-y-8">
        {hasGroups && (
          <div className="border-l-2 border-accent pl-4 sm:pl-5">
            <ResultsView response={response} />
          </div>
        )}
        <div className="border-l-2 border-accent pl-4 sm:pl-5">
          <ClarifyPanel
            intro={response.intent_summary}
            context={response.context}
            contextVariables={response.context_variables ?? []}
            questions={response.questions}
            busy={false}
            refining={hasGroups}
            onSubmit={(answers) => onAnswer(turn, answers)}
            onSkip={() => onSkip(turn)}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="border-l-2 border-accent pl-4 sm:pl-5">
      <ResultsView response={response} />
    </div>
  );
}
