"use client";

import { useEffect, useRef, useState } from "react";

import { AssistantTurnView } from "@/components/AssistantTurn";
import { QueryBar } from "@/components/QueryBar";
import { ThemeToggle } from "@/components/ThemeToggle";
import { UserMessage } from "@/components/UserMessage";
import { previewQuestions, recommendStreaming } from "@/lib/api";
import type { AssistantTurn, Turn } from "@/lib/thread";
import {
  composeFollowUp,
  describeAnswers,
  establishedContext,
  matchedProvisionalAnswers,
  newId,
} from "@/lib/thread";
import type { RecommendRequest } from "@/lib/types";

export default function Home() {
  const [turns, setTurns] = useState<Turn[]>([]);

  // A new search must cancel the previous one, or a slow first response can
  // land after a fast second one and overwrite it.
  const inFlight = useRef<AbortController | null>(null);
  const newestTurnRef = useRef<HTMLDivElement | null>(null);

  // Mirrors AssistantTurn.provisionalAnswers, keyed by turn id, so the
  // completion handler in `run()` can read the *current* tapped answers the
  // instant the real response arrives -- reading through React state there
  // would see whatever `provisionalAnswers` closed over at call time, not
  // whatever the shopper tapped in the meantime. State still holds the same
  // data for rendering; this ref exists only for that one synchronous read.
  const provisionalAnswersRef = useRef<Map<string, Record<string, string>>>(new Map());

  const busy = turns.some((t) => t.kind === "assistant" && t.status === "loading");
  const started = turns.length > 0;

  // Scrolls the newest turn to the *top* of the viewport rather than scrolling
  // to the bottom of the page. A results turn is several screens tall, so
  // bottoming out lands the reader on the last product card of a group they
  // have not seen the heading of; starting at the top follows the reply.
  useEffect(() => {
    newestTurnRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [turns.length]);

  /**
   * Append an assistant turn and drive it to completion from the stream.
   *
   * `enablePreview` fires `/recommend/preview` concurrently with the real
   * request -- best-effort guessed questions shown while the one real LLM
   * call is in flight. Only set for a shopper's own typed message: `answer`
   * and `skip` also call `run`, but by then the shopper is already past
   * clarification, so guessing fresh questions for that turn would be noise,
   * not help.
   */
  async function run(request: RecommendRequest, options: { enablePreview?: boolean } = {}) {
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;

    const id = newId();
    provisionalAnswersRef.current.set(id, {});
    setTurns((prev) => [
      ...prev,
      {
        kind: "assistant",
        id,
        status: "loading",
        stage: null,
        response: null,
        error: null,
        request,
        provisionalQuestions: null,
        provisionalAnswers: {},
      },
    ]);

    const patch = (update: Partial<AssistantTurn>) =>
      setTurns((prev) =>
        prev.map((t) => (t.kind === "assistant" && t.id === id ? { ...t, ...update } : t)),
      );

    if (options.enablePreview) {
      void previewQuestions(request.query, controller.signal).then((questions) => {
        patch({ provisionalQuestions: questions });
      });
    }

    await settle(request, id, patch, controller);
  }

  /**
   * Drives one turn to its final state, re-running in place -- silently,
   * within the same turn -- whenever a provisional answer turns out to match
   * a real question.
   *
   * Previously a match closed the loading turn with `status: "done"` and
   * opened a *second* one for the follow-up: an echoed chip, then a fresh
   * loading state, then the real result. That read as two rounds of asking
   * for something the shopper had already answered while waiting. Recursing
   * here instead means the intermediate clarify response -- the one the tap
   * already resolved -- is never shown at all. The turn stays in `loading`
   * (stage updates keep animating through the second call too, so the wait
   * still reads as progress) until there is nothing left to auto-resolve,
   * and only *that* final state ever becomes visible. Bounded by
   * construction: provisional answers exist only on the first call and are
   * consumed before recursing, so a second match is never possible.
   */
  async function settle(
    request: RecommendRequest,
    id: string,
    patch: (update: Partial<AssistantTurn>) => void,
    controller: AbortController,
  ) {
    try {
      const result = await recommendStreaming(
        request,
        (stage) => patch({ stage }),
        controller.signal,
      );

      const tapped = provisionalAnswersRef.current.get(id) ?? {};
      const matched = matchedProvisionalAnswers(tapped, result.questions);

      if (result.mode === "clarify" && matched.length > 0) {
        provisionalAnswersRef.current.set(id, {});
        await settle(
          { query: request.query, answers: [...(request.answers ?? []), ...matched] },
          id,
          patch,
          controller,
        );
        return;
      }

      provisionalAnswersRef.current.delete(id);
      patch({
        status: "done",
        response: result,
        stage: null,
        provisionalQuestions: null,
      });
    } catch (err) {
      provisionalAnswersRef.current.delete(id);
      if (err instanceof DOMException && err.name === "AbortError") {
        // Superseded by a newer message, which has already appended its own
        // turn. Drop this one rather than leaving a spinner that never ends.
        setTurns((prev) => prev.filter((t) => t.id !== id));
        return;
      }
      patch({
        status: "error",
        error: err instanceof Error ? err.message : "Something went wrong.",
      });
    }
  }

  /** Tap a provisional (still-loading) question's option, or tap it again to clear it. */
  function answerProvisional(turnId: string, slot: string, value: string) {
    const current = provisionalAnswersRef.current.get(turnId) ?? {};
    const next =
      current[slot] === value
        ? Object.fromEntries(Object.entries(current).filter(([key]) => key !== slot))
        : { ...current, [slot]: value };
    provisionalAnswersRef.current.set(turnId, next);
    setTurns((prev) =>
      prev.map((t) =>
        t.kind === "assistant" && t.id === turnId ? { ...t, provisionalAnswers: next } : t,
      ),
    );
  }

  /** The newest completed response, used to carry context into a follow-up. */
  function lastResponse() {
    for (let i = turns.length - 1; i >= 0; i--) {
      const turn = turns[i];
      if (turn.kind === "assistant" && turn.status === "done" && turn.response) {
        return turn.response;
      }
    }
    return null;
  }

  function ask(text: string) {
    // The backend holds no conversation state, so a follow-up is folded back
    // into a single self-contained query before it is sent.
    const earlier = turns
      .filter((t): t is Turn & { kind: "user" } => t.kind === "user" && t.origin === "typed")
      .map((t) => t.text)
      .join(" ; then: ");

    const previous = lastResponse();
    const query = earlier
      ? composeFollowUp(
          earlier,
          previous ? establishedContext(previous.context_variables ?? []) : "",
          text,
        )
      : text;

    setTurns((prev) => [...prev, { kind: "user", id: newId(), text, origin: "typed" }]);
    void run({ query }, { enablePreview: true });
  }

  function answer(turn: AssistantTurn, answers: string[]) {
    const label = turn.response ? describeAnswers(turn.response, answers) : "Answered";
    setTurns((prev) => [...prev, { kind: "user", id: newId(), text: label, origin: "chip" }]);
    void run({
      query: turn.request.query,
      answers: [...(turn.request.answers ?? []), ...answers],
    });
  }

  function skip(turn: AssistantTurn) {
    setTurns((prev) => [
      ...prev,
      { kind: "user", id: newId(), text: "Just show me what you have", origin: "chip" },
    ]);
    void run({
      query: turn.request.query,
      answers: turn.request.answers,
      skip_clarification: true,
    });
  }

  function reset() {
    inFlight.current?.abort();
    setTurns([]);
  }

  const lastAssistantId = [...turns]
    .reverse()
    .find((t): t is AssistantTurn => t.kind === "assistant")?.id;

  return (
    <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col px-5 sm:px-8">
      <header
        className={`flex items-start justify-between gap-4 ${started ? "py-5" : "pt-10 pb-0 sm:pt-16"}`}
      >
        <div className={started ? "" : "max-w-2xl"}>
          <p className="eyebrow text-accent">Shopping Assistant</p>
          {!started && (
            <>
              <h1 className="display mt-3 text-[2rem] sm:text-[3.25rem]">
                Tell me what you&rsquo;re shopping for.
              </h1>
              <p className="editorial mt-4 max-w-xl text-[15px] leading-relaxed text-muted">
                Describe the trip, the occasion, or the person — in plain English. I&rsquo;ll
                ask only what I actually need, then shortlist real products with the
                reasoning shown.
              </p>
            </>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {started && (
            <button
              onClick={reset}
              className="border border-border px-3 py-1.5 text-xs text-muted transition-colors hover:border-foreground hover:text-foreground"
            >
              New chat
            </button>
          )}
          <ThemeToggle />
        </div>
      </header>

      {started && (
        <div className="thread-scroll flex-1 space-y-8 py-4">
          {turns.map((turn, index) => (
            <div
              key={turn.id}
              ref={index === turns.length - 1 ? newestTurnRef : null}
              className="scroll-mt-4"
            >
              {turn.kind === "user" ? (
                <UserMessage text={turn.text} />
              ) : (
                <AssistantTurnView
                  turn={turn}
                  isLast={turn.id === lastAssistantId}
                  onAnswer={answer}
                  onSkip={skip}
                  onRetry={(t) => void run(t.request)}
                  onAnswerProvisional={answerProvisional}
                />
              )}
            </div>
          ))}
        </div>
      )}

      <div
        className={`sticky bottom-0 z-10 bg-background pb-5 ${started ? "pt-4" : "pt-8"}`}
      >
        <QueryBar
          onSubmit={ask}
          busy={busy}
          showExamples={!started}
          placeholder={
            started
              ? "Ask a follow-up, or refine what you're looking for…"
              : "Describe what you're shopping for, the way you'd tell a friend…"
          }
        />
        {!started && (
          <p className="mt-6 border-t border-border pt-4 text-xs text-muted">
            Products are real listings from Amazon.in and Myntra, verified live at
            catalogue build time. Prices and stock may have changed since.
          </p>
        )}
      </div>
    </main>
  );
}
