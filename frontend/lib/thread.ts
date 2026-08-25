/**
 * Conversation state.
 *
 * The backend is stateless and single-shot: every POST carries the whole
 * request, and there is no conversation id. The thread therefore lives entirely
 * in the client, and a follow-up is turned back into one self-contained query
 * before it is sent -- see `composeFollowUp`.
 */

import type {
  ContextVariable,
  RecommendRequest,
  RecommendResponse,
  Stage,
} from "./types";

export interface UserTurn {
  kind: "user";
  id: string;
  text: string;
  /**
   * `chip` marks an echo of a tapped clarifying answer rather than something
   * the shopper wrote. Only `typed` turns are replayed as prior intent when a
   * follow-up is composed -- a chip's information already reaches the model
   * through `answers`, and repeating it as prose would state it twice.
   */
  origin: "typed" | "chip";
}

export interface AssistantTurn {
  kind: "assistant";
  id: string;
  status: "loading" | "done" | "error";
  stage: Stage | null;
  response: RecommendResponse | null;
  error: string | null;
  /** What produced this turn, so retry and clarify replies can re-post it. */
  request: RecommendRequest;
}

export type Turn = UserTurn | AssistantTurn;

/** Mirrors `RecommendRequest.query`'s max_length in the Pydantic schema. */
export const MAX_QUERY_CHARS = 1000;

/**
 * The `known` planning variables, rendered for a human to read.
 *
 * Carried into a follow-up as prose rather than as `answers` chips on purpose.
 * Chip values are applied as hard filters by `merge_answers` *after* the model
 * has interpreted the text, so a chip saying `price_max:1500` would silently
 * overrule a follow-up that says "actually up to 5,000". As prose the two land
 * in the same interpretation pass and the later instruction wins, which is what
 * the shopper meant.
 */
export function establishedContext(variables: ContextVariable[]): string {
  return variables
    .filter((v) => v.status === "known" && v.value)
    .map((v) => `${v.label}: ${v.value}`)
    .join(" · ");
}

/**
 * Fold a follow-up message back into one standalone query.
 *
 * Budgeted to the schema's length cap, shedding the least important part first:
 * the established context goes before the earlier request text does, since the
 * model can re-derive context from the request but not the other way round.
 */
export function composeFollowUp(
  earlier: string,
  established: string,
  followUp: string,
): string {
  if (followUp.length >= MAX_QUERY_CHARS) return followUp.slice(0, MAX_QUERY_CHARS);

  const build = (earlierText: string, context: string) =>
    [
      `Earlier request: ${earlierText}`,
      context ? `Already established: ${context}` : "",
      `Follow-up: ${followUp}`,
    ]
      .filter(Boolean)
      .join("\n");

  const full = build(earlier, established);
  if (full.length <= MAX_QUERY_CHARS) return full;

  const withoutContext = build(earlier, "");
  if (withoutContext.length <= MAX_QUERY_CHARS) return withoutContext;

  const room = MAX_QUERY_CHARS - build("", "").length;
  return build(earlier.slice(0, Math.max(0, room)), "");
}

/** Labels for the chips a shopper tapped, for echoing back into the thread. */
export function describeAnswers(
  response: RecommendResponse,
  values: string[],
): string {
  const labels = response.questions
    .flatMap((q) => q.options)
    .filter((o) => values.includes(o.value))
    .map((o) => o.label);
  return labels.length > 0 ? labels.join(" · ") : "Answered";
}

export function newId(): string {
  return Math.random().toString(36).slice(2, 10);
}
