import { describe, expect, it } from "vitest";

import {
  MAX_QUERY_CHARS,
  composeFollowUp,
  describeAnswers,
  establishedContext,
  matchedProvisionalAnswers,
  newId,
} from "./thread";
import type { ClarifyingQuestion, ContextVariable, RecommendResponse } from "./types";

function makeVariable(overrides: Partial<ContextVariable>): ContextVariable {
  return {
    name: "budget",
    label: "Budget",
    status: "known",
    source: "user",
    value: "₹1,500",
    ...overrides,
  };
}

describe("establishedContext", () => {
  it("joins known variables with a value", () => {
    const result = establishedContext([
      makeVariable({ label: "Budget", value: "₹1,500" }),
      makeVariable({ name: "gender", label: "For", value: "men" }),
    ]);
    expect(result).toBe("Budget: ₹1,500 · For: men");
  });

  it("excludes variables that are only 'needed', not known", () => {
    const result = establishedContext([
      makeVariable({ status: "needed", value: null }),
    ]);
    expect(result).toBe("");
  });

  it("excludes a known variable with no value", () => {
    const result = establishedContext([makeVariable({ value: null })]);
    expect(result).toBe("");
  });

  it("returns an empty string for no variables", () => {
    expect(establishedContext([])).toBe("");
  });
});

describe("composeFollowUp", () => {
  it("builds earlier + established + follow-up in order", () => {
    const result = composeFollowUp("t-shirts", "Budget: ₹1,500", "in blue please");
    expect(result).toBe(
      "Earlier request: t-shirts\nAlready established: Budget: ₹1,500\nFollow-up: in blue please",
    );
  });

  it("omits the established line entirely when there is none", () => {
    const result = composeFollowUp("t-shirts", "", "in blue please");
    expect(result).not.toContain("Already established");
    expect(result).toBe("Earlier request: t-shirts\nFollow-up: in blue please");
  });

  it("returns the raw follow-up unchanged when it alone is already at the cap", () => {
    const longFollowUp = "x".repeat(MAX_QUERY_CHARS);
    const result = composeFollowUp("earlier", "context", longFollowUp);
    expect(result).toBe(longFollowUp.slice(0, MAX_QUERY_CHARS));
  });

  it("truncates an over-long follow-up to the cap", () => {
    const longFollowUp = "x".repeat(MAX_QUERY_CHARS + 50);
    const result = composeFollowUp("earlier", "context", longFollowUp);
    expect(result.length).toBe(MAX_QUERY_CHARS);
  });

  it("sheds established context first when the full composition is too long", () => {
    const longEarlier = "a".repeat(400);
    const established = "b".repeat(400);
    const followUp = "c".repeat(400);
    const result = composeFollowUp(longEarlier, established, followUp);
    expect(result.length).toBeLessThanOrEqual(MAX_QUERY_CHARS);
    expect(result).not.toContain("Already established");
    expect(result).toContain(`Follow-up: ${followUp}`);
  });

  it("truncates the earlier request as a last resort, keeping the follow-up intact", () => {
    const longEarlier = "a".repeat(2000);
    const established = "";
    const followUp = "c".repeat(400);
    const result = composeFollowUp(longEarlier, established, followUp);
    expect(result.length).toBeLessThanOrEqual(MAX_QUERY_CHARS);
    expect(result).toContain(`Follow-up: ${followUp}`);
  });

  it("never produces a result longer than the schema's max_length", () => {
    const longEarlier = "a".repeat(5000);
    const established = "b".repeat(5000);
    const followUp = "c".repeat(999);
    const result = composeFollowUp(longEarlier, established, followUp);
    expect(result.length).toBeLessThanOrEqual(MAX_QUERY_CHARS);
  });
});

describe("describeAnswers", () => {
  const response: RecommendResponse = {
    query_id: "q1",
    mode: "clarify",
    intent_summary: "Looking for a gift",
    context: { location: null, start_date: null, end_date: null, duration_days: null, climate_note: null, recipient: null, climate: null },
    assumptions: [],
    questions: [
      {
        slot: "gender",
        question: "Who is this for?",
        options: [
          { label: "Men", value: "gender:men" },
          { label: "Women", value: "gender:women" },
        ],
        allow_multiple: false,
      },
      {
        slot: "budget",
        question: "What's your budget?",
        options: [{ label: "Under ₹500", value: "price_max:500" }],
        allow_multiple: false,
      },
    ],
    meta: { latency_ms: 100, llm_calls: 1, cached: false, degraded_mode: false, catalogue_size: 1738, notes: [] },
  };

  it("maps tapped values back to their chip labels", () => {
    expect(describeAnswers(response, ["gender:men"])).toBe("Men");
  });

  it("joins labels from multiple answered questions", () => {
    expect(describeAnswers(response, ["gender:men", "price_max:500"])).toBe(
      "Men · Under ₹500",
    );
  });

  it("falls back to 'Answered' when no tapped value matches any option", () => {
    expect(describeAnswers(response, ["occasion:office"])).toBe("Answered");
  });

  it("falls back to 'Answered' for an empty values list", () => {
    expect(describeAnswers(response, [])).toBe("Answered");
  });
});

describe("matchedProvisionalAnswers", () => {
  const realQuestions: ClarifyingQuestion[] = [
    {
      slot: "gender",
      question: "Who is this for?",
      options: [{ label: "Men", value: "gender:men" }],
      allow_multiple: false,
    },
  ];

  it("keeps a provisional answer whose slot matches a real question", () => {
    const result = matchedProvisionalAnswers({ gender: "gender:men" }, realQuestions);
    expect(result).toEqual(["gender:men"]);
  });

  it("drops a provisional answer whose slot the real interpretation never asked about", () => {
    // The guess was wrong for this request -- the real interpretation needed
    // something else entirely, so applying the guess would be silently
    // forcing an unrelated tap onto the actual result.
    const result = matchedProvisionalAnswers(
      { dates: "start_date:2026-10-01,duration_days:7" },
      realQuestions,
    );
    expect(result).toEqual([]);
  });

  it("keeps only the matching subset when some slots match and some do not", () => {
    const result = matchedProvisionalAnswers(
      { gender: "gender:men", dates: "timing:unknown" },
      realQuestions,
    );
    expect(result).toEqual(["gender:men"]);
  });

  it("returns an empty list when nothing was tapped", () => {
    expect(matchedProvisionalAnswers({}, realQuestions)).toEqual([]);
  });

  it("returns an empty list when the real interpretation asked no questions at all", () => {
    expect(matchedProvisionalAnswers({ gender: "gender:men" }, [])).toEqual([]);
  });
});

describe("newId", () => {
  it("returns a non-empty string", () => {
    expect(newId().length).toBeGreaterThan(0);
  });

  it("returns different values across calls", () => {
    const ids = new Set(Array.from({ length: 20 }, () => newId()));
    expect(ids.size).toBeGreaterThan(1);
  });
});
