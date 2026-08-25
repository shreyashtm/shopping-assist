import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeRecommendation, makeResponse } from "@/lib/test-fixtures";

import { ResultsView } from "./ResultsView";

describe("ResultsView", () => {
  it("renders each group's heading and its products", () => {
    render(
      <ResultsView
        response={makeResponse({
          groups: [
            {
              name: "Casual T-Shirts",
              why_needed: "For everyday wear.",
              items: [makeRecommendation()],
            },
          ],
        })}
      />,
    );
    expect(screen.getByText("Casual T-Shirts")).toBeInTheDocument();
    expect(screen.getByText("For everyday wear.")).toBeInTheDocument();
    expect(screen.getByText("Men's Tech 2.0 Short-Sleeve T-Shirt")).toBeInTheDocument();
  });

  it("shows the empty-catalogue message when there are no groups", () => {
    render(<ResultsView response={makeResponse({ groups: [] })} />);
    expect(screen.getByText("Nothing in the catalogue fits that closely.")).toBeInTheDocument();
  });

  it("shows required unfilled slots with the incomplete-answer notice", () => {
    render(
      <ResultsView
        response={makeResponse({
          groups: [],
          unfilled_slots: [
            { name: "Suit Jacket", role: "required", reason: "No formalwear in stock." },
          ],
        })}
      />,
    );
    expect(screen.getByText("Could not cover")).toBeInTheDocument();
    expect(screen.getByText("Suit Jacket")).toBeInTheDocument();
    expect(
      screen.getByText("This is not a complete answer — required items are missing from the catalogue."),
    ).toBeInTheDocument();
  });

  it("does not show the incomplete-answer notice for an optional unfilled slot", () => {
    render(
      <ResultsView
        response={makeResponse({
          groups: [],
          unfilled_slots: [{ name: "Watch", role: "optional", reason: "Nothing close enough." }],
        })}
      />,
    );
    expect(
      screen.queryByText("This is not a complete answer — required items are missing from the catalogue."),
    ).not.toBeInTheDocument();
    // Still surfaced, just without the "incomplete" framing.
    expect(screen.getByText("Watch")).toBeInTheDocument();
  });

  it("shows assumptions when present", () => {
    render(
      <ResultsView response={makeResponse({ assumptions: ["Assumed you want a single item."] })} />,
    );
    expect(screen.getByText("Assumed you want a single item.")).toBeInTheDocument();
  });

  it("shows a degraded-mode notice when meta.degraded_mode is true", () => {
    render(
      <ResultsView
        response={makeResponse({
          groups: [{ name: "Group", why_needed: "x", items: [makeRecommendation()] }],
          meta: {
            latency_ms: 500, llm_calls: 0, cached: false, degraded_mode: true,
            catalogue_size: 1738, notes: ["No LLM configured; used keyword interpretation."],
          },
        })}
      />,
    );
    expect(screen.getByText("Reduced mode.")).toBeInTheDocument();
    expect(screen.getByText(/No LLM configured/)).toBeInTheDocument();
  });

  it("does not show the degraded-mode notice for a full-quality response", () => {
    render(<ResultsView response={makeResponse({ meta: { latency_ms: 500, llm_calls: 1, cached: false, degraded_mode: false, catalogue_size: 1738, notes: [] } })} />);
    expect(screen.queryByText("Reduced mode.")).not.toBeInTheDocument();
  });

  it("renders the product count and catalogue size in the footer", () => {
    render(
      <ResultsView
        response={makeResponse({
          groups: [{ name: "Group", why_needed: "x", items: [makeRecommendation(), makeRecommendation({ product: { ...makeRecommendation().product, id: "p2" } })] }],
        })}
      />,
    );
    expect(screen.getByText(/2 products from 1738 in the catalogue/)).toBeInTheDocument();
  });
});
