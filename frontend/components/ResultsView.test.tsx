import { fireEvent, render, screen } from "@testing-library/react";
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

  it("labels only the first item of a multi-item group as the best match", () => {
    const first = makeRecommendation();
    const second = makeRecommendation({ product: { ...first.product, id: "p2" }, match_score: 0.8 });
    render(
      <ResultsView
        response={makeResponse({
          groups: [
            { name: "Pair", why_needed: "x", items: [first, second] },
            { name: "Solo", why_needed: "y", items: [makeRecommendation({ product: { ...first.product, id: "p3" } })] },
          ],
        })}
      />,
    );
    expect(screen.getAllByText("Best match")).toHaveLength(1);
  });

  it("collapses groups past the third until the shopper opens them", () => {
    const base = makeRecommendation();
    const groups = Array.from({ length: 5 }, (_, g) => ({
      name: `Group ${g + 1}`,
      why_needed: `Reason ${g + 1}`,
      items: [0, 1].map((i) =>
        makeRecommendation({
          product: { ...base.product, id: `g${g}-${i}`, title: `Item ${g + 1}.${i + 1}` },
        }),
      ),
    }));
    render(<ResultsView response={makeResponse({ groups })} />);

    // Every group's heading and reason stays visible, so the kit reads whole.
    for (let g = 1; g <= 5; g++) {
      expect(screen.getByText(`Group ${g}`)).toBeInTheDocument();
      expect(screen.getByText(`Reason ${g}`)).toBeInTheDocument();
    }
    expect(screen.getByText("Item 3.1")).toBeInTheDocument();
    expect(screen.queryByText("Item 4.1")).not.toBeInTheDocument();
    expect(screen.queryByText("Item 5.1")).not.toBeInTheDocument();

    const buttons = screen.getAllByRole("button", { name: "Show 2 picks" });
    expect(buttons).toHaveLength(2);
    fireEvent.click(buttons[0]);

    expect(screen.getByText("Item 4.1")).toBeInTheDocument();
    expect(screen.queryByText("Item 5.1")).not.toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Show 2 picks" })).toHaveLength(1);
  });

  it("does not collapse anything for three groups or fewer", () => {
    const base = makeRecommendation();
    const groups = [1, 2, 3].map((g) => ({
      name: `G${g}`,
      why_needed: "x",
      items: [makeRecommendation({ product: { ...base.product, id: `p${g}` } })],
    }));
    render(<ResultsView response={makeResponse({ groups })} />);
    expect(screen.queryByRole("button", { name: /Show \d+ pick/ })).not.toBeInTheDocument();
  });
});
