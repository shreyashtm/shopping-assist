import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { makeProduct, makeRecommendation } from "@/lib/test-fixtures";

import { ProductCard } from "./ProductCard";

describe("ProductCard", () => {
  it("renders the product title and formatted price", () => {
    render(<ProductCard item={makeRecommendation({ product: makeProduct({ price_inr: 1487 }) })} />);
    expect(screen.getByText("Men's Tech 2.0 Short-Sleeve T-Shirt")).toBeInTheDocument();
    expect(screen.getByText(/1,487/)).toBeInTheDocument();
  });

  it("shows a discount badge when mrp exceeds price", () => {
    render(
      <ProductCard
        item={makeRecommendation({ product: makeProduct({ price_inr: 1487, mrp_inr: 2075 }) })}
      />,
    );
    expect(screen.getByText("28% off")).toBeInTheDocument();
  });

  it("shows no discount badge when there is no mrp", () => {
    render(
      <ProductCard item={makeRecommendation({ product: makeProduct({ mrp_inr: null }) })} />,
    );
    expect(screen.queryByText(/% off/)).not.toBeInTheDocument();
  });

  it("uses the brand as the eyebrow label when it is real", () => {
    render(<ProductCard item={makeRecommendation({ product: makeProduct({ brand: "Reebok" }) })} />);
    expect(screen.getByText("Reebok")).toBeInTheDocument();
  });

  it("falls back to the subcategory when brand is 'Generic'", () => {
    render(
      <ProductCard
        item={makeRecommendation({
          product: makeProduct({ brand: "Generic", subcategory: "Casual Sneakers" }),
        })}
      />,
    );
    expect(screen.getByText("Casual Sneakers")).toBeInTheDocument();
    expect(screen.queryByText("Generic")).not.toBeInTheDocument();
  });

  it("renders the explanation reason", () => {
    render(
      <ProductCard
        item={makeRecommendation({ reason: "Rated to -10C, covering the -15C nights." })}
      />,
    );
    expect(screen.getByText("Rated to -10C, covering the -15C nights.")).toBeInTheDocument();
  });

  it("links to the retailer with the retailer name", () => {
    render(
      <ProductCard
        item={makeRecommendation({
          product: makeProduct({ retailer: "Myntra", product_url: "https://myntra.com/x" }),
        })}
      />,
    );
    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "https://myntra.com/x");
    expect(link).toHaveTextContent("Myntra");
  });

  it("shows no provenance badge for a verified product", () => {
    render(
      <ProductCard item={makeRecommendation({ product: makeProduct({ link_status: "verified" }) })} />,
    );
    expect(screen.queryByText("Archival listing")).not.toBeInTheDocument();
    expect(screen.queryByText("Unverified link")).not.toBeInTheDocument();
  });

  it("shows an 'Archival listing' badge for an archival product", () => {
    render(
      <ProductCard item={makeRecommendation({ product: makeProduct({ link_status: "archival" }) })} />,
    );
    expect(screen.getByText("Archival listing")).toBeInTheDocument();
  });

  it("shows an 'Unverified link' badge for a blocked product", () => {
    render(
      <ProductCard item={makeRecommendation({ product: makeProduct({ link_status: "blocked" }) })} />,
    );
    expect(screen.getByText("Unverified link")).toBeInTheDocument();
  });

  it("shows the rating and review count when present", () => {
    render(
      <ProductCard
        item={makeRecommendation({ product: makeProduct({ rating: 4.1, review_count: 903 }) })}
      />,
    );
    expect(screen.getByText(/4\.1/)).toBeInTheDocument();
    expect(screen.getByText(/903/)).toBeInTheDocument();
  });
});
