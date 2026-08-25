import type { LinkStatus, Product, Recommendation, RecommendResponse } from "./types";

export function makeProduct(overrides: Partial<Product> = {}): Product {
  return {
    id: "p1",
    title: "Men's Tech 2.0 Short-Sleeve T-Shirt",
    brand: "Under Armour",
    category: "Men's Apparel",
    subcategory: "T-Shirts",
    price_inr: 1487,
    mrp_inr: 2075,
    description: "A test product.",
    attributes: {
      gender: "men",
      season: ["all-season"],
      use_case: [],
      occasion: [],
      material: "polyester",
      temp_rating_c: null,
      is_giftable: false,
      water_resistance: null,
      layer: null,
      formality: null,
      breathability: null,
    },
    rating: 4.5,
    review_count: 903,
    retailer: "Amazon.in",
    product_url: "https://example.com/product",
    image_url: "https://example.com/image.jpg",
    in_stock: true,
    link_status: "verified",
    link_verified_at: null,
    ...overrides,
  };
}

export function makeRecommendation(overrides: Partial<Recommendation> = {}): Recommendation {
  return {
    product: makeProduct(),
    reason: "Suits everyday; built with polyester.",
    match_score: 0.95,
    ...overrides,
  };
}

export function makeLinkStatusProduct(status: LinkStatus): Product {
  return makeProduct({ link_status: status });
}

export function makeResponse(overrides: Partial<RecommendResponse> = {}): RecommendResponse {
  return {
    query_id: "q1",
    mode: "results",
    intent_summary: "A casual t-shirt.",
    context: {
      location: null,
      start_date: null,
      end_date: null,
      duration_days: null,
      climate_note: null,
      recipient: null,
      climate: null,
    },
    assumptions: [],
    context_variables: [],
    questions: [],
    groups: [],
    unfilled_slots: [],
    meta: {
      latency_ms: 1000,
      llm_calls: 1,
      cached: false,
      degraded_mode: false,
      catalogue_size: 1738,
      notes: [],
    },
    ...overrides,
  };
}
