/**
 * Mirrors the backend Pydantic schemas (backend/app/schemas/).
 * Kept hand-written rather than generated: the surface is small, and an explicit
 * copy makes contract drift show up as a TypeScript error at the call site.
 */

export type Gender = "men" | "women" | "unisex" | "kids";
export type Season = "summer" | "monsoon" | "winter" | "all-season";
export type LinkStatus = "verified" | "blocked" | "archival";
export type WaterResistance = "none" | "repellent" | "waterproof";
export type Layer = "base" | "mid" | "outer" | "standalone";
export type Formality = "casual" | "smart_casual" | "formal";
export type Breathability = "low" | "medium" | "high";

export interface ProductAttributes {
  gender: Gender;
  season: Season[];
  use_case: string[];
  occasion: string[];
  material: string | null;
  temp_rating_c: number | null;
  is_giftable: boolean;
  water_resistance: WaterResistance | null;
  layer: Layer | null;
  formality: Formality | null;
  breathability: Breathability | null;
}

export interface Product {
  id: string;
  title: string;
  brand: string;
  category: string;
  subcategory: string;
  price_inr: number;
  mrp_inr: number | null;
  description: string;
  attributes: ProductAttributes;
  rating: number | null;
  review_count: number | null;
  retailer: string;
  product_url: string;
  image_url: string | null;
  in_stock: boolean;
  link_status: LinkStatus;
  link_verified_at: string | null;
}

export interface Recommendation {
  product: Product;
  reason: string;
  match_score: number;
}

export interface RecommendationGroup {
  name: string;
  why_needed: string;
  items: Recommendation[];
}

export type ClimateSource =
  | "measured"
  | "climatological"
  | "user"
  | "inferred"
  | "unobtainable";

export interface ClimateContext {
  source: ClimateSource;
  summary: string;
  place_resolved: string | null;
  latitude: number | null;
  longitude: number | null;
  elevation_m: number | null;
  temp_min_c: number | null;
  temp_max_c: number | null;
  precipitation_mm: number | null;
  window_start: string | null;
  window_end: string | null;
  as_of: string | null;
}

export interface ResolvedContext {
  location: string | null;
  start_date: string | null;
  end_date: string | null;
  duration_days: number | null;
  climate_note: string | null;
  recipient: string | null;
  climate: ClimateContext | null;
}

export type ContextVariableStatus = "known" | "needed" | "unobtainable";
export type ContextVariableSource = "user" | "external" | "inferred";

export interface ContextVariable {
  name: string;
  label: string;
  status: ContextVariableStatus;
  source: ContextVariableSource | null;
  value: string | null;
}

export interface UnfilledSlot {
  name: string;
  role: string;
  reason: string;
}

export interface ResponseMeta {
  latency_ms: number;
  llm_calls: number;
  cached: boolean;
  degraded_mode: boolean;
  catalogue_size: number;
  notes: string[];
}

export interface QuestionOption {
  label: string;
  value: string;
}

export interface ClarifyingQuestion {
  slot: string;
  question: string;
  options: QuestionOption[];
  allow_multiple: boolean;
}

export type ResponseMode = "results" | "clarify";

export interface RecommendResponse {
  query_id: string;
  mode: ResponseMode;
  intent_summary: string;
  context: ResolvedContext;
  assumptions: string[];
  context_variables?: ContextVariable[];
  questions: ClarifyingQuestion[];
  groups?: RecommendationGroup[];
  unfilled_slots?: UnfilledSlot[];
  meta: ResponseMeta;
}

export interface RecommendRequest {
  query: string;
  /** Machine values from tapped chips, e.g. ["occasion:daily-wear"]. */
  answers?: string[];
  /** Set by the "just show me now" escape hatch. */
  skip_clarification?: boolean;
}

/** Stages emitted by the streaming endpoint, in the order they occur. */
export type Stage =
  | "interpreting"
  | "checking conditions"
  | "searching"
  | "cached";
