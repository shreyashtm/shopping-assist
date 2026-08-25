import Image from "next/image";

import type { Recommendation } from "@/lib/types";

function formatInr(value: number): string {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
}

/**
 * A shopper should never have to infer provenance from the retailer domain.
 * "archival" records (Kaggle-sourced, Feb 2024, US-priced at a fixed
 * conversion rate) and "blocked" ones (bot protection prevented a live
 * check) never went through the same verification as "verified" listings,
 * so both get an explicit, honest label rather than looking identical to a
 * confirmed Amazon.in/Myntra page.
 */
function ProvenanceBadge({ status }: { status: Recommendation["product"]["link_status"] }) {
  if (status === "verified") return null;
  const label = status === "archival" ? "Archival listing" : "Unverified link";
  const title =
    status === "archival"
      ? "From a historical dataset, not a live scrape -- price and availability are illustrative, not current."
      : "Retailer bot protection prevented confirming this link at catalogue build time.";
  return (
    <span
      title={title}
      className="border border-border px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted"
    >
      {label}
    </span>
  );
}

export function ProductCard({ item }: { item: Recommendation }) {
  const { product, reason } = item;
  // Many Amazon listings genuinely carry no brand; enrichment records that
  // honestly as "Generic", which reads oddly as a label. Fall back to the
  // subcategory, which is always meaningful.
  const label = product.brand && product.brand !== "Generic" ? product.brand : product.subcategory;
  const discount =
    product.mrp_inr && product.mrp_inr > product.price_inr
      ? Math.round((1 - product.price_inr / product.mrp_inr) * 100)
      : null;

  return (
    <article className="group flex flex-col border border-border bg-surface transition-colors hover:border-foreground">
      <div className="relative aspect-square w-full border-b border-border bg-surface-muted">
        {product.image_url ? (
          <Image
            src={product.image_url}
            alt={product.title}
            fill
            sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 25vw"
            className="object-contain p-4"
            unoptimized
          />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-muted">
            No image
          </div>
        )}
        {discount && (
          <span className="absolute left-0 top-0 bg-accent px-2 py-1 text-[11px] font-medium text-accent-contrast">
            {discount}% off
          </span>
        )}
      </div>

      <div className="flex flex-1 flex-col gap-3 p-4">
        <div>
          <p className="eyebrow text-muted">{label}</p>
          <h4 className="mt-1 line-clamp-2 text-sm leading-snug">{product.title}</h4>
        </div>

        <div className="flex items-baseline gap-2">
          <span className="text-lg font-medium tracking-tight">
            {formatInr(product.price_inr)}
          </span>
          {product.mrp_inr && discount && (
            <span className="text-xs text-muted line-through">
              {formatInr(product.mrp_inr)}
            </span>
          )}
        </div>

        {/* The explanation is the product here. Serif and a rule set it apart
            from the listing chrome as an editorial aside rather than a spec. */}
        <p className="editorial border-l-2 border-accent pl-3 text-[13px] leading-relaxed">
          {reason}
        </p>

        <div className="mt-auto flex items-center justify-between gap-2 pt-2 text-xs">
          <span className="text-muted">
            {product.rating ? `${product.rating.toFixed(1)} ★` : ""}
            {product.rating && product.review_count
              ? ` · ${product.review_count.toLocaleString("en-IN")}`
              : ""}
          </span>
          <a
            href={product.product_url}
            target="_blank"
            rel="noopener noreferrer"
            className="font-medium text-accent underline-offset-4 hover:underline"
          >
            {product.retailer} ↗
          </a>
        </div>
        {product.link_status !== "verified" && (
          <div>
            <ProvenanceBadge status={product.link_status} />
          </div>
        )}
      </div>
    </article>
  );
}
