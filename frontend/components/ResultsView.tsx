import type { RecommendResponse, UnfilledSlot } from "@/lib/types";

import { ContextStrip } from "./ContextStrip";
import { ContextVariablesPanel } from "./ContextVariablesPanel";
import { ProductCard } from "./ProductCard";

function UnfilledSlotsPanel({ slots }: { slots: UnfilledSlot[] }) {
  if (slots.length === 0) return null;

  const required = slots.filter((s) => s.role === "required");
  const other = slots.filter((s) => s.role !== "required");

  return (
    <div className="border border-maroon/30 bg-surface-muted px-5 py-4">
      <p className="eyebrow text-maroon">Could not cover</p>
      {required.length > 0 && (
        <p className="mt-1 text-sm font-medium">
          This is not a complete answer — required items are missing from the catalogue.
        </p>
      )}
      <ul className="mt-3 space-y-2">
        {[...required, ...other].map((slot) => (
          <li key={slot.name} className="border-l-2 border-maroon/40 pl-3 text-sm">
            <span className="font-medium">{slot.name}</span>
            {slot.role === "required" && (
              <span className="ml-2 text-[10px] font-medium uppercase tracking-wide text-maroon">
                required
              </span>
            )}
            <p className="mt-0.5 text-xs text-muted">{slot.reason}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function ResultsView({ response }: { response: RecommendResponse }) {
  const groups = response.groups ?? [];
  const unfilled = response.unfilled_slots ?? [];
  const contextVariables = response.context_variables ?? [];
  const total = groups.reduce((n, g) => n + g.items.length, 0);
  const missingRequired = unfilled.some((s) => s.role === "required");

  return (
    <section className="animate-rise space-y-10">
      <div className="space-y-4">
        <h2 className="display text-2xl sm:text-[28px]">{response.intent_summary}</h2>
        <ContextStrip context={response.context} />
        <ContextVariablesPanel variables={contextVariables} />

        {missingRequired && (
          <UnfilledSlotsPanel slots={unfilled.filter((s) => s.role === "required")} />
        )}

        {response.assumptions.length > 0 && (
          <div className="text-xs text-muted">
            <p className="eyebrow text-foreground">Assumed for you</p>
            <ul className="mt-1.5 space-y-1">
              {response.assumptions.map((a) => (
                <li key={a} className="border-l border-border pl-3">
                  {a}
                </li>
              ))}
            </ul>
          </div>
        )}

        {response.meta.degraded_mode && (
          <div className="border border-maroon/30 bg-surface-muted px-4 py-3 text-xs text-muted">
            <p>
              <span className="font-medium text-maroon">Reduced mode.</span> These are real
              catalogue matches, but produced without full AI reasoning — so the
              explanations are less specific than usual.
            </p>
            {response.meta.notes.length > 0 && (
              <ul className="mt-1.5 space-y-0.5">
                {response.meta.notes.map((note) => (
                  <li key={note}>— {note}</li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      {groups.length === 0 && (
        <div className="border border-border bg-surface px-5 py-6">
          <p className="text-sm font-medium">Nothing in the catalogue fits that closely.</p>
          <p className="mt-1.5 text-sm text-muted">
            This catalogue holds {response.meta.catalogue_size} products across a dozen
            categories, so it won&rsquo;t cover everything. Try widening the budget, or
            describing the occasion instead of the product.
          </p>
          {response.meta.notes.length > 0 && (
            <ul className="mt-3 space-y-0.5 text-xs text-muted">
              {response.meta.notes.map((note) => (
                <li key={note}>— {note}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {groups.length > 0 && (
        <div className="space-y-2">
          {missingRequired && (
            <p className="text-xs font-medium uppercase tracking-wide text-muted">Offered instead</p>
          )}
          {groups.map((group, index) => (
            <div key={group.name} className="space-y-4">
              <div className="border-t border-foreground pt-4">
                <div className="flex items-baseline gap-3">
                  <span className="eyebrow text-accent">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <h3 className="display text-xl sm:text-2xl">{group.name}</h3>
                </div>
                <p className="editorial mt-1.5 max-w-2xl text-sm leading-relaxed text-muted">
                  {group.why_needed}
                </p>
              </div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                {group.items.map((item) => (
                  <ProductCard key={item.product.id} item={item} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {!missingRequired && unfilled.length > 0 && (
        <UnfilledSlotsPanel slots={unfilled} />
      )}

      <p className="border-t border-border pt-4 text-xs text-muted">
        {total} products from {response.meta.catalogue_size} in the catalogue ·{" "}
        {response.meta.latency_ms} ms
        {response.meta.llm_calls > 0 &&
          ` · ${response.meta.llm_calls} AI call${response.meta.llm_calls === 1 ? "" : "s"}`}
        {response.meta.cached && " · served from cache"}
      </p>
    </section>
  );
}
