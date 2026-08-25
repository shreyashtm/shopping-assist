import type { ClimateContext, ResolvedContext } from "@/lib/types";

const PROVENANCE_LABEL: Record<
  NonNullable<ClimateContext["source"]>,
  { text: string; caution: boolean }
> = {
  measured: { text: "Forecast", caution: false },
  climatological: { text: "Typical for this time of year", caution: false },
  user: { text: "You told us", caution: false },
  inferred: { text: "Estimated — not verified", caution: true },
  unobtainable: { text: "Could not verify", caution: true },
};

/** Trip facts and measured conditions, shared by results and clarify views. */
export function ContextStrip({ context }: { context: ResolvedContext }) {
  const facts = [
    context.location,
    context.duration_days ? `${context.duration_days} days` : null,
    context.start_date
      ? new Date(context.start_date).toLocaleDateString("en-IN", {
          day: "numeric",
          month: "short",
          year: "numeric",
        })
      : null,
    context.recipient ? `for ${context.recipient}` : null,
  ].filter(Boolean) as string[];

  const climate = context.climate;
  const note = climate?.summary || context.climate_note;
  const provenance = climate ? PROVENANCE_LABEL[climate.source] : null;

  if (!facts.length && !note) return null;

  return (
    <div className="border-l-2 border-teal bg-surface-muted px-4 py-3">
      {facts.length > 0 && (
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          {facts.map((f) => (
            <span key={f} className="text-xs font-medium">
              {f}
            </span>
          ))}
        </div>
      )}
      {note && (
        <div className="mt-2 space-y-1">
          {provenance && (
            <p
              className={`text-[10px] font-medium uppercase tracking-wide ${
                provenance.caution ? "text-maroon" : "text-muted"
              }`}
            >
              {provenance.text}
            </p>
          )}
          <p className="editorial text-[13px] leading-relaxed text-muted">{note}</p>
        </div>
      )}
    </div>
  );
}
