import type { ContextVariable } from "@/lib/types";

const STATUS_STYLE: Record<ContextVariable["status"], string> = {
  known: "border-border text-foreground",
  needed: "border-accent text-accent",
  unobtainable: "border-maroon/40 text-maroon",
};

const SOURCE_LABEL: Record<NonNullable<ContextVariable["source"]>, string> = {
  user: "you",
  external: "looked up",
  inferred: "inferred",
};

export function ContextVariablesPanel({ variables }: { variables: ContextVariable[] }) {
  if (variables.length === 0) return null;

  return (
    <div className="text-xs text-muted">
      <p className="eyebrow text-foreground">Planning context</p>
      <ul className="mt-2 flex flex-wrap gap-2">
        {variables.map((v) => (
          <li
            key={v.name}
            className={`border px-2.5 py-1 ${STATUS_STYLE[v.status]}`}
            title={v.value ?? undefined}
          >
            <span className="font-medium">{v.label}</span>
            {v.status === "known" && v.value && (
              <span className="text-muted"> · {v.value.slice(0, 48)}</span>
            )}
            {v.status === "known" && v.source && (
              <span className="text-muted"> ({SOURCE_LABEL[v.source]})</span>
            )}
            {v.status === "needed" && <span> · needed</span>}
            {v.status === "unobtainable" && <span> · unverified</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}
