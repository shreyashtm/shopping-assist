/**
 * A message the shopper sent.
 *
 * Right-aligned and bordered rather than a rounded bubble: the design language
 * is zero-radius throughout, so the turn is distinguished by alignment and an
 * accent rule instead of by a pill shape.
 */
export function UserMessage({ text }: { text: string }) {
  return (
    <div className="animate-rise flex justify-end">
      <div className="max-w-[42rem] border-r-2 border-accent bg-surface-muted px-4 py-3">
        <p className="whitespace-pre-wrap text-sm leading-relaxed">{text}</p>
      </div>
    </div>
  );
}
