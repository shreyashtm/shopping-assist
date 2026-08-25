import type { RecommendRequest, RecommendResponse, Stage } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

/**
 * Posts a shopping request.
 *
 * Returns either recommendations or the questions needed to produce them --
 * see `RecommendResponse.mode`. Answering re-posts the same query with
 * `answers` populated.
 */
export async function recommend(
  body: RecommendRequest,
  signal?: AbortSignal,
): Promise<RecommendResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/v1/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError("Could not reach the assistant. Is the API running?", 0);
  }

  if (!res.ok) {
    const detail = await res
      .json()
      .then((d: { detail?: string }) => d.detail)
      .catch(() => null);
    throw new ApiError(detail ?? `Request failed (${res.status})`, res.status);
  }
  return res.json();
}


/**
 * Streams a shopping request, reporting each pipeline stage as it happens.
 *
 * Uses fetch + a ReadableStream rather than EventSource because EventSource is
 * GET-only, and putting the shopper's request text into a URL is both fragile
 * and needless exposure. Falls back to the blocking endpoint if streaming is
 * unavailable, so the feature can never make the app worse than before it.
 */
export async function recommendStreaming(
  body: RecommendRequest,
  onStage: (stage: Stage) => void,
  signal?: AbortSignal,
): Promise<RecommendResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/v1/recommend/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError("Could not reach the assistant. Is the API running?", 0);
  }

  if (!res.ok || !res.body) return recommend(body, signal);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: RecommendResponse | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line; keep any partial tail.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const event = /^event: (.+)$/m.exec(frame)?.[1];
      const raw = /^data: (.+)$/m.exec(frame)?.[1];
      if (!event || !raw) continue;
      const data = JSON.parse(raw);
      if (event === "stage") onStage(data.stage as Stage);
      else if (event === "result") result = data as RecommendResponse;
      else if (event === "error") throw new ApiError(data.detail, 500);
    }
  }

  if (!result) throw new ApiError("The assistant stopped before returning results.", 500);
  return result;
}
