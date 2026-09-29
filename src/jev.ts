// Calls to Jev (TypeSafe AI) from the browser. api.typesafe.ai sends no CORS
// headers for this site, so every call goes through the Netlify Function in
// netlify/functions/jev.ts, which forwards the key and never stores or logs it.
import type { JevRequest, JevResponse } from "./agent";

const PROXY = "/.netlify/functions/jev";

async function call<T>(key: string, init: RequestInit): Promise<T> {
  const response = await fetch(PROXY, { ...init, headers: { authorization: `Bearer ${key}`, "content-type": "application/json" } });
  const body = await response.json().catch(() => null);
  if (response.ok) return body as T;
  const detail = body?.detail?.message ?? body?.detail ?? body?.message;
  if (response.status === 401 || response.status === 403) throw new Error(`Jev rejected the key (${response.status}). ${detail ?? ""}`.trim());
  if (response.status === 429 || response.status === 529) throw new Error(`Jev is busy (${response.status}). Try again in a moment.`);
  throw new Error(`Jev call failed (${response.status}). ${typeof detail === "string" ? detail : JSON.stringify(detail ?? "")}`.trim());
}

export const askJev = (key: string, request: JevRequest, signal: AbortSignal) =>
  call<JevResponse>(key, { method: "POST", body: JSON.stringify(request), signal });

// GET /v1/models lists the names the account can send, currently the aliases.
export const listModels = async (key: string) =>
  (await call<{ models: { name: string }[] }>(key, { method: "GET" })).models.map((m) => m.name);

// TypeSafe's published price per token, by the versioned model ID a response names (docs.typesafe.ai/models,
// checked 2026-09-29): Jev 1.13 costs $0.042 per million input tokens, and output tokens are free.
// A model not listed here has no known price: the page shows its tokens and "cost unknown".
const PRICES: Record<string, { input: number; output: number }> = { "jev-1.13.0": { input: 0.042e-6, output: 0 } };
export const PRICE_NOTE = "Jev 1.13: $0.042 per million input tokens, output tokens free (docs.typesafe.ai/models)";

// Dollars for one call, or null when the model's price or the token count is unknown.
export function jevCost(model: string, usage: JevResponse["usage"]) {
  const price = PRICES[model];
  return price && usage ? usage.input_tokens * price.input + usage.output_tokens * price.output : null;
}
