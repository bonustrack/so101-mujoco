// Proxy from this site to Jev, because api.typesafe.ai does not allow browser
// calls from other origins. GET lists models, POST asks a question. The API key
// arrives in the Authorization header of each request and is passed on as is:
// it is never stored, and nothing here logs requests.
const API = "https://api.typesafe.ai/v1";

export default async function jev(request: Request): Promise<Response> {
  if (request.method !== "GET" && request.method !== "POST") return new Response(null, { status: 405 });
  const authorization = request.headers.get("authorization");
  if (!authorization?.startsWith("Bearer ")) return Response.json({ detail: { message: "Add your Jev API key." } }, { status: 401 });
  const upstream = await fetch(`${API}/${request.method === "GET" ? "models" : "systemone"}`, {
    method: request.method,
    headers: { authorization, "content-type": "application/json" },
    body: request.method === "POST" ? await request.text() : undefined,
  });
  return new Response(await upstream.text(), {
    status: upstream.status,
    headers: { "content-type": upstream.headers.get("content-type") ?? "application/json", "cache-control": "no-store" },
  });
}
