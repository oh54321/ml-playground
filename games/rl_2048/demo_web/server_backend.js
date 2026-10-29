// The Python demo serves this adapter as backend.js. Its controls and board use
// the same app as the static browser demo.
async function request(path, body) {
  const response = await fetch(path, body === undefined ? undefined : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

export const backend = {
  state: () => request("/api/state"),
  newGame: () => request("/api/new", {}),
  move: (action) => request("/api/move", { action }),
  step: () => request("/api/step", {}),
  history: () => request("/api/history"),
  snapshot: (gameId, number) => request(`/api/history/${gameId}/${number}`),
};
