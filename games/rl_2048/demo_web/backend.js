import { BrowserGame, encodeBoard } from "./game.js";

let game;
let metadata;
let initialization;
let policyLoading;
let session;
let ort;
let modelMessage = "Loading model… You can already play.";

async function fetchAsset(path) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(new URL(path, import.meta.url), { signal: controller.signal });
    if (!response.ok) throw new Error(`${path} could not be loaded (${response.status}).`);
    // Consume the body before clearing the timeout.
    return path.endsWith(".json") ? await response.json() : await response.arrayBuffer();
  } finally { clearTimeout(timeout); }
}

function modelUnavailable(cause) {
  const detail = cause instanceof Error ? cause.message : String(cause);
  modelMessage = `Model unavailable: ${detail} You can still play.`;
}

function ensureGame() {
  if (!initialization) {
    initialization = (async () => {
      try {
        const data = await fetchAsset("model.json");
        const { height, width, observation_encoding: encoding, in_channels: channels } = data;
        if (!Number.isInteger(height) || !Number.isInteger(width) || height < 1 || width < 1
            || height * width < 3 || !["numeric", "numeric_onehot"].includes(encoding)
            || channels !== (encoding === "numeric" ? 1 : height * width + 3)) {
          throw new Error("model.json has unsupported board or observation settings.");
        }
        metadata = data;
      } catch (cause) { modelUnavailable(cause); }
      game = new BrowserGame(metadata || {});
      return game;
    })();
  }
  return initialization;
}

async function loadPolicy() {
  await ensureGame();
  if (!metadata) return;
  let timeout;
  let timedOut = false;
  try {
    await Promise.race([
      (async () => {
        const runtime = await import("./vendor/ort.wasm.min.mjs");
        ort = runtime.default || runtime;
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
        const model = await fetchAsset("policy.onnx");
        const loaded = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
        if (timedOut) {
          await loaded.release();
          throw new Error("Loading the model timed out.");
        }
        if (!loaded.inputNames.includes("observation") || !loaded.outputNames.includes("logits")) {
          await loaded.release();
          throw new Error("The policy must accept observation and return logits.");
        }
        return loaded;
      })(),
      new Promise((_, reject) => {
        timeout = setTimeout(() => {
          timedOut = true;
          reject(new Error("Loading the model timed out. Reload to retry."));
        }, 60000);
      }),
    ]).then(loaded => { session = loaded; });
    modelMessage = "";
  } catch (cause) { modelUnavailable(cause); }
  finally { clearTimeout(timeout); }
}

function currentState() {
  return { ...game.state(), has_model: Boolean(session), model_message: modelMessage };
}

export const backend = {
  async state() {
    await ensureGame();
    return currentState();
  },
  ready() {
    if (!policyLoading) policyLoading = loadPolicy();
    return policyLoading;
  },
  async newGame() {
    await ensureGame();
    game.newGame();
    return currentState();
  },
  async move(action) {
    await ensureGame();
    game.move(action);
    return currentState();
  },
  async step() {
    await ensureGame();
    if (!session) throw new Error(modelMessage || "No model loaded.");
    const state = game.state();
    if (state.done) return currentState();
    const observation = new ort.Tensor("float32", encodeBoard(state.board, metadata.observation_encoding),
      [1, metadata.in_channels, metadata.height, metadata.width]);
    const result = await session.run({ observation });
    const logits = result.logits?.data;
    if (!logits || logits.length !== 4 || !Array.from(logits).every(Number.isFinite)) {
      throw new Error("The model returned invalid action scores.");
    }
    const legal = [...state.legal_actions].sort((a, b) => a - b);
    let action = legal[0];
    for (const candidate of legal) {
      if (logits[candidate] > logits[action]) action = candidate;
    }
    game.move(action);
    return currentState();
  },
  async history() {
    await ensureGame();
    return { state: currentState(), history: game.history };
  },
  async snapshot(gameId, number) {
    await ensureGame();
    return game.snapshot(gameId, number);
  },
};
