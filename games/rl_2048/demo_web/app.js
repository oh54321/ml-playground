import { backend } from "./backend.js";

const board = document.querySelector("#board");
const score = document.querySelector("#score");
const moves = document.querySelector("#moves");
const topTile = document.querySelector("#top-tile");
const status = document.querySelector("#status");
const error = document.querySelector("#error");
const modelInfo = document.querySelector("#model-info");
const newButton = document.querySelector("#new-game");
const playButton = document.querySelector("#play-pause");
const stepButton = document.querySelector("#step");
const speed = document.querySelector("#speed");
const speedValue = document.querySelector("#speed-value");
const historyPanel = document.querySelector("#history-panel");
const historyCount = document.querySelector("#history-count");
const historyList = document.querySelector("#history-list");
const returnButton = document.querySelector("#return-live");
const directionButtons = [...document.querySelectorAll("[data-action]")];
const actionNames = ["left", "up", "right", "down"];
const keyActions = { ArrowLeft: 0, ArrowUp: 1, ArrowRight: 2, ArrowDown: 3,
  a: 0, w: 1, d: 2, s: 3 };
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

let game = null;
let selectedMove = null;
let preview = null;
let cells = [];
let history = [];
let historyLoadedFor = null;
let historyNeeded = false;
let working = true; // Initial state is loading.
let pending = null;
let playing = false;
let autoTimer = null;
let lastStarted = 0;
let refreshNeeded = false;

function stopAuto() {
  playing = false;
  clearTimeout(autoTimer);
  autoTimer = null;
  if (pending?.auto) pending = null;
  renderControls();
  renderStatus();
}

function renderControls() {
  const ready = game !== null;
  const canMove = ready && !game.done && selectedMove === null;
  returnButton.hidden = selectedMove === null;
  const hasModel = canMove && game.has_model;
  // Keep controls steady while requests run. Intent is queued below.
  for (const button of directionButtons) button.disabled = !canMove;
  playButton.disabled = !hasModel;
  playButton.textContent = playing ? "Pause model" : "Play model";
  stepButton.disabled = !hasModel;
  speed.disabled = !hasModel;
  modelInfo.hidden = ready && game.has_model;
  if (ready && !game.has_model) {
    modelInfo.textContent = game.model_message || "No model loaded. You can still play.";
  }
}

function renderStatus() {
  if (!game) return;
  if (selectedMove !== null) {
    status.textContent = preview
      ? `Viewing ${selectedMove === 0 ? "the starting board" : `move ${selectedMove} of ${game.moves}`}. Live game paused.`
      : `Loading move ${selectedMove}…`;
    return;
  }
  const move = game.last_move;
  const last = move ? `Move ${move.number}: ${actionNames[move.action]} · +${move.reward}. ` : "";
  status.textContent = game.done ? `${last}Game over · ${Number(game.score).toLocaleString()} points`
    : `${last}${playing ? "Model is playing…" : "Your turn."}`;
}

function ensureCells(grid) {
  if (cells.length === grid.length && cells[0]?.length === grid[0]?.length) return;
  cells = [];
  board.replaceChildren();
  board.style.setProperty("--columns", grid[0].length);
  board.setAttribute("aria-label", `${grid.length} by ${grid[0].length} 2048 board`);
  for (const row of grid) {
    const rowElement = document.createElement("div");
    rowElement.className = "board-row";
    rowElement.setAttribute("role", "row");
    const rowCells = [];
    for (const _ of row) {
      const tile = document.createElement("div");
      tile.className = "tile";
      tile.setAttribute("role", "gridcell");
      rowElement.append(tile);
      rowCells.push(tile);
    }
    board.append(rowElement);
    cells.push(rowCells);
  }
}

function showBoard(grid) {
  ensureCells(grid);
  for (let row = 0; row < grid.length; row++) {
    for (let col = 0; col < grid[row].length; col++) {
      const tile = cells[row][col];
      const value = grid[row][col];
      tile.classList.remove("pop");
      tile.textContent = value ? String(value) : "";
      tile.setAttribute("aria-label", value ? String(value) : "Empty");
      tile.dataset.value = String(value);
      tile.dataset.digits = String(value).length;
      tile.dataset.high = String(value > 2048);
    }
  }
}

function historyItem(move) {
  const item = document.createElement("li");
  const button = document.createElement("button");
  button.type = "button";
  button.dataset.move = move.number;
  button.setAttribute("aria-pressed", String(selectedMove === move.number));
  button.textContent = move.number === 0 ? "Starting board"
    : `#${move.number} · ${actionNames[move.action]} · +${move.reward} · score ${move.score}`;
  button.addEventListener("click", () => {
    selectedMove = move.number;
    preview = null;
    queue({ type: "view", number: move.number, gameId: game.game_id });
    renderState();
    for (const entry of historyList.querySelectorAll("button")) {
      entry.setAttribute("aria-pressed", String(Number(entry.dataset.move) === selectedMove));
    }
  });
  item.append(button);
  return item;
}

function renderHistory() {
  if (!game) return;
  historyCount.textContent = `${game.moves} ${game.moves === 1 ? "move" : "moves"}`;
  if (!historyPanel.open) return;
  historyList.replaceChildren();
  if (historyLoadedFor !== game.game_id) {
    const item = document.createElement("li");
    item.textContent = "Loading move history…";
    historyList.append(item);
    return;
  }
  for (const move of [...history].reverse()) {
    historyList.append(historyItem(move));
  }
  historyList.append(historyItem({ number: 0 }));
}

function renderState() {
  const shown = preview || game;
  score.textContent = Number(shown.score).toLocaleString();
  moves.textContent = Number(shown.moves).toLocaleString();
  topTile.textContent = Math.max(0, ...shown.board.flat()).toLocaleString();
  renderStatus();
  renderControls();
  historyCount.textContent = `${game.moves} ${game.moves === 1 ? "move" : "moves"}`;
}

function showError(cause) {
  stopAuto();
  error.textContent = cause instanceof Error ? cause.message : "Could not update the game.";
  error.hidden = false;
}

async function request(operation) {
  const data = await operation();
  error.hidden = true;
  return data;
}

const wait = (milliseconds) => new Promise(resolve => setTimeout(resolve, milliseconds));

async function receiveState(next, animate = true) {
  const previous = game;
  const sameGame = previous?.game_id === next.game_id;
  let addedMove = null;
  if (!sameGame) {
    selectedMove = null;
    preview = null;
    history = [];
    historyLoadedFor = next.moves === 0 ? next.game_id : null;
    historyNeeded = historyPanel.open && next.moves > 0;
  } else if (next.moves > previous.moves && historyLoadedFor === next.game_id) {
    if (next.moves === previous.moves + 1 && next.last_move) {
      history.push(next.last_move);
      addedMove = next.last_move;
    }
    else {
      historyLoadedFor = null;
      historyNeeded = historyPanel.open;
    }
  }
  game = next;
  if (game.done || !game.has_model) stopAuto();
  renderState();
  if (!sameGame || (historyPanel.open && historyLoadedFor !== game.game_id)) renderHistory();
  else if (historyPanel.open && addedMove) {
    historyList.prepend(historyItem(addedMove));
  }

  if (selectedMove !== null) return;
  const moved = animate && sameGame && next.moves === previous.moves + 1
    && next.before_spawn && next.last_move?.spawn;
  if (!moved || reducedMotion.matches || Number(speed.value) >= 35) {
    showBoard(next.board);
    return;
  }
  showBoard(next.before_spawn);
  const rate = Number(speed.value);
  await wait(Math.min(80, 400 / rate));
  if (selectedMove !== null) return;
  showBoard(next.board);
  const { row, col } = next.last_move.spawn;
  const tile = cells[row]?.[col];
  if (tile) {
    tile.style.setProperty("--pop-duration", `${Math.min(130, 600 / rate)}ms`);
    tile.classList.add("pop");
  }
  await wait(Math.min(130, 600 / rate));
}

function scheduleAuto() {
  if (!playing || autoTimer || working || pending || !game || game.done || selectedMove !== null) return;
  const interval = 1000 / Number(speed.value);
  const delay = Math.max(0, interval - (performance.now() - lastStarted));
  autoTimer = setTimeout(() => {
    autoTimer = null;
    if (!playing || pending) return;
    pending = { type: "step", auto: true };
    pump();
  }, delay);
}

async function pump() {
  if (working) return;
  if (pending) {
    const command = pending;
    pending = null;
    if (command.type === "move" && (!game || game.done || !game.legal_actions.includes(command.action))) {
      pump();
      return;
    }
    if (command.type === "step" && (!game?.has_model || game.done)) { pump(); return; }
    working = true;
    lastStarted = performance.now();
    try {
      if (command.type === "view") {
        const snapshot = await request(() => backend.snapshot(command.gameId, command.number));
        if (selectedMove === command.number && game.game_id === snapshot.game_id) {
          preview = snapshot;
          showBoard(preview.board);
          renderState();
        }
        return;
      }
      if (command.type === "live") {
        selectedMove = null;
        preview = null;
        await receiveState(await request(() => backend.state()), false);
        renderHistory();
        return;
      }
      const operation = command.type === "new" ? () => backend.newGame()
        : command.type === "move" ? () => backend.move(command.action) : () => backend.step();
      await receiveState(await request(operation));
      if (command.type === "new") lastStarted = 0;
    } catch (cause) { showError(cause); }
    finally { working = false; pump(); }
    return;
  }
  if (historyNeeded && historyPanel.open && game && historyLoadedFor !== game.game_id) {
    historyNeeded = false;
    working = true;
    try {
      const data = await request(() => backend.history());
      if (pending?.type !== "new") {
        await receiveState(data.state, false);
        history = data.history;
        historyLoadedFor = data.state.game_id;
        historyNeeded = false;
        renderHistory();
      }
    } catch (cause) { historyNeeded = false; showError(cause); }
    finally { working = false; pump(); }
    return;
  }
  if (refreshNeeded) {
    refreshNeeded = false;
    working = true;
    try { await receiveState(await request(() => backend.state()), false); }
    catch (cause) { showError(cause); }
    finally { working = false; pump(); }
    return;
  }
  scheduleAuto();
}

function queue(command) {
  if (selectedMove !== null && (command.type === "move" || command.type === "step")) return;
  stopAuto();
  if (command.type === "new" || pending?.type !== "new") pending = command;
  clearTimeout(autoTimer);
  autoTimer = null;
  pump();
}

newButton.addEventListener("click", () => queue({ type: "new" }));
returnButton.addEventListener("click", () => queue({ type: "live" }));
playButton.addEventListener("click", () => {
  if (playing) { stopAuto(); return; }
  if (!game?.has_model || game.done || selectedMove !== null) return;
  playing = true;
  lastStarted = 0;
  renderControls();
  renderStatus();
  pump();
});
stepButton.addEventListener("click", () => queue({ type: "step" }));
for (const button of directionButtons) {
  button.addEventListener("click", () => queue({ type: "move", action: Number(button.dataset.action) }));
}
speed.addEventListener("input", () => {
  speedValue.textContent = `${speed.value} moves/sec`;
  if (playing && autoTimer) { clearTimeout(autoTimer); autoTimer = null; scheduleAuto(); }
});
historyPanel.addEventListener("toggle", () => {
  if (!historyPanel.open || !game) return;
  if (historyLoadedFor !== game.game_id) historyNeeded = true;
  renderHistory();
  pump();
});
window.addEventListener("keydown", (event) => {
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.target instanceof Element && event.target.closest("input, textarea, select, [contenteditable]")) return;
  const action = keyActions[event.key.length === 1 ? event.key.toLowerCase() : event.key];
  if (action === undefined || !game || game.done) return;
  event.preventDefault();
  queue({ type: "move", action });
});
let touchStart = null;
board.addEventListener("touchstart", (event) => {
  const touch = event.changedTouches[0];
  touchStart = { x: touch.clientX, y: touch.clientY };
}, { passive: true });
board.addEventListener("touchend", (event) => {
  if (!touchStart) return;
  const touch = event.changedTouches[0];
  const dx = touch.clientX - touchStart.x;
  const dy = touch.clientY - touchStart.y;
  touchStart = null;
  if (Math.max(Math.abs(dx), Math.abs(dy)) < 28) return;
  queue({ type: "move", action: Math.abs(dx) > Math.abs(dy) ? (dx < 0 ? 0 : 2) : (dy < 0 ? 1 : 3) });
}, { passive: true });

renderControls();
(async () => {
  try { await receiveState(await request(() => backend.state())); }
  catch (cause) { showError(cause); }
  finally { working = false; pump(); }
  if (backend.ready) {
    try {
      await backend.ready();
      refreshNeeded = true;
      pump();
    } catch (cause) { showError(cause); }
  }
})();
