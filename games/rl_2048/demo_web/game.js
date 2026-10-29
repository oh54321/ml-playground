const copyBoard = board => board.map(row => [...row]);
const copyMove = move => move ? {...move, spawn: {...move.spawn}} : null;

function dimensions(board) {
  const height = board.length, width = board[0]?.length;
  if (!height || !width || board.some(row => row.length !== width)) {
    throw new Error('Expected a nonempty rectangular board');
  }
  if (board.some(row => row.some(tile => !Number.isFinite(tile) ||
      (tile !== 0 && (tile < 1 || 2 ** Math.round(Math.log2(tile)) !== tile))))) {
    throw new Error('Tiles must be zero or positive powers of two');
  }
  return {height, width};
}

// Actions match the Python environment: left, up, right, down.
export function slide(board, action) {
  if (!Number.isInteger(action) || action < 0 || action > 3) {
    throw new Error('Choose left, up, right, or down');
  }
  const {height, width} = dimensions(board);
  const output = board.map(row => row.map(() => 0));
  const vertical = action === 1 || action === 3;
  const reverse = action === 2 || action === 3;
  const length = vertical ? height : width;
  const coordinate = (line, position) => {
    const offset = reverse ? length - position - 1 : position;
    return vertical ? [offset, line] : [line, offset];
  };
  let reward = 0;
  for (let line = 0; line < (vertical ? width : height); line++) {
    const tiles = [];
    for (let position = 0; position < length; position++) {
      const [row, col] = coordinate(line, position);
      if (board[row][col]) tiles.push(board[row][col]);
    }
    for (let source = 0, target = 0; source < tiles.length; target++) {
      let value = tiles[source++];
      if (source < tiles.length && value === tiles[source]) {
        value *= 2;
        reward += value;
        source++;
      }
      const [row, col] = coordinate(line, target);
      output[row][col] = value;
    }
  }
  const moved = output.some((row, r) => row.some((value, c) => value !== board[r][c]));
  return {board: output, reward, moved};
}

// Flat NCHW, batch size 1: numeric exponents followed by exponent categories.
export function encodeBoard(board, encoding = 'numeric') {
  const {height, width} = dimensions(board);
  if (!['numeric', 'numeric_onehot'].includes(encoding)) throw new Error('Unknown observation encoding');
  const cells = height * width, channels = encoding === 'numeric' ? 1 : cells + 3;
  const result = new Float32Array(cells * channels);
  board.flat().forEach((tile, index) => {
    const exponent = tile ? Math.log2(tile) : 0;
    result[index] = exponent;
    if (channels > 1) {
      if (exponent >= channels - 1) throw new Error('Tile exponent exceeds one-hot observation range');
      result[(exponent + 1) * cells + index] = 1;
    }
  });
  return result;
}

export class BrowserGame {
  #board;
  #boards = [];
  #history = [];
  #beforeSpawn = null;
  #score = 0;
  #gameId = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);

  constructor({height = 4, width = 4} = {}) {
    if (![height, width].every(value => Number.isInteger(value) && value > 0)) {
      throw new Error('Board dimensions must be positive integers');
    }
    this.#board = Array.from({length: height}, () => Array(width).fill(0));
    this.newGame();
  }

  get history() { return this.#history.map(copyMove); }

  #spawn() {
    const empty = [];
    this.#board.forEach((row, r) => row.forEach((value, c) => { if (!value) empty.push([r, c]); }));
    const [row, col] = empty[Math.floor(Math.random() * empty.length)];
    const value = Math.random() < 0.9 ? 2 : 4;
    this.#board[row][col] = value;
    return {row, col, value};
  }

  state() {
    const legal = [0, 1, 2, 3].filter(action => slide(this.#board, action).moved);
    return {game_id: this.#gameId, board: copyBoard(this.#board), score: this.#score,
      moves: this.#history.length, done: legal.length === 0, legal_actions: legal,
      last_move: copyMove(this.#history.at(-1)),
      before_spawn: this.#beforeSpawn && copyBoard(this.#beforeSpawn)};
  }

  newGame() {
    this.#gameId = (this.#gameId + 1) % Number.MAX_SAFE_INTEGER;
    this.#board = this.#board.map(row => row.map(() => 0));
    this.#history = [];
    this.#score = 0;
    this.#beforeSpawn = null;
    for (let i = 0; i < Math.min(2, this.#board.length * this.#board[0].length); i++) this.#spawn();
    this.#boards = [copyBoard(this.#board)];
    return this.state();
  }

  move(action) {
    const result = slide(this.#board, action);
    if (result.moved) {
      this.#board = result.board;
      this.#beforeSpawn = copyBoard(this.#board);
      this.#score += result.reward;
      const spawn = this.#spawn();
      this.#history.push({number: this.#history.length + 1, action,
        reward: result.reward, score: this.#score, spawn});
      this.#boards.push(copyBoard(this.#board));
    }
    return this.state();
  }

  snapshot(gameId, number) {
    if (gameId !== this.#gameId) throw new Error('This history belongs to a previous game. Return to the live game.');
    if (!Number.isInteger(number) || number < 0 || number >= this.#boards.length) throw new Error('Move not found');
    const move = number ? this.#history[number - 1] : null;
    return {game_id: gameId, moves: number, board: copyBoard(this.#boards[number]),
      score: move ? move.score : 0, last_move: copyMove(move)};
  }
}
