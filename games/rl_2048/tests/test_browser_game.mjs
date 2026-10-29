import assert from 'node:assert/strict';
import test from 'node:test';
import {BrowserGame, encodeBoard, slide} from '../demo_web/game.js';

test('each tile merges once, with numeric tiles above the 32-bit range', () => {
  const board = [[2, 2, 2, 2], [4, 4, 8, 0], [2 ** 32, 2 ** 32, 0, 0]];
  assert.deepEqual(slide(board, 0), {
    board: [[4, 4, 0, 0], [8, 8, 0, 0], [2 ** 33, 0, 0, 0]],
    reward: 16 + 2 ** 33, moved: true,
  });
  assert.deepEqual(board[0], [2, 2, 2, 2]);
  assert.deepEqual(slide([[2, 2, 4, 0]], 2), {board: [[0, 0, 4, 4]], reward: 4, moved: true});
});

test('rectangular boards slide vertically and no-op moves return independent boards', () => {
  const board = [[2, 0, 4], [2, 4, 4]];
  assert.deepEqual(slide(board, 1), {board: [[4, 4, 8], [0, 0, 0]], reward: 12, moved: true});
  assert.deepEqual(slide(board, 3), {board: [[0, 0, 0], [4, 4, 8]], reward: 12, moved: true});
  const stable = [[2, 4, 8], [4, 8, 16]];
  for (const action of [0, 1, 2, 3]) {
    const result = slide(stable, action);
    assert.deepEqual(result, {board: stable, reward: 0, moved: false});
    assert.notEqual(result.board[0], stable[0]);
  }
});

test('encoding preserves NCHW numeric and exponent-category channel identities', () => {
  const board = [[0, 2, 4], [8, 16, 128]];
  assert.deepEqual(encodeBoard(board, 'numeric'), new Float32Array([0, 1, 2, 3, 4, 7]));
  const encoded = encodeBoard(board, 'numeric_onehot');
  assert.equal(encoded.length, 9 * 6);
  for (let channel = 1; channel < 9; channel++) {
    assert.deepEqual([...encoded.slice(channel * 6, (channel + 1) * 6)],
      [0, 1, 2, 3, 4, 7].map(exponent => Number(exponent === channel - 1)));
  }
  assert.throws(() => encodeBoard([[0, 256, 0], [0, 0, 0]], 'numeric_onehot'), /exponent/);
});

test('history snapshots are immutable reads and reset invalidates old history', () => {
  const random = Math.random;
  Math.random = () => 0;
  try {
    const game = new BrowserGame({height: 2, width: 3});
    const initial = game.state();
    assert.deepEqual(initial.board, [[2, 2, 0], [0, 0, 0]]);
    assert.equal(initial.last_move, null);
    assert.equal(initial.before_spawn, null);
    const moved = game.move(0);
    assert.deepEqual(moved.before_spawn, [[4, 0, 0], [0, 0, 0]]);
    assert.deepEqual(moved.board, [[4, 2, 0], [0, 0, 0]]);
    assert.deepEqual(game.history, [{number: 1, action: 0, reward: 4, score: 4,
      spawn: {row: 0, col: 1, value: 2}}]);
    const saved = game.snapshot(initial.game_id, 1);
    saved.board[0][0] = 1024;
    saved.last_move.spawn.value = 1024;
    moved.board[0][0] = 1024;
    moved.before_spawn[0][0] = 1024;
    game.history[0].spawn.value = 1024;
    assert.deepEqual(game.snapshot(initial.game_id, 0).board, initial.board);
    assert.deepEqual(game.snapshot(initial.game_id, 1), {
      game_id: initial.game_id, moves: 1, board: [[4, 2, 0], [0, 0, 0]], score: 4,
      last_move: {number: 1, action: 0, reward: 4, score: 4, spawn: {row: 0, col: 1, value: 2}},
    });
    const beforeNoOp = game.state();
    assert.deepEqual(game.move(0), beforeNoOp);
    assert.equal(game.history.length, 1);
    assert.throws(() => game.move(4), /Choose/);
    assert.throws(() => game.snapshot(initial.game_id, 2), /Move not found/);
    const reset = game.newGame();
    assert.notEqual(reset.game_id, initial.game_id);
    assert.equal(reset.moves, 0);
    assert.equal(reset.score, 0);
    assert.deepEqual(game.history, []);
    assert.throws(() => game.snapshot(initial.game_id, 0), /previous game/);
    assert.deepEqual(game.snapshot(reset.game_id, 0).board, reset.board);
  } finally {
    Math.random = random;
  }
});
