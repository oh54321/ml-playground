"""Play locally: python -m games.rl_2048.demo --checkpoint <run>/best.pt."""

import argparse
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets

import numpy as np
import torch

from core.models.conv_actor_critic import ConvActorCriticConfig, ConvActorCriticModel
from core.models.d4_conv_actor_critic import D4ConvActorCriticModel
from games.rl_2048.environment.game_2048 import Game2048, ObservationEncoding
from games.rl_2048.environment._game_2048_kernels import move_int64
from games.rl_2048.environment.symmetry_2048 import SQUARE_SYMMETRY_SPECS


def load_policy(path: Path):
    """Load inference weights and board settings from a clean training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = ConvActorCriticConfig(**checkpoint["model_config"])
    encoding = ObservationEncoding(checkpoint["observation_encoding"])
    if (config.output_dim != 4 or config.height * config.width < 3
            or config.in_channels != encoding.in_channels(config.height, config.width)):
        raise ValueError("Checkpoint dimensions do not match a 2048 policy")
    if checkpoint["architecture"] == "cnn":
        model = ConvActorCriticModel(config)
    elif checkpoint["architecture"] == "d4":
        permutations = {spec.transform: spec.action_permutation for spec in SQUARE_SYMMETRY_SPECS}
        model = D4ConvActorCriticModel(config, permutations)
    else:
        raise ValueError("Unknown checkpoint architecture")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, encoding


class DemoGame:
    """One local game, shared by the browser's manual and model controls."""

    def __init__(self, checkpoint: Path | None = None):
        self.model = None
        settings = {}
        if checkpoint is not None:
            self.model, encoding = load_policy(checkpoint)
            settings = dict(height=self.model.config.height, width=self.model.config.width,
                            observation_encoding=encoding)
        self.game_id = secrets.randbits(53)
        self.game = Game2048(seed=self.game_id, **settings)
        self.history = []
        self.before_spawn = None
        self.boards = [self.game.state.grid.tolist()]

    def state(self) -> dict:
        last_move = self.history[-1] if self.history else None
        return {"game_id": self.game_id, "board": self.game.state.grid.tolist(), "score": self.game.total_score,
                "moves": len(self.history), "done": self.game.done,
                "legal_actions": self.game.legal_actions(), "has_model": self.model is not None,
                "last_move": last_move, "before_spawn": self.before_spawn}

    def new_game(self) -> dict:
        self.game_id = secrets.randbits(53)
        self.game.reset(seed=self.game_id)
        self.history.clear()
        self.before_spawn = None
        self.boards = [self.game.state.grid.tolist()]
        return self.state()

    def snapshot(self, game_id: int, number: int) -> dict:
        """Read a saved board without changing the live game or its randomness."""
        if game_id != self.game_id:
            raise ValueError("This history belongs to a previous game. Return to the live game.")
        if not 0 <= number < len(self.boards):
            raise ValueError("Move not found")
        move = self.history[number - 1] if number else None
        return {"game_id": game_id, "moves": number, "board": self.boards[number],
                "score": move["score"] if move else 0, "last_move": move}

    def move(self, action: int) -> dict:
        if type(action) is not int or not 0 <= action < 4:
            raise ValueError("Choose left, up, right, or down")
        if not self.game.done and action in self.game.legal_actions():
            # Use the game's own merge kernel to expose the board before spawning.
            # This preview does not consume randomness or change the game state.
            merged = np.zeros_like(self.game.state.grid)
            move_int64(self.game.state.grid, action, merged)
            result = self.game.step(action)
            row, col = np.argwhere(self.game.state.grid != merged)[0]
            self.before_spawn = merged.tolist()
            self.history.append({"number": len(self.history) + 1, "action": action,
                                 "reward": int(result.reward), "score": self.game.total_score,
                                 "spawn": {"row": int(row), "col": int(col),
                                           "value": int(self.game.state.grid[row, col])}})
            self.boards.append(self.game.state.grid.tolist())
        return self.state()

    @torch.inference_mode()
    def model_step(self) -> dict:
        if self.model is None:
            raise ValueError("Start the demo with --checkpoint to enable the model")
        if self.game.done:
            return self.state()
        legal = self.game.legal_actions()
        logits = self.model(self.game.state.to_tensor().unsqueeze(0)).action_logits[0]
        if not torch.isfinite(logits).all():
            raise ValueError("The model returned nonfinite action scores")
        action = legal[int(logits[legal].argmax())]
        return self.move(action)


class DemoHandler(BaseHTTPRequestHandler):
    def reply(self, body: bytes, status: int = 200, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json_reply(self, data: dict, status: int = 200) -> None:
        self.reply(json.dumps(data).encode(), status)

    def local_request(self) -> bool:
        hosts = {f"localhost:{self.server.server_port}", f"127.0.0.1:{self.server.server_port}"}
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in hosts or (origin and origin not in {f"http://{host}" for host in hosts}):
            self.json_reply({"error": "This demo accepts local browser requests only"}, 403)
            return False
        return True

    def do_GET(self) -> None:
        if not self.local_request():
            return
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            self.json_reply(self.server.demo.state())
            return
        if path == "/api/history":
            self.json_reply({"state": self.server.demo.state(), "history": self.server.demo.history})
            return
        if path.startswith("/api/history/"):
            try:
                game_id, number = map(int, path.removeprefix("/api/history/").split("/"))
                self.json_reply(self.server.demo.snapshot(game_id, number))
            except ValueError as error:
                self.json_reply({"error": str(error)}, 400)
            return
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8")}
        if path not in assets:
            self.json_reply({"error": "Not found"}, 404)
            return
        filename, content_type = assets[path]
        self.reply((Path(__file__).with_name("demo_web") / filename).read_bytes(), content_type=content_type)

    def do_POST(self) -> None:
        if not self.local_request():
            return
        if self.headers.get_content_type() != "application/json":
            self.json_reply({"error": "Expected JSON"}, 400)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024:
                raise ValueError("Expected a small JSON request")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            if self.path == "/api/new":
                state = self.server.demo.new_game()
            elif self.path == "/api/move":
                state = self.server.demo.move(body.get("action"))
            elif self.path == "/api/step":
                state = self.server.demo.model_step()
            else:
                self.json_reply({"error": "Not found"}, 404)
                return
            self.json_reply(state)
        except (ValueError, UnicodeDecodeError) as error:
            self.json_reply({"error": str(error)}, 400)

    def log_message(self, format, *args) -> None:
        pass  # Autoplay makes many requests; keep the terminal quiet.


class DemoServer(HTTPServer):
    def __init__(self, port: int, demo: DemoGame):
        self.demo = demo
        super().__init__(("127.0.0.1", port), DemoHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play 2048 locally, with an optional trained model")
    parser.add_argument("--checkpoint", type=Path, help="a clean training checkpoint, such as best.pt")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    demo = DemoGame(args.checkpoint)
    inference = demo.model.prepared_inference() if demo.model is not None else nullcontext()
    with inference, DemoServer(args.port, demo) as server:
        print(f"Open http://127.0.0.1:{server.server_port} — Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
