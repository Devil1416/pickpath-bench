"""
server/app.py — OpenEnv-compliant server entry point
=====================================================
Required by the OpenEnv multi-mode deployment spec.
Exposes the full OpenEnv REST API:
  POST /reset   — reset environment, return initial observation
  POST /step    — take one action, return RewardModel
  GET  /state   — return current observation (no side effects)
  GET  /tasks   — list all available tasks
  GET  /health  — liveness probe

Also mounts the full web UI (grid visualiser + SSE runner) inherited
from the root app.py so a single process satisfies both the API checker
and the Hugging Face Space dashboard requirement.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
from typing import Generator

from flask import Flask, Response, jsonify, request

# ── make sure env/ and project root are importable from any working directory ─
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from env.env import WarehouseBotEnv          # noqa: E402
from env.graders import grade_episode        # noqa: E402
from env.models import ObservationModel      # noqa: E402
from env.tasks import get_task, list_tasks   # noqa: E402

try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False

# ── config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MAX_STEPS    = 120

_client = None
if HF_TOKEN and _openai_available:
    _client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ── planner helpers (identical to root app.py) ────────────────────────────────
ACTION_DELTAS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ACTION_ORDER  = ("right", "down", "left", "up")


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _nearest(pos, items):
    return min(items, key=lambda p: (_manhattan(pos, p), p))


def _planner_action(obs, target, visit_counts, prev_pos):
    cur   = (obs.agent_position.row, obs.agent_position.col)
    walls = {(o.row, o.col) for o in obs.obstacles}
    moves = []
    for act in ACTION_ORDER:
        dr, dc = ACTION_DELTAS[act]
        nxt = (cur[0] + dr, cur[1] + dc)
        if not (0 <= nxt[0] < obs.grid_size and 0 <= nxt[1] < obs.grid_size):
            continue
        if nxt in walls:
            continue
        moves.append((act, nxt, _manhattan(nxt, target)))
    if not moves:
        return "up"
    moves = [m for m in moves if m[1] != prev_pos] or moves
    moves.sort(key=lambda m: (m[2] + visit_counts.get(m[1], 0) * 0.15,
                               ACTION_ORDER.index(m[0])))
    return moves[0][0]


def _llm_action(obs):
    if not _client:
        return None
    try:
        prompt = (
            f"Warehouse robot on {obs.grid_size}x{obs.grid_size} grid.\n"
            f"Position: row={obs.agent_position.row} col={obs.agent_position.col}\n"
            f"Items: {[(p.row, p.col) for p in obs.item_positions]}\n"
            f"Walls: {[(o.row, o.col) for o in obs.obstacles]}\n"
            'Reply ONLY with JSON: {"action":"up"|"down"|"left"|"right"}'
        )
        resp = _client.chat.completions.create(
            model=MODEL_NAME, max_tokens=32,
            messages=[
                {"role": "system", "content": "Navigation agent. JSON only."},
                {"role": "user",   "content": prompt},
            ],
        )
        text   = (resp.choices[0].message.content or "").strip()
        text   = text.replace("```json", "").replace("```", "")
        action = json.loads(text).get("action", "")
        if action in ACTION_DELTAS:
            return action
    except Exception:
        pass
    return None


# ── in-memory environment store (keyed by task_id) ───────────────────────────
_active_envs: dict[str, WarehouseBotEnv] = {}

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


# ── OpenEnv REST API ──────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Liveness probe."""
    return jsonify({"status": "ok"}), 200


@app.route("/tasks", methods=["GET"])
def openenv_tasks():
    """GET /tasks — list all available tasks."""
    out = [
        {
            "task_id":     t.task_id,
            "name":        t.name,
            "grid_size":   t.grid_size,
            "total_items": len(t.item_positions),
            "max_steps":   t.max_steps,
        }
        for t in list_tasks()
    ]
    return jsonify(out), 200


@app.route("/reset", methods=["POST"])
def openenv_reset():
    """POST /reset — reset environment and return initial observation."""
    body    = request.get_json(silent=True) or {}
    task_id = body.get("task_id", "easy")

    try:
        env = WarehouseBotEnv(task_id=task_id)
        obs = env.reset(task_id)
        _active_envs[task_id] = env
        return jsonify(obs.model_dump()), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/step", methods=["POST"])
def openenv_step():
    """POST /step — take one action and return RewardModel."""
    body    = request.get_json(silent=True) or {}
    task_id = body.get("task_id", "easy")
    action  = body.get("action")

    if action not in ACTION_DELTAS:
        return jsonify({"error": f"Invalid action: {action!r}. "
                        "Must be one of: up, down, left, right"}), 400

    env = _active_envs.get(task_id)
    if env is None:
        env = WarehouseBotEnv(task_id=task_id)
        env.reset(task_id)
        _active_envs[task_id] = env

    result = env.step(action)
    return jsonify(result.model_dump()), 200


@app.route("/state", methods=["GET"])
def openenv_state():
    """GET /state — return current observation without side effects."""
    task_id = request.args.get("task_id", "easy")

    env = _active_envs.get(task_id)
    if env is None:
        env = WarehouseBotEnv(task_id=task_id)
        env.reset(task_id)
        _active_envs[task_id] = env

    return jsonify(env.state().model_dump()), 200


# ── SSE run stream (used by the web UI) ───────────────────────────────────────

def _run_all_tasks(emit):
    scores = {}
    for task_def in list_tasks():
        tid  = task_def.task_id
        task = get_task(tid)
        env  = WarehouseBotEnv(task_id=tid)
        obs  = env.reset(tid)

        remaining    = sorted((i.row, i.col) for i in obs.item_positions)
        target       = _nearest((obs.agent_position.row, obs.agent_position.col), remaining)
        prev_pos     = None
        visit_counts = {(obs.agent_position.row, obs.agent_position.col): 1}
        rewards      = []
        step_n       = 0

        emit({"type": "start", "task": tid, "model": MODEL_NAME,
              "grid_size": task.grid_size,
              "items":     [[i.row, i.col] for i in task.item_positions],
              "obstacles": [[o.row, o.col] for o in task.obstacles],
              "start":     [task.start_position.row, task.start_position.col]})

        while not obs.done and step_n < MAX_STEPS:
            cur_pos = (obs.agent_position.row, obs.agent_position.col)
            action  = (_llm_action(obs) if _client else None) or \
                      _planner_action(obs, target, visit_counts, prev_pos)

            result  = env.step(action)
            obs     = result.observation
            step_n += 1
            rewards.append(result.reward)

            nxt_pos  = (obs.agent_position.row, obs.agent_position.col)
            prev_pos = cur_pos
            visit_counts[nxt_pos] = visit_counts.get(nxt_pos, 0) + 1

            if result.info.item_collected:
                remaining = sorted((i.row, i.col) for i in obs.item_positions)
                if remaining:
                    target = _nearest(nxt_pos, remaining)

            emit({"type": "step", "task": tid, "step": step_n, "action": action,
                  "reward": result.reward, "done": result.done,
                  "invalid": result.info.invalid_move,
                  "collected": result.info.item_collected,
                  "agent":    [obs.agent_position.row, obs.agent_position.col],
                  "items":    [[i.row, i.col] for i in obs.item_positions],
                  "obstacles":[[o.row, o.col] for o in obs.obstacles],
                  "picked":   [[p.row, p.col] for p in obs.picked_items]})

            if result.done:
                break

        score   = grade_episode(task_id=tid, actual_steps=obs.step_count,
                                items_collected=len(obs.picked_items))
        success = len(obs.picked_items) == len(task.item_positions)
        scores[tid] = score
        emit({"type": "end", "task": tid, "success": success,
              "steps": step_n, "score": score, "rewards": rewards})

    emit({"type": "done", "scores": scores,
          "overall": sum(scores.values()) / len(scores) if scores else 0})
    return scores


@app.route("/run")
def run_stream():
    q: queue.Queue = queue.Queue()

    def emit(event):
        q.put(event)

    def worker():
        try:
            _run_all_tasks(emit)
        except Exception as exc:
            q.put({"type": "done", "scores": {}, "overall": 0, "error": str(exc)})
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def generate() -> Generator:
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    port = int(os.getenv("PORT", 7860))
    print(f"===== PickPath-Bench server starting on port {port} =====", flush=True)
    app.run(host="0.0.0.0", port=port, threaded=True)

@app.route("/")
def home():
    return {
        "message": "PickPath Bench running",
        "status": "ok"
    }

if __name__ == "__main__":
    main()

