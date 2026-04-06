"""
app.py — Warehouse Bot Env  |  Web UI + inference runner
=========================================================
Serves a live dashboard on port 7860.
  GET  /           → dashboard HTML
  GET  /run        → SSE stream of inference logs
  GET  /results    → JSON of last run results
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
from typing import Generator

from flask import Flask, Response, jsonify, render_template_string

sys.path.insert(0, os.path.dirname(__file__))
from env.env import WarehouseBotEnv
from env.graders import grade_episode
from env.models import ObservationModel
from env.tasks import get_task, list_tasks

try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
BENCHMARK    = "warehouse-bot-env"
MAX_STEPS    = 120

_client = None
if HF_TOKEN and _openai_available:
    _client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ── planner ───────────────────────────────────────────────────────────────────
ACTION_DELTAS = {"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
ACTION_ORDER  = ("right","down","left","up")

def _manhattan(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])
def _nearest(pos,items): return min(items,key=lambda p:(_manhattan(pos,p),p))

def _planner_action(obs, target, visit_counts, prev_pos):
    cur   = (obs.agent_position.row, obs.agent_position.col)
    walls = {(o.row,o.col) for o in obs.obstacles}
    moves = []
    for act in ACTION_ORDER:
        dr,dc = ACTION_DELTAS[act]
        nxt = (cur[0]+dr, cur[1]+dc)
        if not (0<=nxt[0]<obs.grid_size and 0<=nxt[1]<obs.grid_size): continue
        if nxt in walls: continue
        moves.append((act, nxt, _manhattan(nxt,target)))
    if not moves: return "up"
    moves = [m for m in moves if m[1]!=prev_pos] or moves
    moves.sort(key=lambda m:(m[2]+visit_counts.get(m[1],0)*0.15, ACTION_ORDER.index(m[0])))
    return moves[0][0]

def _llm_action(obs):
    if not _client: return None
    try:
        prompt = (
            f"Warehouse robot on {obs.grid_size}x{obs.grid_size} grid.\n"
            f"Position: row={obs.agent_position.row} col={obs.agent_position.col}\n"
            f"Items: {[(p.row,p.col) for p in obs.item_positions]}\n"
            f"Walls: {[(o.row,o.col) for o in obs.obstacles]}\n"
            'Reply ONLY with JSON: {"action":"up"|"down"|"left"|"right"}'
        )
        resp = _client.chat.completions.create(
            model=MODEL_NAME, max_tokens=32,
            messages=[
                {"role":"system","content":"Navigation agent. JSON only."},
                {"role":"user","content":prompt},
            ],
        )
        text   = (resp.choices[0].message.content or "").strip().replace("```json","").replace("```","")
        action = json.loads(text).get("action","")
        if action in ACTION_DELTAS: return action
    except Exception:
        pass
    return None

# ── runner ────────────────────────────────────────────────────────────────────

def run_all_tasks(emit) -> dict:
    """Run all tasks, calling emit(event_dict) for each step."""
    scores = {}
    for task_def in list_tasks():
        tid  = task_def.task_id
        task = get_task(tid)
        env  = WarehouseBotEnv(task_id=tid)
        obs  = env.reset(tid)

        remaining    = sorted((i.row,i.col) for i in obs.item_positions)
        target       = _nearest((obs.agent_position.row,obs.agent_position.col), remaining)
        prev_pos     = None
        visit_counts = {(obs.agent_position.row,obs.agent_position.col):1}
        rewards      = []
        step_n       = 0

        emit({"type":"start","task":tid,"model":MODEL_NAME,
              "grid_size":task.grid_size,
              "items":[[i.row,i.col] for i in task.item_positions],
              "obstacles":[[o.row,o.col] for o in task.obstacles],
              "start":[task.start_position.row,task.start_position.col]})

        while not obs.done and step_n < MAX_STEPS:
            cur_pos = (obs.agent_position.row,obs.agent_position.col)
            action  = (_llm_action(obs) if _client else None) or _planner_action(obs,target,visit_counts,prev_pos)

            result = env.step(action)
            obs    = result.observation
            step_n += 1
            rewards.append(result.reward)

            nxt_pos  = (obs.agent_position.row,obs.agent_position.col)
            prev_pos = cur_pos
            visit_counts[nxt_pos] = visit_counts.get(nxt_pos,0)+1

            if result.info.item_collected:
                remaining = sorted((i.row,i.col) for i in obs.item_positions)
                if remaining: target = _nearest(nxt_pos,remaining)

            emit({"type":"step","task":tid,"step":step_n,"action":action,
                  "reward":result.reward,"done":result.done,
                  "invalid":result.info.invalid_move,
                  "collected":result.info.item_collected,
                  "agent":[obs.agent_position.row,obs.agent_position.col],
                  "items":[[i.row,i.col] for i in obs.item_positions],
                  "obstacles":[[o.row,o.col] for o in obs.obstacles],
                  "picked":[[p.row,p.col] for p in obs.picked_items]})

            if result.done: break

        score   = grade_episode(task_id=tid,actual_steps=obs.step_count,items_collected=len(obs.picked_items))
        success = len(obs.picked_items)==len(task.item_positions)
        scores[tid] = score
        emit({"type":"end","task":tid,"success":success,"steps":step_n,
              "score":score,"rewards":rewards})

    emit({"type":"done","scores":scores,
          "overall":sum(scores.values())/len(scores) if scores else 0})
    return scores

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
_last_results = {}

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Warehouse Bot Env</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0a0c10;--surface:#111318;--border:#1e2128;
  --accent:#00e5a0;--accent2:#0095ff;--warn:#ff6b35;
  --text:#e8eaf0;--muted:#4a5060;--dim:#1a1d24;
}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;min-height:100vh;overflow-x:hidden}

header{padding:2rem 2.5rem 1.5rem;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.logo{font-family:'Space Mono',monospace;font-size:1.1rem;color:var(--accent);letter-spacing:-.02em}
.logo span{color:var(--muted)}
.badge{font-family:'Space Mono',monospace;font-size:11px;padding:4px 10px;border:1px solid var(--accent);border-radius:2px;color:var(--accent)}

.main{display:grid;grid-template-columns:1fr 340px;gap:0;height:calc(100vh - 73px)}

/* left panel */
.left{padding:2rem 2.5rem;overflow-y:auto;border-right:1px solid var(--border)}
.task-tabs{display:flex;gap:0;margin-bottom:2rem;border:1px solid var(--border);border-radius:4px;overflow:hidden}
.task-tab{flex:1;padding:10px;font-family:'Space Mono',monospace;font-size:12px;text-align:center;cursor:pointer;background:var(--surface);color:var(--muted);border:none;transition:all .2s}
.task-tab.active{background:var(--dim);color:var(--accent);border-bottom:2px solid var(--accent)}

.grid-wrap{display:flex;justify-content:center;margin-bottom:2rem}
canvas{border:1px solid var(--border);border-radius:4px;image-rendering:pixelated}

.stats{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:2rem}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:14px}
.stat-label{font-family:'Space Mono',monospace;font-size:10px;color:var(--muted);margin-bottom:6px;text-transform:uppercase}
.stat-value{font-family:'Space Mono',monospace;font-size:1.4rem;color:var(--text);font-weight:700}
.stat-value.green{color:var(--accent)}
.stat-value.blue{color:var(--accent2)}
.stat-value.orange{color:var(--warn)}

.run-btn{width:100%;padding:14px;background:var(--accent);color:#000;font-family:'Space Mono',monospace;font-size:13px;font-weight:700;border:none;border-radius:4px;cursor:pointer;letter-spacing:.05em;transition:opacity .2s}
.run-btn:disabled{opacity:.4;cursor:not-allowed}
.run-btn:hover:not(:disabled){opacity:.85}

/* right panel — log */
.right{display:flex;flex-direction:column}
.log-header{padding:1.25rem 1.5rem;border-bottom:1px solid var(--border);font-family:'Space Mono',monospace;font-size:11px;color:var(--muted);display:flex;justify-content:space-between;align-items:center}
.log-clear{background:none;border:1px solid var(--border);color:var(--muted);font-family:'Space Mono',monospace;font-size:10px;padding:3px 8px;border-radius:2px;cursor:pointer}
.log-clear:hover{color:var(--text);border-color:var(--muted)}
.log-body{flex:1;overflow-y:auto;padding:1rem 1.5rem;font-family:'Space Mono',monospace;font-size:11px;line-height:1.8}
.log-line{margin-bottom:2px;white-space:pre-wrap;word-break:break-all}
.log-start{color:var(--accent2)}
.log-step{color:var(--muted)}
.log-step.reward-good{color:var(--accent)}
.log-step.reward-bad{color:var(--warn)}
.log-end{color:var(--accent)}
.log-done{color:var(--text);border-top:1px solid var(--border);margin-top:8px;padding-top:8px}

/* score board */
.scoreboard{border-top:1px solid var(--border);padding:1.25rem 1.5rem}
.score-title{font-family:'Space Mono',monospace;font-size:10px;color:var(--muted);margin-bottom:10px;text-transform:uppercase}
.score-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.score-name{font-family:'Space Mono',monospace;font-size:11px;color:var(--muted)}
.score-bar-wrap{flex:1;margin:0 12px;height:4px;background:var(--border);border-radius:2px;overflow:hidden}
.score-bar{height:100%;background:var(--accent);border-radius:2px;transition:width .6s ease;width:0%}
.score-num{font-family:'Space Mono',monospace;font-size:11px;color:var(--text);min-width:40px;text-align:right}

/* legend */
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:1.5rem;font-size:12px;color:var(--muted)}
.legend-item{display:flex;align-items:center;gap:6px}
.legend-dot{width:12px;height:12px;border-radius:2px}

@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.running .run-btn{animation:pulse 1.5s ease-in-out infinite}
</style>
</head>
<body>
<header>
  <div class="logo">warehouse<span>_</span>bot<span>.</span>env</div>
  <div class="badge">openenv</div>
</header>

<div class="main">
  <div class="left" id="left">
    <div class="task-tabs">
      <button class="task-tab active" onclick="switchTask('easy')">easy</button>
      <button class="task-tab" onclick="switchTask('medium')">medium</button>
      <button class="task-tab" onclick="switchTask('hard')">hard</button>
    </div>

    <div class="legend">
      <div class="legend-item"><div class="legend-dot" style="background:#0095ff"></div> agent</div>
      <div class="legend-item"><div class="legend-dot" style="background:#00e5a0"></div> item</div>
      <div class="legend-item"><div class="legend-dot" style="background:#333840"></div> wall</div>
      <div class="legend-item"><div class="legend-dot" style="background:#1a2a1a"></div> collected</div>
    </div>

    <div class="grid-wrap"><canvas id="grid" width="420" height="420"></canvas></div>

    <div class="stats">
      <div class="stat"><div class="stat-label">Step</div><div class="stat-value blue" id="s-step">—</div></div>
      <div class="stat"><div class="stat-label">Reward</div><div class="stat-value" id="s-reward">—</div></div>
      <div class="stat"><div class="stat-label">Score</div><div class="stat-value green" id="s-score">—</div></div>
    </div>

    <button class="run-btn" id="run-btn" onclick="startRun()">▶  RUN ALL TASKS</button>
  </div>

  <div class="right">
    <div class="log-header">
      <span>stdout log</span>
      <button class="log-clear" onclick="clearLog()">clear</button>
    </div>
    <div class="log-body" id="log"></div>
    <div class="scoreboard">
      <div class="score-title">Results</div>
      <div class="score-row"><span class="score-name">easy</span><div class="score-bar-wrap"><div class="score-bar" id="bar-easy"></div></div><span class="score-num" id="num-easy">—</span></div>
      <div class="score-row"><span class="score-name">medium</span><div class="score-bar-wrap"><div class="score-bar" id="bar-medium"></div></div><span class="score-num" id="num-medium">—</span></div>
      <div class="score-row"><span class="score-name">hard</span><div class="score-bar-wrap"><div class="score-bar" id="bar-hard"></div></div><span class="score-num" id="num-hard">—</span></div>
      <div class="score-row" style="margin-top:8px;border-top:1px solid var(--border);padding-top:8px">
        <span class="score-name" style="color:var(--text)">overall</span>
        <div class="score-bar-wrap"><div class="score-bar" id="bar-overall" style="background:var(--accent2)"></div></div>
        <span class="score-num" id="num-overall" style="color:var(--accent2)">—</span>
      </div>
    </div>
  </div>
</div>

<script>
const CELL = 60;
const canvas = document.getElementById('grid');
const ctx    = canvas.getContext('2d');

// state per task
const tasks = {
  easy:   {grid_size:5,agent:[0,0],items:[],obstacles:[],picked:[],step:0,reward:0,score:null},
  medium: {grid_size:6,agent:[0,0],items:[],obstacles:[],picked:[],step:0,reward:0,score:null},
  hard:   {grid_size:7,agent:[0,0],items:[],obstacles:[],picked:[],step:0,reward:0,score:null},
};
let activeTask = 'easy';
let running    = false;

function switchTask(tid) {
  activeTask = tid;
  document.querySelectorAll('.task-tab').forEach((t,i)=>{
    t.classList.toggle('active', ['easy','medium','hard'][i]===tid);
  });
  drawGrid();
  updateStats();
}

function resizeCanvas(size) {
  const dim = size * CELL;
  canvas.width  = dim;
  canvas.height = dim;
}

function drawGrid() {
  const t    = tasks[activeTask];
  const size = t.grid_size;
  resizeCanvas(size);
  ctx.clearRect(0,0,canvas.width,canvas.height);

  // background cells
  for(let r=0;r<size;r++) for(let c=0;c<size;c++) {
    ctx.fillStyle = '#111318';
    ctx.fillRect(c*CELL+1, r*CELL+1, CELL-2, CELL-2);
  }

  // grid lines
  ctx.strokeStyle = '#1e2128';
  ctx.lineWidth   = 1;
  for(let i=0;i<=size;i++) {
    ctx.beginPath(); ctx.moveTo(i*CELL,0); ctx.lineTo(i*CELL,size*CELL); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0,i*CELL); ctx.lineTo(size*CELL,i*CELL); ctx.stroke();
  }

  // obstacles
  t.obstacles.forEach(([r,c])=>{
    ctx.fillStyle='#242830';
    ctx.fillRect(c*CELL+1,r*CELL+1,CELL-2,CELL-2);
    ctx.fillStyle='#1a1d24';
    ctx.fillRect(c*CELL+4,r*CELL+4,CELL-8,CELL-8);
  });

  // picked items (faded)
  t.picked.forEach(([r,c])=>{
    ctx.fillStyle='#0a1a0a';
    ctx.fillRect(c*CELL+1,r*CELL+1,CELL-2,CELL-2);
    ctx.strokeStyle='#1a3a1a';
    ctx.lineWidth=1;
    ctx.strokeRect(c*CELL+8,r*CELL+8,CELL-16,CELL-16);
  });

  // items
  t.items.forEach(([r,c])=>{
    const x=c*CELL+CELL/2, y=r*CELL+CELL/2;
    ctx.fillStyle='#00e5a015';
    ctx.fillRect(c*CELL+1,r*CELL+1,CELL-2,CELL-2);
    ctx.fillStyle='#00e5a0';
    ctx.beginPath();
    ctx.roundRect(c*CELL+14,r*CELL+14,CELL-28,CELL-28,4);
    ctx.fill();
    ctx.fillStyle='#000';
    ctx.font='bold 14px Space Mono,monospace';
    ctx.textAlign='center';
    ctx.textBaseline='middle';
    ctx.fillText('▪',x,y);
  });

  // agent
  const [ar,ac] = t.agent;
  const ax=ac*CELL+CELL/2, ay=ar*CELL+CELL/2;
  ctx.fillStyle='#0095ff22';
  ctx.fillRect(ac*CELL+1,ar*CELL+1,CELL-2,CELL-2);
  ctx.fillStyle='#0095ff';
  ctx.beginPath();
  ctx.arc(ax,ay,CELL/2-10,0,Math.PI*2);
  ctx.fill();
  ctx.fillStyle='#fff';
  ctx.font='bold 16px sans-serif';
  ctx.textAlign='center';
  ctx.textBaseline='middle';
  ctx.fillText('◆',ax,ay);
}

function updateStats() {
  const t = tasks[activeTask];
  document.getElementById('s-step').textContent   = t.step || '—';
  const rv = t.reward;
  const el = document.getElementById('s-reward');
  el.textContent = t.step ? rv.toFixed(2) : '—';
  el.style.color = rv>=49?'var(--accent)':rv<0?'var(--warn)':'var(--text)';
  document.getElementById('s-score').textContent  = t.score!=null ? t.score.toFixed(4) : '—';
}

function log(html, cls='') {
  const div = document.getElementById('log');
  div.innerHTML += `<div class="log-line ${cls}">${html}</div>`;
  div.scrollTop = div.scrollHeight;
}
function clearLog() { document.getElementById('log').innerHTML=''; }

function setScore(tid, score) {
  document.getElementById('bar-'+tid).style.width = (score*100)+'%';
  document.getElementById('num-'+tid).textContent = score.toFixed(4);
}

function startRun() {
  if(running) return;
  running = true;
  document.getElementById('run-btn').disabled = true;
  document.getElementById('left').classList.add('running');
  clearLog();

  const es = new EventSource('/run');
  es.onmessage = e => {
    const d = JSON.parse(e.data);

    if(d.type==='start') {
      tasks[d.task].grid_size  = d.grid_size;
      tasks[d.task].agent      = d.start;
      tasks[d.task].items      = d.items.map(x=>[...x]);
      tasks[d.task].obstacles  = d.obstacles.map(x=>[...x]);
      tasks[d.task].picked     = [];
      tasks[d.task].step       = 0;
      tasks[d.task].score      = null;
      switchTask(d.task);
      log(`[START] task=${d.task} model=${d.model}`,'log-start');
    }

    if(d.type==='step') {
      const t = tasks[d.task];
      t.agent   = d.agent;
      t.items   = d.items;
      t.picked  = d.picked;
      t.obstacles = d.obstacles;
      t.step    = d.step;
      t.reward  = d.reward;
      if(d.task===activeTask){ drawGrid(); updateStats(); }
      const cls = d.reward>=49?'log-step reward-good':d.reward<-1?'log-step reward-bad':'log-step';
      const err = d.invalid?'invalid_move':'null';
      log(`[STEP] step=${d.step} action=${d.action} reward=${d.reward.toFixed(2)} done=${d.done} error=${err}`,cls);
    }

    if(d.type==='end') {
      tasks[d.task].score = d.score;
      if(d.task===activeTask) updateStats();
      setScore(d.task, d.score);
      log(`[END] success=${d.success} steps=${d.steps} score=${d.score.toFixed(2)}`,'log-end');
    }

    if(d.type==='done') {
      const ov = d.overall;
      document.getElementById('bar-overall').style.width=(ov*100)+'%';
      document.getElementById('num-overall').textContent=ov.toFixed(4);
      log(`\n=== Overall: ${ov.toFixed(4)} ===`,'log-done');
      es.close();
      running=false;
      document.getElementById('run-btn').disabled=false;
      document.getElementById('left').classList.remove('running');
    }
  };
  es.onerror = () => {
    es.close();
    running=false;
    document.getElementById('run-btn').disabled=false;
    document.getElementById('left').classList.remove('running');
  };
}

// init draw
drawGrid();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/run")
def run_stream():
    q: queue.Queue = queue.Queue()

    def emit(event):
        q.put(event)

    def worker():
        try:
            run_all_tasks(emit)
        except Exception as e:
            q.put({"type": "done", "scores": {}, "overall": 0, "error": str(e)})
        finally:
            q.put(None)  # sentinel

    threading.Thread(target=worker, daemon=True).start()

    def generate() -> Generator:
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/results")
def results():
    return jsonify(_last_results)


if __name__ == "__main__":
    # Also print inference logs to stdout for HF Space log viewer
    import subprocess, sys
    print("===== Warehouse Bot Env UI starting on port 7860 =====", flush=True)
    app.run(host="0.0.0.0", port=7860, threaded=True)
