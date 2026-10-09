"""Write a RoboMME leaderboard submission file (doc/submission/<model>.md) from one or more finished run directories of
astra_waypoint_eval_v2.py, in the layout of RoboMME's doc/submission/model_example.md. Refuses to write if any
scheduled episode is unscored or missing.

    python3 make_submission_md.py --runs runs/e2e_v2_test --out submission/gpt-6-astra.md
"""
import argparse
import json
from pathlib import Path

SUITES = [("Counting", ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"]),
          ("Permanence", ["VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap"]),
          ("Reference", ["PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"]),
          ("Imitation", ["MoveCube", "InsertPeg", "PatternLock", "RouteStick"])]
TASKS = [t for _, ts in SUITES for t in ts]


def load_run(run):
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["dataset"] == "test" and cfg["action_space"] == "waypoint" and cfg["max_steps"] == 1300, cfg
    rates, models, dates, missing = {}, set(), [], []
    for t in TASKS:
        rs = []
        for e in range(50):
            p = run / f"{t}_ep{e}" / "result.json"
            r = json.loads(p.read_text()) if p.exists() else None
            if r is None or r.get("success") is None:
                missing.append(f"{t}_ep{e}")
                continue
            rs.append(r)
            models.add(r.get("api_model"))
            dates += [r.get("started_utc"), r.get("finished_utc")]
        rates[t] = 100.0 * sum(bool(r["success"]) for r in rs) / 50
    if missing:
        raise SystemExit(f"{run}: {len(missing)} episodes missing or unscored, e.g. {missing[:5]}")
    dates = sorted(d for d in dates if d)
    return cfg, rates, models, (dates[0][:10], dates[-1][:10])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="one run directory per run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="GPT-6-Astra (end-to-end, waypoint)")
    a = ap.parse_args()
    runs = [load_run(r) for r in a.runs]
    cfg = runs[0][0]
    n = len(runs)
    head = "".join(f"<th>Run {i + 1}</th>" for i in range(n)) + "<th><b>Avg</b></th>"
    rows = []
    for suite, ts in SUITES:
        for k, t in enumerate(ts):
            vals = [r[1][t] for r in runs]
            cells = "".join(f"<td>{v:.1f}</td>" for v in vals) + f"<td><b>{sum(vals) / n:.1f}</b></td>"
            lead = f'<td rowspan="4">{suite}</td>\n  ' if k == 0 else ""
            rows.append(f"<tr>\n  {lead}<td>{t}</td>{cells}\n</tr>" if k == 0 else f"<tr><td>{t}</td>{cells}</tr>")
    overall = [sum(r[1][t] for t in TASKS) / len(TASKS) for r in runs]
    rows.append('<tr>\n  <td colspan="2"><b>Overall</b></td>' + "".join(f"<td><b>{v:.1f}</b></td>" for v in overall)
                + f"<td><b>{sum(overall) / n:.1f}</b></td>\n</tr>")
    models = sorted({m for r in runs for m in r[2] if m})
    dates = (min(r[3][0] for r in runs), max(r[3][1] for r in runs))
    runs_note = ("We report a single run (the leaderboard page recommends multiple runs; the submission template asks "
                 "for three)." if n == 1 else f"We report {n} independent runs; the benchmark seed is fixed internally, "
                 "so run-to-run variance comes from the model's sampling.")
    md = f"""# {a.name}

### [Website]() | [Paper]() | [Code]()

## Introduction
A general-purpose vision-language model (OpenAI `{', '.join(models)}`, accessed through the API) used directly as the
policy: at every step it receives the task goal, the camera frames and the robot's own state, and outputs the next
end-effector waypoint (position, orientation) and gripper command. RoboMME's motion planner executes each waypoint.
Memory is the model's own context: the whole episode (demonstration video, all execution frames and its previous
answers) stays in one conversation. No training or fine-tuning on RoboMME.

## Results

> Multi-task setting with a single model for all 16 tasks, on the **test** split, **`waypoint`** action space,
> `max_steps={cfg['max_steps']}`, 50 episodes per task, evaluated with the loop of RoboMME's `scripts/evaluation.py`.
> {runs_note} The benchmark seed is fixed internally. Model version as reported by the API: `{', '.join(models)}`;
> evaluation dates {dates[0]} to {dates[1]} (UTC).

### Table

<table>
<tr>
  <th rowspan="2">Suite</th>
  <th rowspan="2">Task</th>
</tr>
<tr>
  {head}
</tr>
{chr(10).join(rows)}
</table>

### Evaluation Details
- **Inputs per step** (identical for all tasks): task goal; front-camera frames of the demonstration/history video and of
  every executed motion, one in every {cfg['frame_stride']} control steps plus the last frame; the current wrist-camera frame
  (256 x 256 RGB); TCP pose, finger positions and elapsed control steps; front/wrist camera intrinsics and extrinsics
  (RoboMME's documented optional observations `include_*_camera_*`). No depth, no simulator state, no available
  multi-choice options, no oracle subgoals.
- **Prompt**: one system prompt for all tasks (released with the code), describing the waypoint action format, the
  world frame (as in RoboMME's `env_format.md`) and general facts about the robot interface (timing of motions, gripper
  behaviour). The only task-dependent sentence states the hardware of PatternLock and RouteStick (the robot holds a
  stick, so the gripper command is ignored).
- **Episode endings besides the environment's** (all counted as failures): the model declares the task complete before
  the environment judges success; {cfg['max_consecutive_noops']} consecutive actions that execute no steps; an unusable answer;
  {cfg['max_model_calls']} model calls (in practice the 1300-step limit binds first). Environment errors (e.g. unreachable poses)
  count as failures, as in `evaluation.py`.
- No training data from RoboMME is used.

### Released Checkpoints
The model is a closed API model; no checkpoint can be released. Evaluation code, prompts, configuration and all
per-episode logs are released at [Code]().

### Citations
```
```
"""
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(md)
    print(f"wrote {a.out}: overall {', '.join(f'{v:.2f}' for v in overall)}")


if __name__ == "__main__":
    main()
