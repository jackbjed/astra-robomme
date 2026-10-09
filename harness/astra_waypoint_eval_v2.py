"""gpt-6-astra on RoboMME, end-to-end with the `waypoint` action space (v2, for a leaderboard submission).

Astra chooses every end-effector pose and gripper command; RoboMME's motion planner only moves the TCP to the commanded
pose (screw, then RRT* fallback). No oracle skill, no target snapping, no privileged state.

Benchmark protocol, as RoboMME's scripts/evaluation.py (copied verbatim as evaluation_official.py; identical to GitHub
main): BenchmarkEnvBuilder(dataset, action_space="waypoint", max_steps=1300), episodes range(get_episode_num()) (50 per
task on test), task goal info["task_goal"][0], step until terminated/truncated, success iff info["status"] == "success",
env errors (FailAwareWrapper status "error", e.g. a planning failure) count as failures.

Model inputs (identical for all 16 tasks): the task goal; front-camera frames of the history and of every motion, one
in every FRAME_STRIDE control steps plus the last frame, and the current wrist frame (256 x 256 RGB); the robot's own
state (TCP pose, finger positions, elapsed control steps); the camera calibration RoboMME exposes as documented optional
observations (include_{front,wrist}_camera_{intrinsic,extrinsic}). Never used: depth, maniskill_obs, available
multi-choice options, info's *_subgoal_online fields. World-frame facts in the prompt (robot base at (-0.615, 0, 0),
axes) are from RoboMME's doc/env_format.md. The system prompt is this harness's own (RoboMME ships none for waypoint);
it includes MODEL_NOTES, general facts about the robot and this interface (timing, gripper, empty actions) that are
the same for every task. The only task-dependent sentence states the robot's hardware: in PatternLock and RouteStick
the robot holds a stick, so the gripper command is ignored.

Episode endings beyond the official loop (each can only lower the score, all scored as failures and logged as
end_reason): the model answers "waypoint": null (declares the task complete) before the env judged success - the
official loop has no way for a policy to stop, and a "stay" waypoint crashes RoboMME's planner (a zero-displacement
waypoint raises inside its RRT* fallback); MAX_CONSECUTIVE_NOOPS actions in a row that execute no steps; an unparseable
or malformed answer; MAX_MODEL_CALLS answers (set high, so that in practice the env's 1300-step limit binds first).

Motion timings in MODEL_NOTES were measured on the train split (scripted random waypoints, 2026-10-08), not on test.

v2 changes vs astra_waypoint_eval.py (pilot, runs/waypoint_pilot_2ep): self-contained (no import of the oracle-planner
harness); --frame-stride (default 3); MODEL_NOTES in the system prompt; an explicit message after an action that
executed no steps; MAX_MODEL_CALLS 40 -> 150; result.json records the API-reported model version and UTC start/end;
--dataset (test / val); summary + leaderboard-format table (make_submission_md.py).

    export OPENAI_API_KEY=...
    CUDA_VISIBLE_DEVICES=1 ~/micromamba/envs/robomme/bin/python astra_waypoint_eval_v2.py --run-dir runs/e2e_v2_test --episodes $(seq 0 49)
    ~/micromamba/envs/robomme/bin/python astra_waypoint_eval_v2.py --run-dir runs/e2e_v2_test --episodes $(seq 0 49) --summarize
"""
import argparse
import base64
import datetime
import hashlib
import json
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import openai
import torch
from openai import OpenAI

from robomme.env_record_wrapper import BenchmarkEnvBuilder

HARNESS_VERSION = "waypoint_v2"
MAX_STEPS, MAX_MODEL_CALLS, MAX_CONSECUTIVE_NOOPS = 1300, 150, 3
CONTROL_HZ = 20
FRAME_STRIDE = 3
STICK_TASKS = ("PatternLock", "RouteStick")
PRICE = {"input": 10.0, "cached": 1.0, "cache_write": 12.5, "output": 50.0}     # $ per 1M tokens, gpt-6-astra
RETRYABLE = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError)

SYSTEM = """You control a simulated Franka Panda robot arm on a table, by choosing end-effector waypoints.
World frame, metres: the robot's base is at (-0.615, 0, 0) and faces +x; +y is the robot's left, +z is up.

Each turn, answer with ONE JSON object:
{"description": "what happened in the latest frames and what you will do next",
 "waypoint": [x, y, z, roll, pitch, yaw],
 "gripper": -1, 0 or 1}
The robot moves its TCP (the point between the fingertips) to (x, y, z) with orientation (roll, pitch, yaw) in
radians, using the same convention as the TCP pose reported to you, along a planned path; then it applies the
gripper command: -1 = close, 1 = open, 0 = keep. Pointing straight down is roll = -3.1416, pitch = 0; yaw turns the
hand about the vertical axis. If a pose cannot be reached the episode fails. A waypoint identical to the current
TCP pose also fails: to only open or close the gripper, move the TCP by at least 2 mm (for example 5 mm up or down).
STICK_NOTE
Observations: frames from the fixed front camera (256 x 256) in time order, one every FRAME_STRIDE control steps
(FRAME_DT s) of the video so far and of each motion, then the current front and wrist images; the TCP pose; the finger
positions (0.04 m each = fully open, 0 = closed); the elapsed control steps; and the camera calibration: pinhole
intrinsics K and world-to-camera extrinsics [R|t] (OpenCV convention: camera x right, y down, z forward;
pixel = K (R X_world + t), dividing by depth). You get no depth and no object positions. The task may require
remembering what happened earlier in the video.

MODEL_NOTES
The episode ends automatically when the environment judges success or failure, or after 1300 control steps.
If you believe the task is complete, answer with "waypoint": null."""

MODEL_NOTES = """Notes on the robot and this interface:
- Time: the simulation runs at 20 control steps per second, and time passes only while the robot executes a motion.
  A motion takes longer the farther it goes: about 0.15-0.3 s for a few millimetres, 0.4-1.1 s for 1-5 cm and
  1-2.5 s for 10-30 cm. Everything else in the scene keeps moving during a motion, so to act at a particular moment,
  start the motion that much earlier. A small motion (a few millimetres) is a short way to let a little time pass.
- Gripper: it opens or closes only when you command it (-1 or 1); with 0 it keeps its current state, also after
  touching, pushing or pressing something.
- If an action executes no steps, the next message says so explicitly: then the robot did not move, no time passed
  and the scene is unchanged.
"""

NOOP_TEXT = ("The last action executed no steps: the robot did not move and no time passed, so the scene is unchanged. "
             "Choose the next action.")


def system_prompt(task):
    note = ("In this task the robot holds a stick instead of a gripper: the gripper command is ignored.\n"
            if task in STICK_TASKS else "")
    return (SYSTEM.replace("STICK_NOTE\n", note).replace("MODEL_NOTES\n", MODEL_NOTES)
            .replace("FRAME_STRIDE", str(FRAME_STRIDE)).replace("FRAME_DT", f"{FRAME_STRIDE / CONTROL_HZ:g}"))


def to_np(t):
    return (t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)).astype(np.uint8)


def image_item(rgb):
    png = cv2.imencode(".png", cv2.cvtColor(to_np(rgb), cv2.COLOR_RGB2BGR))[1].tobytes()
    return {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(png).decode()}


def text_item(text):
    return {"type": "input_text", "text": text}


def frame_indices(n):
    """One frame in every FRAME_STRIDE, plus the last (current) frame."""
    return sorted(set(range(0, n, FRAME_STRIDE)) | {n - 1})


def parse_answer(text):
    """The JSON object asked for, with or without ```json fences; None if there is none."""
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1].removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start:end + 1])
    except (ValueError, json.JSONDecodeError):
        return None


class VideoRecorder:
    """As in evaluation_official.py: history frames with a red border, then every step frame (front | wrist)."""
    def __init__(self, fps=30):
        self.fps, self.frames = fps, []

    def add(self, obs, history=False):
        n = len(obs["front_rgb_list"])
        for i, (f, w) in enumerate(zip(obs["front_rgb_list"], obs["wrist_rgb_list"])):
            frame = np.hstack([to_np(f), to_np(w)])
            if history and i < n - 1:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 10)
            self.frames.append(frame)

    def save(self, path):
        if self.frames:
            imageio.mimsave(path, self.frames, fps=self.fps)


def fmt(a, nd=4):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def state_text(obs, info, first=False):
    eef = np.asarray(obs["eef_state_list"][-1]).reshape(-1)
    grip = np.asarray(obs["gripper_state_list"][-1]).reshape(-1)
    s = {"tcp_pose_xyz_rpy": fmt(eef), "finger_positions_m": fmt(grip),
         "elapsed_control_steps": int(np.asarray(info.get("elapsed_steps", 0)).reshape(-1)[-1]),
         "wrist_camera_K": fmt(np.asarray(info["wrist_camera_intrinsic"]).reshape(3, 3), 2),
         "wrist_camera_extrinsic_world_to_camera": fmt(np.asarray(obs["wrist_camera_extrinsic_list"][-1]).reshape(3, 4))}
    if first:
        s["front_camera_K"] = fmt(np.asarray(info["front_camera_intrinsic"]).reshape(3, 3), 2)
        s["front_camera_extrinsic_world_to_camera"] = fmt(np.asarray(obs["front_camera_extrinsic_list"][-1]).reshape(3, 4))
    return text_item("Robot state and calibration: " + json.dumps(s))


def observation(obs, info, label, first=False):
    fr, wr = obs["front_rgb_list"], obs["wrist_rgb_list"]
    idx = frame_indices(len(fr))
    out = [text_item(f"{label}: front-camera frames {idx} of {len(fr)} (last = current):")]
    out += [image_item(fr[i]) for i in idx]
    out += [text_item("Current wrist camera:"), image_item(wr[-1]), state_text(obs, info, first)]
    return out


def to_action(answer):
    """(7-vector, None) or (None, kind): declared_complete / unparseable / invalid_action."""
    if not isinstance(answer, dict):
        return None, "unparseable"
    if "waypoint" in answer and answer["waypoint"] is None:
        return None, "declared_complete"
    wp, g = answer.get("waypoint"), answer.get("gripper", 0)
    if not (isinstance(wp, list) and len(wp) == 6 and all(isinstance(v, (int, float)) for v in wp)) \
            or g not in (-1, 0, 1):
        return None, "invalid_action"
    return np.array([*wp, g], dtype=np.float64), None


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def run_episode(client, model, task, episode, out_dir, dataset):
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_dir = out_dir / "observations"
    obs_dir.mkdir(exist_ok=True)
    started = utc_now()
    env = BenchmarkEnvBuilder(env_id=task, dataset=dataset, action_space="waypoint", gui_render=False,
                              max_steps=MAX_STEPS).make_env_for_episode(
        episode, include_front_camera_extrinsic=True, include_front_camera_intrinsic=True,
        include_wrist_camera_extrinsic=True, include_wrist_camera_intrinsic=True)
    obs, info = env.reset()
    goal = info["task_goal"][0]
    rec = VideoRecorder()
    rec.add(obs, history=True)
    log = open(out_dir / "exchange.jsonl", "w")
    usage = {"calls": 0, "api_s": 0.0, "retries": 0, "input": 0, "cached": 0, "cache_write": 0, "output": 0,
             "reasoning": 0, "api_model": None}
    request = dict(model=model, instructions=system_prompt(task))

    def ask(**kw):
        logged = []
        for it in kw["input"]:
            safe = []
            for p in it["content"]:
                if p["type"] == "input_image":
                    path = obs_dir / f"{len(list(obs_dir.iterdir())):04d}.png"
                    path.write_bytes(base64.b64decode(p["image_url"].split(",", 1)[1]))
                    safe.append({"type": "input_image", "image_url": f"<{path.relative_to(out_dir)}>"})
                else:
                    safe.append(p)
            logged.append({**it, "content": safe})
        log.write(json.dumps({"request": {"previous_response_id": kw.get("previous_response_id"), "input": logged}}) + "\n")
        for attempt in range(6):
            t = time.time()
            try:
                r = client.responses.create(**kw, **request)
                break
            except RETRYABLE:
                usage["retries"] += 1
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt * 5)
        usage["api_s"] += time.time() - t
        d = r.model_dump(mode="json")
        log.write(json.dumps({"response": d}) + "\n")
        log.flush()
        u = d["usage"]
        usage["calls"] += 1
        usage["input"] += u["input_tokens"]
        usage["cached"] += (u.get("input_tokens_details") or {}).get("cached_tokens") or 0
        usage["cache_write"] += (u.get("input_tokens_details") or {}).get("cache_write_tokens") or 0
        usage["output"] += u["output_tokens"]
        usage["reasoning"] += (u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0
        usage["api_model"] = d.get("model")
        return r

    t0 = time.time()
    first = [text_item(f"Task goal: {goal}")] + observation(obs, info, "Video so far", first=True)
    response = ask(input=[{"role": "user", "content": first}])
    status, end_reason, steps, noops = "ongoing", None, [], 0
    while True:
        answer = parse_answer(response.output_text)
        action, kind = to_action(answer)
        if action is None:
            steps.append({"answer": response.output_text[-400:], "invalid_kind": kind})
            status, end_reason = kind, f"model_answer:{kind}"
            break
        new_obs, rew, term, trunc, new_info = env.step(action)
        if new_obs is None or (new_info or {}).get("status") == "error":   # FailAwareWrapper: planning failure etc.
            steps.append({"action": action.tolist(), "error": (new_info or {}).get("error_message")})
            status, end_reason = "error", "env_error"
            break
        n = len(new_obs.get("front_rgb_list") or [])
        if n == 0:
            noops += 1
            steps.append({"action": action.tolist(), "frames": 0, "noop": True})
        else:
            noops = 0
            obs, info = new_obs, new_info
            rec.add(obs)
            st = info.get("status", "unknown")
            steps.append({"action": action.tolist(), "frames": n, "status": st,
                          "elapsed_steps": int(np.asarray(info.get("elapsed_steps", 0)).reshape(-1)[-1]),
                          "tcp_after": fmt(np.asarray(obs["eef_state_list"][-1]).reshape(-1))})
            if bool(term) or bool(trunc) or st in ("success", "fail", "timeout", "error"):
                status, end_reason = st, f"env:{st}"
                break
        if noops >= MAX_CONSECUTIVE_NOOPS:
            status, end_reason = "no_progress", f"limit:{MAX_CONSECUTIVE_NOOPS}_consecutive_noops"
            break
        if usage["calls"] >= MAX_MODEL_CALLS:
            status, end_reason = "model_call_limit", f"limit:{MAX_MODEL_CALLS}_model_calls"
            break
        follow = observation(obs, info, "Frames recorded during that motion") if n else \
            [text_item(NOOP_TEXT), text_item("Current front camera:"), image_item(obs["front_rgb_list"][-1]),
             text_item("Current wrist camera:"), image_item(obs["wrist_rgb_list"][-1]), state_text(obs, info)]
        response = ask(previous_response_id=response.id, input=[{"role": "user", "content": follow}])
    env.close()
    log.close()
    rec.save(out_dir / "episode.mp4")
    cost = (usage["cached"] * PRICE["cached"] + usage["cache_write"] * PRICE["cache_write"] + usage["output"] * PRICE["output"]
            + (usage["input"] - usage["cached"] - usage["cache_write"]) * PRICE["input"]) / 1e6
    row = {"task": task, "episode": episode, "dataset": dataset, "goal": goal, "status": status,
           "success": status == "success", "end_reason": end_reason, "harness_version": HARNESS_VERSION,
           "api_model": usage["api_model"], "started_utc": started, "finished_utc": utc_now(),
           "model_calls": usage["calls"], "waypoints_executed": sum(1 for s in steps if s.get("frames")),
           "sim_elapsed_steps": next((s["elapsed_steps"] for s in reversed(steps) if "elapsed_steps" in s), None),
           "steps": steps, "usage": usage, "cost_usd": round(cost, 4), "episode_s": round(time.time() - t0, 1)}
    (out_dir / "result.json").write_text(json.dumps(row, indent=2, default=str))
    return row


def summarize(run_dir, tasks, episodes):
    """Every scheduled episode is reported; rates are final only when nothing is unscored."""
    rows = []
    for t in tasks:
        for e in episodes:
            p = run_dir / f"{t}_ep{e}" / "result.json"
            r = json.loads(p.read_text()) if p.exists() else {"task": t, "episode": e, "status": "not_run", "success": None}
            rows.append(r)
    unscored = [f"{r['task']}_ep{r['episode']}:{r['status']}" for r in rows if r.get("success") is None]
    if unscored:
        print(f"WARNING: {len(unscored)} of {len(rows)} scheduled episodes are unscored; NOT final: {unscored[:20]}")
    print(f"{'task':18s} {'success/scheduled':>17s} {'cost':>8s}")
    per = {}
    for t in tasks:
        rs = [r for r in rows if r["task"] == t]
        per[t] = {"success": sum(bool(r.get("success")) for r in rs), "scheduled": len(rs),
                  "unscored": sum(r.get("success") is None for r in rs)}
        print(f"{t:18s} {per[t]['success']:>8d}/{len(rs):<8d} ${sum(r.get('cost_usd') or 0 for r in rs):>7.2f}")
    print(f"{'TOTAL':18s} {sum(bool(r.get('success')) for r in rows):>8d}/{len(rows):<8d} "
          f"${sum(r.get('cost_usd') or 0 for r in rows):>7.2f}")
    (run_dir / "summary.json").write_text(json.dumps({"scheduled": len(rows), "unscored": unscored, "per_task": per,
                                                      "api_models": sorted({str(r.get('api_model')) for r in rows})}, indent=2))


def software_versions():
    """Versions that produced the results (benchmark commit, simulator, API client, GPU), for the submission."""
    import importlib.metadata as md
    import platform
    import subprocess
    import robomme
    def git(path):
        out = subprocess.run(["git", "-C", str(path), "log", "-1", "--format=%H"], capture_output=True, text=True)
        return out.stdout.strip() or None
    pkgs = {}
    for name in ("robomme", "mani_skill", "sapien", "torch", "openai", "numpy"):
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pkgs[name] = None
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", "-i",
                          __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]],
                         capture_output=True, text=True).stdout.strip()
    return {"python": platform.python_version(), "packages": pkgs, "gpu": gpu,
            "robomme_benchmark_commit": git(Path(robomme.__file__).resolve().parent),
            "harness_repo_commit": git(Path(__file__).resolve().parent)}


def main():
    global FRAME_STRIDE
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt-6-astra")
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--episodes", nargs="*", type=int, required=True)
    p.add_argument("--dataset", choices=["test", "val"], default="test")
    p.add_argument("--frame-stride", type=int, default=FRAME_STRIDE)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    FRAME_STRIDE = args.frame_stride
    run_dir = Path(args.run_dir)
    tasks = args.tasks or BenchmarkEnvBuilder.get_task_list()
    if not args.summarize:
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = {"harness": Path(__file__).name, "harness_version": HARNESS_VERSION,
               "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "model": args.model,
               "action_space": "waypoint", "dataset": args.dataset, "max_steps": MAX_STEPS,
               "max_model_calls": MAX_MODEL_CALLS, "max_consecutive_noops": MAX_CONSECUTIVE_NOOPS,
               "frame_stride": FRAME_STRIDE, "system_prompts": {t: system_prompt(t) for t in BenchmarkEnvBuilder.get_task_list()},
               "noop_text": NOOP_TEXT, "stick_tasks": list(STICK_TASKS), "software": software_versions()}
        cp = run_dir / "config.json"
        if cp.exists():
            old = json.loads(cp.read_text())
            diff = [k for k in cfg if old.get(k) != cfg[k]]
            if diff:
                raise SystemExit(f"{cp} differs from this configuration in {diff}; use a new --run-dir")
        else:
            cp.write_text(json.dumps(cfg, indent=2))
        client = OpenAI()
        for task in tasks:
            for ep in args.episodes:
                d = run_dir / f"{task}_ep{ep}"
                if (d / "result.json").exists() and json.loads((d / "result.json").read_text())["status"] != "harness_error":
                    continue
                print(f"[wp] {task} episode {ep} ...", flush=True)
                try:
                    row = run_episode(client, args.model, task, ep, d, args.dataset)
                    print(f"[wp]   -> {row['status']} ({row['end_reason']}) calls={row['model_calls']} "
                          f"waypoints={row['waypoints_executed']} cost=${row['cost_usd']}", flush=True)
                except Exception as ex:                    # harness failure: recorded unscored, rerun on resume
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "result.json").write_text(json.dumps({"task": task, "episode": ep, "status": "harness_error",
                                                               "success": None, "error": repr(ex), "cost_usd": None}))
                    print(f"[wp]   -> harness error, recorded as unscored: {ex!r}", flush=True)
    summarize(run_dir, tasks, args.episodes)


if __name__ == "__main__":
    main()
