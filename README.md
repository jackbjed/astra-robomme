# GPT-6-Astra on RoboMME (end-to-end, `waypoint` action space)

Evaluation code, prompts, configuration and per-episode logs for our [RoboMME](https://github.com/RoboMME/robomme_benchmark)
leaderboard submission: OpenAI's `gpt-6-astra`, used through the API as the robot policy, with no training or fine-tuning on
RoboMME. At every step the model receives the task goal, camera frames and the robot's own state, and outputs the next
end-effector waypoint (position, orientation) and gripper command. RoboMME's motion planner only executes each waypoint.
Memory is the model's own context: the whole episode (demonstration video, all execution frames and the model's previous
answers) stays in one conversation.

## Results (test split, 50 episodes per task, one run)

| Suite | Task | Success (%) |
|---|---|---|
| Counting | BinFill | 100 |
| | PickXtimes | 100 |
| | SwingXtimes | 0 |
| | StopCube | 56 |
| Permanence | VideoUnmask | 100 |
| | VideoUnmaskSwap | 98 |
| | ButtonUnmask | 100 |
| | ButtonUnmaskSwap | 98 |
| Reference | PickHighlight | 98 |
| | VideoRepick | 100 |
| | VideoPlaceButton | 100 |
| | VideoPlaceOrder | 98 |
| Imitation | MoveCube | 54 |
| | InsertPeg | 98 |
| | PatternLock | 100 |
| | RouteStick | 98 |
| **Overall** | | **87.38** (699/800) |

All 800 scheduled episodes are scored (no crashes or unscored episodes). Model version as reported by the API: `gpt-6-astra`
(the API offers no dated snapshot); evaluated 2026-10-08 to 2026-10-09 UTC.

**SwingXtimes (0/50).** RoboMME's "right-side target" is the robot's right (smaller y); the front camera faces the robot,
so the image is mirrored, and the model consistently took "right" from the image. It performed every back-and-forth, in
mirrored order, so each episode failed at the final button press. A separate **ablation** (`harness/astra_waypoint_eval_v3.py
--prompt-variant lr_note`) adds one task-independent convention note ("left and right are from the robot's point of view;
the front camera is mirrored"), similar to the conventions RoboMME's own oracle-planner prompts state for PatternLock and
RouteStick. With it, SwingXtimes is 50/50 on test (9/10 on validation). That note was written **after** this run's test
results, so it is **not** part of the submitted configuration; the submitted score is 87.38.

## Protocol and compliance with the leaderboard rules

The episode loop is RoboMME's `scripts/evaluation.py` (included verbatim in `third_party/robomme/`, Apache-2.0):
`BenchmarkEnvBuilder(dataset="test", action_space="waypoint", max_steps=1300)`, episodes `range(get_episode_num())`
(50 per task), task goal `info["task_goal"][0]`, step until terminated/truncated, success iff `info["status"] == "success"`;
environment errors (e.g. an unreachable pose) count as failures.

- **Action space:** `waypoint` (allowed; `multi_choice` is not used).
- **Single model for all tasks:** one model and one system prompt for all 16 tasks. The only task-dependent sentence
  states the hardware of PatternLock and RouteStick (the robot holds a stick, so the gripper command is ignored).
- **No ground-truth information:** the model sees only front and wrist RGB frames (256 x 256), the TCP pose, finger
  positions and elapsed control steps, and the front/wrist camera intrinsics and extrinsics (RoboMME's documented optional
  observations `include_*_camera_*`). Never used: depth, simulator state (`maniskill_obs`), the multi-choice options,
  RoboMME's oracle subgoal fields. World-frame facts in the prompt (robot base at (-0.615, 0, 0), axes) are from RoboMME's
  `doc/env_format.md`.
- **Runs:** one run. The leaderboard page recommends multiple runs; the submission template asks for three.
- **Benchmark version:** RoboMME commit `856bc3a` (its environments, task definitions, seeds and `evaluation.py` are
  identical to `main` as of 2026-10-09).

### Model inputs, per step
Task goal; front-camera frames of the demonstration/history video and of every executed motion, one in every 3 control
steps (0.15 s) plus the last frame; the current wrist-camera frame; robot state and camera calibration. The exact system
prompt for each task, the message sent after an action that executes no steps, and all limits are frozen in
`results/e2e_v2_test_800/config.json`.

### Episode endings besides the environment's (all scored as failures)
The model declares the task complete before the environment judges success (`evaluation.py`'s loop has no way for a policy
to stop, and a "stay in place" waypoint crashes RoboMME's planner); 3 consecutive actions that execute no steps; an unusable
answer; 150 model calls (never reached: the maximum was 29). Outcome counts: 699 success, 72 fail, 24 declared complete,
3 environment errors, 2 unusable answers.

### Disclosures
- The system prompt is ours (RoboMME ships none for `waypoint`). Its "Notes on the robot and this interface" (motion timing,
  gripper behaviour, explicit message after an action that executes no steps) are general facts identical for all tasks.
  They were written after analysing failures of a separate `multi_choice` (oracle-planner) evaluation of this model on the
  test split. The motion timings they quote were measured on the **train** split.
- The frame density (1 in 3 frames) was likewise chosen after that separate `multi_choice` evaluation.
- A 2-episode pilot of an earlier version of this harness ran on test episodes 0-1; the prompt was not tuned on it.
- `gpt-6-astra` is a closed API model; no checkpoint can be released.

## Reproduce

```bash
# RoboMME benchmark at commit 856bc3a (see https://github.com/RoboMME/robomme_benchmark), Python 3.11,
# mani_skill 3.0.0b21, sapien 3.0.3, torch 2.9.1, openai 2.42.0 (exact versions: results/e2e_v2_test_800/config.json)
export OPENAI_API_KEY=...
python harness/astra_waypoint_eval_v2.py --run-dir runs/e2e_v2_test_800 --episodes $(seq 0 49)   # resumable
python harness/astra_waypoint_eval_v2.py --run-dir runs/e2e_v2_test_800 --episodes $(seq 0 49) --summarize
python harness/make_submission_md.py --runs runs/e2e_v2_test_800 --out submission/gpt-6-astra.md
```
`config.json` records the harness's SHA-256 (`harness/astra_waypoint_eval_v2.py` matches it), the benchmark commit and
software versions. Cost of the full run: about $310 at list API prices.

## Repository layout

| Path | Contents |
|---|---|
| `harness/astra_waypoint_eval_v2.py` | the evaluation harness used for the submission |
| `harness/astra_waypoint_eval_v3.py` | v2 + the optional `lr_note` ablation (default identical to v2) |
| `harness/make_submission_md.py` | writes the leaderboard `.md` from a finished run; refuses if any episode is missing or unscored |
| `third_party/robomme/` | RoboMME's `scripts/evaluation.py`, verbatim, with its Apache-2.0 license |
| `results/e2e_v2_test_800/` | submitted run: `config.json`, `summary.json`, and per episode `result.json` + `exchange.jsonl` |
| `results/e2e_v3_lrnote_swing_{test,val}/` | SwingXtimes left/right-note ablation |
| `submission/` | the leaderboard submission file |

Per episode, `result.json` holds the outcome, end reason, API-reported model version, UTC start/end, every action and its
result, token usage and cost. `exchange.jsonl` holds every request (images replaced by their file names) and every raw
API response; the model's encrypted reasoning blobs are omitted. The images sent to the model (13 GB) and the episode
videos (0.6 GB) are not in this repository; they are available on request.

## License

Our code is MIT-licensed (`LICENSE`). `third_party/robomme/evaluation.py` is RoboMME's, under Apache-2.0 (`third_party/robomme/LICENSE`).
