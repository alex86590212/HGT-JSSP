# HGT-JSSP — Training & Implementation Notes

## Bugs Fixed

### `zone.occupied` was stale (graph_builder.py)
`zone.occupied` was set to `True` on first use and never reset. The policy was seeing
`occ=1` for every zone that had ever been used, even when the zone was long free.
Fixed to derive occupation from `zone.time_free > env.current_time` — matching what
the feasibility checker actually uses.

> **Impact on existing checkpoints:** `results/` and `results_v2/` were trained with
> the stale feature. Their weights partially compensate for the wrong input. Any run
> started after the fix gets the correct signal from step 1.

### `igreedy` had the early-break bug (metrics.py)
`igreedy()` broke out of the scheduling loop when the feasible set was empty instead
of advancing time like `run_episode` does. This made the iGreedy baseline measure
partial schedules while HGT measured complete ones — an unfair comparison.
Fixed to use `next_feasible_time` consistently.

### Episode early-break ("temporal gap") (trainer.py)
Originally, `run_episode` would `break` when `feasible_mask.any() == False`.
In scenarios with staggered arrivals this ended episodes early — unscheduled vehicles
kept `earliest_finish = arrival + sum(p)` → zero waiting time → artificially low metrics.
Fixed by advancing `env.current_time` to `next_feasible_time()` and continuing.

> The "original weights never reached 1s" result was a measurement artifact from this.
> Eval numbers before and after the fix are not directly comparable.

---

## Training Observations (results_v2, ~34k episodes)

### Eval waiting time
- Best: **0.756s** at ~episode 17k
- Two bumps (episodes 1k–3k and 12k–15k) are curriculum transition artifacts — model
  hits harder scenarios before adapting, then recovers. Expected.
- Hovering around 0.85–0.95s in later episodes without clearly converging.
- **Needs iGreedy comparison** to know if this is actually good. Run `python eval.py`.

### Critic loss explosion at episode 15k
- 0–15k: well-behaved, oscillates 0–0.5
- 15k+: **explodes** to persistent 0.5–3.5 with spikes to 3.5+

The 15k mark is the curriculum transition to hard (4–7 vehicles). More vehicles means
larger cumulative negative rewards — return magnitude scales roughly with `n_vehicles`.
The critic's MSE target shifts to a much larger range and never recovers.

**This is the root cause of all the other variance.** A broken critic produces bad
advantage estimates → actor gets garbage gradient signal → actor loss stays noisy →
eval never converges.

### Entropy collapse then jump
- Collapses to near 0 in the first ~2k episodes — policy over-committed immediately
  on easy 2-vehicle scenarios
- Recovers, then jumps at 15k because hard scenarios have more feasible actions per
  step — larger action space raises entropy even for the same policy uncertainty
- Slowly trending down after 30k, which is healthy, but still noisy

### Actor loss not tightening
The actor loss variance is statistically identical between episode 5k and 34k.
A converging policy should show the loss compressing toward a tighter band over time.
This is a downstream symptom of the critic explosion above.

**Secondary cause: single-episode PPO.** Each gradient update sees exactly one
trajectory. With one sample, advantage estimates are high-variance even with a working
critic.

---

## Potential Improvements

### 1. Batched PPO updates (highest priority)
Collect a rollout buffer of N episodes before each update instead of updating every
episode. This is the standard PPO setup and directly addresses the variance problem.

```python
# trainer.py — sketch
ROLLOUT_EPISODES = 8
buffer = []
for episode in ...:
    transitions, stats = run_episode(policy, env, scenario)
    buffer.extend(transitions)
    if len(buffer) >= ROLLOUT_EPISODES * avg_steps:
        ppo_update(policy, optimizer, buffer, ...)
        buffer = []
```

### 2. Time-skip penalty
When `current_time` is advanced via `next_feasible_time()`, no transition is recorded
and no gradient flows. The policy can't distinguish states that precede costly idle gaps.
A simple fix: accumulate a `pending_reward` during time skips and add it to the next
real transition's reward.

### 3. Normalize rewards per episode length (highest priority alongside batching)
Reward magnitude scales with number of vehicles (more vehicles → larger negative reward).
This is the direct cause of the critic loss explosion at the 15k curriculum transition.
The critic's MSE target jumps to a much larger range when vehicle count increases, and
it never recovers — which cascades into bad advantages and a noisy actor.

Dividing the step reward by `n_vehicles` keeps returns on a consistent scale across
all curriculum phases and should prevent the explosion.

```python
# intersection.py _compute_reward — or normalize in trainer.py after env.step()
reward = reward / len(env.vehicles)
```

### 4. Entropy coefficient schedule
A fixed `entropy_coef=0.01` provides the same exploration pressure at episode 1 and
episode 50k. Consider annealing it (e.g. 0.01 → 0.001) so the policy commits more
as training progresses.

---

## Evaluation

```bash
# Compare best checkpoint vs iGreedy across all tiers
python eval.py --checkpoint results_v2/checkpoint_best.pt

# Compare multiple checkpoints
python eval.py --checkpoint results_v2/checkpoint_17000.pt  # likely best actual weights
```

Note: `checkpoint_best.pt` is saved at whichever eval interval had the lowest
`waiting_time` — but eval uses only 50 hard scenarios, so a lucky eval can save a
checkpoint that isn't genuinely best. Use `--n-scenarios 200` for a more reliable
comparison.

## Baseline Results (results_v2/checkpoint_best.pt, 200 scenarios)

Confirmed stable at n=200 (vs n=50 — numbers held and slightly improved).

| Tier   | iGreedy (s) | HGT (s) | Improvement |
|--------|-------------|---------|-------------|
| easy   | 0.111       | 0.082   | +25.9%      |
| medium | 0.620       | 0.387   | +37.6%      |
| hard   | 2.083       | 1.062   | +49.0%      |
| overall| 0.938       | 0.510   | +45.6%      |

Hard tier is the strongest result — where the graph structure (conflict edges, urgency
features) does the most work. iGreedy's 2.08s vs HGT's 1.10s means vehicles at a busy
intersection wait roughly half as long.

Easy tier is weakest (+26%) because iGreedy is near-optimal when there are no conflicts.

**Target for next run (with reward normalization + batched rollouts):** hard tier >55%,
reached in fewer than 34k episodes.
