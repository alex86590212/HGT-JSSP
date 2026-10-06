import sys
sys.path.insert(0, ".")
from dynamic_scheduler.data.traffic_generator import TrafficGenerator
from intersection_scheduler.data.scenario_generator_4x4 import SAME_LANE_GROUPS, _MANOEUVRE_TO_LANE

rates = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00, 1.20]
seeds = [0, 1, 2]
episode_duration = 60.0

all_gaps = []
per_rate = {}
below_1s = 0
below_0_5s = 0
below_0_1s = 0
zero_gap = 0
total_pairs = 0

for rate in rates:
    rate_gaps = []
    for seed in seeds:
        gen = TrafficGenerator(seed=seed * 100 + int(rate * 1000))
        for _ in range(15):
            arrivals = gen.generate_episode_arrivals(episode_duration, arrival_rate=rate)
            by_lane = {}
            for v in arrivals:
                lane = _MANOEUVRE_TO_LANE[v.manoeuvre]
                by_lane.setdefault(lane, []).append(v.arrival_time)
            for lane, times in by_lane.items():
                times.sort()
                for i in range(1, len(times)):
                    gap = times[i] - times[i-1]
                    rate_gaps.append(gap)
                    all_gaps.append(gap)
                    total_pairs += 1
                    if gap < 1.0:
                        below_1s += 1
                    if gap < 0.5:
                        below_0_5s += 1
                    if gap < 0.1:
                        below_0_1s += 1
                    if gap < 1e-6:
                        zero_gap += 1
    if rate_gaps:
        rate_gaps.sort()
        n = len(rate_gaps)
        per_rate[rate] = (min(rate_gaps), rate_gaps[n//100] if n>=100 else rate_gaps[0], sum(rate_gaps)/n, sum(1 for g in rate_gaps if g<1.0)/n*100)

print(f"Total same-lane consecutive pairs: {total_pairs}")
print(f"Min gap overall: {min(all_gaps):.4f}s")
print(f"Pairs with gap < 1.0s (violates offline min_headway floor): {below_1s} ({below_1s/total_pairs*100:.1f}%)")
print(f"Pairs with gap < 0.5s: {below_0_5s} ({below_0_5s/total_pairs*100:.1f}%)")
print(f"Pairs with gap < 0.1s: {below_0_1s} ({below_0_1s/total_pairs*100:.1f}%)")
print(f"Pairs with gap ~0 (< 1e-6s, simultaneous): {zero_gap}")
print()
print("Per-rate: min_gap | p1_gap | mean_gap | pct<1.0s")
for rate, (mn, p1, mean, pct) in per_rate.items():
    print(f"  rate={rate:.2f}: min={mn:.4f}s  p1={p1:.4f}s  mean={mean:.4f}s  pct<1.0s={pct:.1f}%")
