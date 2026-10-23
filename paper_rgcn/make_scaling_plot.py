"""Scaling plot for the R-GCN paper, from runs/compare_eval_wstar.csv (HGT columns are ignored)."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("../runs/compare_eval_wstar.csv")))
x = [float(r["rate"]) for r in rows]
col = lambda k: [float(r[k]) for r in rows]

series = [
    ("igreedy_wt_mean", "iGreedy", dict(color="#E45756", marker="o", ls="-")),
    ("lifo_wt_mean", "LIFO", dict(color="#7f7f7f", marker="v", ls="--")),
    ("backpressure_wt_mean", "Backpressure", dict(color="#8c8c2b", marker="s", ls="-")),
    ("rgcn_wt_mean", "R-GCN (ours)", dict(color="#2B6E6B", marker="*", ls="-", lw=2.2, ms=9)),
    ("w_star", "$W^*$ (solved episodes)", dict(color="black", marker="", ls=":")),
]

fig, ax = plt.subplots(figsize=(3.5, 2.6))
for key, label, st in series:
    st = dict(st)
    lw = st.pop("lw", 1.3)
    ms = st.pop("ms", 4)
    ax.plot(x, col(key), label=label, lw=lw, ms=ms, **st)
ax.set_xlabel("Arrival rate $\\lambda$ (veh/s)", fontsize=8)
ax.set_ylabel("Mean waiting time (s)", fontsize=8)
ax.tick_params(labelsize=7)
ax.grid(alpha=0.3, lw=0.4)
ax.legend(fontsize=6, frameon=False, loc="upper left")
fig.tight_layout(pad=0.4)
fig.savefig("scaling_plot.pdf")
print("saved")
