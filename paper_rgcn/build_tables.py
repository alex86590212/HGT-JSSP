"""Generate LaTeX result tables from runs/compare_eval_wstar.csv (R-GCN vs iGreedy, B.P., LIFO; other columns ignored)."""
import csv

rows = list(csv.DictReader(open("../runs/compare_eval_wstar.csv")))
f = lambda r, k: float(r[k])
mean = lambda xs: sum(xs) / len(xs)
sg = lambda x: f"{x:+.1f}".replace("-", "$-$")
BASE = ["igreedy", "backpressure", "lifo"]
cols = BASE + ["rgcn"]


def best_bold(vals):
    lo = min(vals)
    return [("\\textbf{%.3f}" % v) if v == lo else "%.3f" % v for v in vals]


def red(w_b, w_r):
    return (w_b - w_r) / w_b * 100


# Table 1: scaling
o = [r"""\begin{table}[t]
\centering
\setlength{\tabcolsep}{4pt}
\caption{Mean per-vehicle waiting time (s) versus arrival rate (45 paired episodes per rate and policy). Avg.\ $n$ veh.\ is the mean number of vehicles per episode. Bold marks the lowest value in each row. The bottom row is the unweighted mean over the eight rates.}
\label{tab:scaling}
\begin{tabular}{lccccc}
\toprule
Rate & Avg. & iGreedy & B.P. & LIFO & R-GCN \\
(veh/s) & $n$ veh & & & & (ours) \\
\midrule"""]
for r in rows:
    cells = best_bold([round(f(r, c + "_wt_mean"), 3) for c in cols])
    o.append(f"{f(r,'rate'):.2f} & {f(r,'n_veh_avg'):.1f} & " + " & ".join(cells) + r" \\")
o.append(r"\midrule")
means = {c: mean([f(r, c + "_wt_mean") for r in rows]) for c in cols}
o.append("Mean & -- & " + " & ".join(best_bold([round(means[c], 3) for c in cols])) + r" \\")
o.append(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}")
t1 = "\n".join(o) + "\n"

# Table 2: reduction + W/T/L
o = [r"""\begin{table}[t]
\centering
\setlength{\tabcolsep}{4pt}
\caption{Waiting-time reduction of the R-GCN relative to each baseline, $100(\bar W_B-\bar W_{\mathrm{R\text{-}GCN}})/\bar W_B$ (\%; negative values mean the R-GCN is worse), and the R-GCN's win/tie/loss count against iGreedy over the 45 paired episodes (ties: $|\Delta W|\le10^{-6}$\,s).}
\label{tab:reduction}
\begin{tabular}{lcccc}
\toprule
Rate & iGreedy & B.P. & LIFO & W/T/L vs.\ iGreedy \\
\midrule"""]
for r in rows:
    cells = " & ".join(sg(red(f(r, b + "_wt_mean"), f(r, "rgcn_wt_mean"))) for b in BASE)
    o.append(f"{f(r,'rate'):.2f} & {cells} & {r['rgcn_wins']}/{r['rgcn_ties']}/{r['rgcn_losses']}" + r" \\")
o.append(r"\midrule")
o.append("Mean & " + " & ".join(sg(red(means[b], means["rgcn"])) for b in BASE) + r" & -- \\")
o.append(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}")
t2 = "\n".join(o) + "\n"

# Table 3: W* and defer
o = [r"""\begin{table}[t]
\centering
\setlength{\tabcolsep}{4pt}
\caption{Receding-horizon reference $W^*$ and R-GCN deferral rate. $W^*$ is the mean over the episodes in which every CP-SAT subproblem was certified optimal within 5\,s, so its sample differs from the 45-episode means in Table~\ref{tab:scaling} and is not directly comparable to them. ``Defer'' is the fraction of R-GCN decisions that chose DEFER.}
\label{tab:wstar}
\begin{tabular}{lcc}
\toprule
Rate (veh/s) & $W^*$ (s) & Defer \\
\midrule"""]
for r in rows:
    o.append(f"{f(r,'rate'):.2f} & {f(r,'w_star'):.3f} & {f(r,'rgcn_defer_frac'):.2f}" + r" \\")
o.append(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}")
t3 = "\n".join(o) + "\n"

for name, c in [("tab_scaling", t1), ("tab_reduction", t2), ("tab_wstar", t3)]:
    open(name + ".tex", "w").write(c)
print("means", {c: round(m, 4) for c, m in means.items()})
print("reduction of means", {b: round(red(means[b], means["rgcn"]), 1) for b in BASE})
for r in rows:
    print(r["rate"], "R-GCN-iG abs", round(f(r, "rgcn_wt_mean") - f(r, "igreedy_wt_mean"), 3),
          "R-GCN-BP", round(f(r, "rgcn_wt_mean") - f(r, "backpressure_wt_mean"), 3),
          "R-GCN-LIFO", round(f(r, "rgcn_wt_mean") - f(r, "lifo_wt_mean"), 3))
