"""Export a trained admission policy checkpoint to ONNX for viewing in Netron.

No retraining: loads the weights, traces one real decision state and writes
the graph. Shapes are fixed to that example, which is fine for inspection.

  PYTHONPATH=. python export_onnx.py --package admission_rl \
      --checkpoint runs/admission_rl_v2/checkpoint_best.pt --output runs/admission_rl_v2/policy.onnx
  PYTHONPATH=. python export_onnx.py --package admission_rgcn \
      --checkpoint runs/admission_rgcn_v1/checkpoint_best.pt --output runs/admission_rgcn_v1/policy.onnx
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import textwrap

import torch
import torch.nn as nn

from dynamic_scheduler.data.traffic_generator import TrafficGenerator
from dynamic_scheduler.environment.gapfree_admission import AdmissionEnv


def _patch_hgtconv_for_tracing() -> None:
    """HGTConv keeps running node-count offsets with `cumsum += x.size(0)`.
    While tracing, size() is a tensor, so `+=` mutates in place and every
    recorded offset aliases the final total. Rebind instead of mutating."""
    from torch_geometric.nn.conv import hgt_conv

    for name in ("_cat", "_construct_src_node_feat"):
        src = textwrap.dedent(inspect.getsource(getattr(hgt_conv.HGTConv, name)))
        src = src.replace("cumsum += ", "cumsum = cumsum + ")
        ns: dict = {}
        exec(src, vars(hgt_conv), ns)  # noqa: S102
        setattr(hgt_conv.HGTConv, name, ns[name])


class _TypeView:
    def __init__(self, batch):
        self.batch = batch


class _GraphShim:
    """Just the attributes AdmissionPolicy.forward reads from a PyG Batch."""

    def __init__(self, x_dict, edge_index_dict):
        self.x_dict = x_dict
        self.edge_index_dict = edge_index_dict
        self.num_graphs = 1
        self._views = {nt: _TypeView(torch.zeros(x.shape[0], dtype=torch.long)) for nt, x in x_dict.items()}

    def __getitem__(self, node_type):
        return self._views[node_type]


class _Wrapper(nn.Module):
    def __init__(self, policy, node_types, edge_types):
        super().__init__()
        self.policy = policy
        self.node_types = node_types
        self.edge_types = edge_types

    def forward(self, *args):
        n, e = len(self.node_types), len(self.edge_types)
        x_dict = dict(zip(self.node_types, args[:n]))
        edge_index_dict = dict(zip(self.edge_types, args[n:n + e]))
        feasible_mask, defer_allowed = args[n + e], args[n + e + 1]
        logits, value = self.policy(_GraphShim(x_dict, edge_index_dict), feasible_mask, defer_allowed)
        return logits, value


def example_state(pkg, env_cfg):
    """A real mid-episode decision state (some committed vehicles, several pending)."""
    graph = importlib.import_module(f"{pkg}.graph")
    T, dw = env_cfg["episode_duration"], env_cfg["detection_window"]
    arrivals = TrafficGenerator(seed=5).generate_episode_arrivals(T, arrival_rate=1.0)
    env = AdmissionEnv(detection_window=dw, commit_window=env_cfg["commit_window"])
    env.reset(arrivals, T)
    env.current_time = 0.0
    pending = sorted([v for v in arrivals if v.arrival_time <= dw], key=lambda v: (v.arrival_time, v.id))
    out = None
    for _ in range(4):
        order = [v.id for v in pending]
        status = {v.id: env.admission_check(v) for v in pending}
        out = graph.build_admission_graph(env, order, status)
        feasible = [v for v in pending if status[v.id][0]]
        if not feasible:
            break
        env.admit(feasible[0])
        pending.remove(feasible[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, choices=["admission_rl", "admission_rgcn"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    _patch_hgtconv_for_tracing()
    common = importlib.import_module(f"{args.package}.common")
    graph = importlib.import_module(f"{args.package}.graph")
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")  # noqa: S301
    policy = common.make_policy(ckpt["config"]["model"])
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    data, mask = example_state(args.package, ckpt["config"]["env"])
    inputs = (
        *[data[nt].x for nt in graph.NODE_TYPES],
        *[data[et].edge_index for et in graph.EDGE_TYPES],
        mask,
        torch.tensor([True]),
    )
    names = (
        [f"x_{nt}" for nt in graph.NODE_TYPES]
        + [f"edge_{s}_{r}_{d}" for s, r, d in graph.EDGE_TYPES]
        + ["feasible_mask", "defer_allowed"]
    )
    wrapper = _Wrapper(policy, graph.NODE_TYPES, graph.EDGE_TYPES).eval()
    with torch.no_grad():
        ref_logits, ref_value = wrapper(*inputs)
    kwargs = dict(input_names=names, output_names=["logits", "value"], opset_version=args.opset, dynamo=False)
    try:
        torch.onnx.export(wrapper, inputs, args.output, **kwargs)
    except torch.onnx.errors.UnsupportedOperatorError as err:
        # PyG's attention uses ops ONNX cannot express. Keep them as plain
        # aten:: nodes: fine for Netron, but the file is not runnable.
        print(f"unsupported operator ({err}); retrying with unsupported ops kept as aten:: nodes")
        torch.onnx.export(wrapper, inputs, args.output,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, **kwargs)
    print(f"wrote {args.output}  (example: {data['pending'].x.shape[0]} pending, "
          f"{data['committed'].x.shape[0]} committed)")
    print(f"torch logits[:5]={ref_logits[0, :5].tolist()}  value={float(ref_value[0]):.4f}")


if __name__ == "__main__":
    main()
