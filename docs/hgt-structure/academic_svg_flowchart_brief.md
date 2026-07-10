# Academic SVG Flowchart Draft Brief

## Purpose
Create an academic, publication-ready SVG flowchart that explains the HGT-based actor-critic scheduler used in the HGT-JSSP framework. The diagram should be suitable for a paper figure, thesis slide, or conference presentation.

## Figure Title
**HGT-based Actor-Critic Scheduler for JSSP Intersection Control**

## Overall Figure Message
The current traffic state is encoded as a heterogeneous graph with operation, vehicle, and zone nodes. A 3-layer HGT encoder performs relation-aware message passing and outputs operation embeddings. The actor scores all operation nodes and applies a hard feasible-action mask, while the critic pools unscheduled operation embeddings to estimate the PPO value.

## Canvas and Style
- SVG canvas size: `1600 x 950`.
- Background: white or very light gray `#F8FAFC`.
- Style: clean academic systems diagram, not marketing style.
- Use rounded rectangles with radius `8`.
- Use thin but visible strokes, `1.5-2 px`.
- Font: `Inter`, `Arial`, or `Helvetica`.
- Main title size: `30-34 px`, bold.
- Section title size: `18-22 px`, bold.
- Body text size: `13-15 px`.
- Formula text size: `13-14 px`, monospace or serif math-like style.
- Avoid decorative gradients or shadows. Use flat colors and clear grouping.

## Color Encoding
- Operation nodes / operation embeddings: blue `#2563EB`.
- Vehicle nodes: green `#059669`.
- Zone nodes: orange `#D97706`.
- HGT encoder blocks: purple `#7C3AED`.
- Actor branch: teal `#0F766E`.
- Critic branch: amber/brown `#B45309`.
- Feasible mask / invalid action suppression: red `#DC2626`.
- Neutral blocks: light slate `#F1F5F9`, border `#CBD5E1`.
- Arrows: dark slate `#475569`.

## High-level Layout
Use a left-to-right architecture flow with five major regions:

1. **Traffic State**
2. **Heterogeneous Graph Input**
3. **Type-specific Projection**
4. **3-layer HGT Encoder**
5. **Actor-Critic Decision Module**

Recommended x-axis placement:
- Region 1: `x=60-250`
- Region 2: `x=330-600`
- Region 3: `x=680-900`
- Region 4: `x=980-1240`
- Region 5: `x=1320-1540`

Use arrows between regions:
`Traffic State -> Heterogeneous Graph -> Type-specific Projection -> HGT Encoder -> Operation Embeddings -> Actor/Critic`.

## Region 1: Traffic State
Draw a rounded rectangle titled:
**Current Traffic State**

Inside the block, list:
- Vehicles and arrival times
- Routes through conflict zones
- Zone availability
- Scheduled / unscheduled operations
- Feasible action set

Small subtitle:
`Decision step t`

Suggested visual:
Include a small minimal intersection icon or a 3x3 grid of conflict zones, but keep it simple and schematic.

## Region 2: Heterogeneous Graph Input
Draw a larger rounded rectangle titled:
**Heterogeneous Graph \(G_t\)**

Inside it, draw three horizontal node-type bands:

1. Blue band:
   `Operation nodes`
   `X_op ∈ R^{n_o × 6}`

2. Green band:
   `Vehicle nodes`
   `X_veh ∈ R^{n_v × 4}`

3. Orange band:
   `Zone nodes`
   `X_zone ∈ R^{n_z × 5}`

Add a small relation legend inside or below this region:
- `seq: operation -> operation`
- `lane: operation -> operation`
- `conflict: operation <-> operation`
- `owns: vehicle -> operation`
- `hosts: zone -> operation`

Important visual detail:
Show operation nodes as the action units. Vehicle and zone nodes are contextual sources.

## Region 3: Type-specific Input Projection
Draw a rounded rectangle titled:
**Type-specific Linear Projection**

Inside the block, show:

```text
H_tau^0 = σ(X_tau W_in^tau + b_in^tau)
```

Then show three mappings:

```text
operation: [n_o, 6] -> [n_o, 128]
vehicle:   [n_v, 4] -> [n_v, 128]
zone:      [n_z, 5] -> [n_z, 128]
```

Small annotation:
`Shared hidden dimension F = 128`

## Region 4: 3-layer HGT Encoder
Draw a large purple container titled:
**HGT Encoder: 3 × HGTConv**

Inside, draw three stacked or sequential sub-blocks:
- `HGTConv Layer 1`
- `HGTConv Layer 2`
- `HGTConv Layer 3`

Each layer should indicate:

```text
Input:  operation [n_o, 128]
        vehicle   [n_v, 128]
        zone      [n_z, 128]

Output: operation [n_o, 128]
```

Add a small note inside the container:
`Vehicle and zone embeddings are carried forward as contextual sources.`

## Q/K/V Inset inside HGT Encoder
Inside the HGT encoder region, include an inset panel titled:
**Multi-head Q/K/V Computation**

Show the following flow:

```text
h_tau^ell [128]
   -> Linear(128, 384)
   -> split into K, Q, V
   -> reshape to [4 heads, 32 dim/head]
```

Use exact formula:

```text
[K_tau || Q_tau || V_tau] = H_tau^ell W_KQV^tau + b_KQV^tau
```

Add dimensions:

```text
F = 128, H = 4, d_h = 32
```

Then add relation-aware projection formula:

```text
Q_v = Q_{tau_t}(v)
K_u^r = W_K^r K_{tau_s}(u)
V_u^r = W_V^r V_{tau_s}(u)
```

## Attention Formula Panel
Add a compact formula panel titled:
**Relation-aware Attention**

Use:

```text
score(u,v,r) = Q_v · K_u^r / sqrt(32)
alpha(u,v,r) = softmax(score)
message = alpha · V_u^r
```

Add a short annotation:
`Messages are aggregated over seq, lane, conflict, owns, and hosts relations.`

Implementation fidelity note:
Do **not** draw edge attributes as an explicit `phi(e_uv)` attention term. The current implementation constructs edge attributes in the graph, but the HGT forward pass does not pass them into HGTConv.

## Region 5: Operation Embeddings
Between the HGT encoder and actor-critic heads, draw a blue output block:

**Final Operation Embeddings**

Text:

```text
H_op^L ∈ R^{n_o × 128}
```

Add subtitle:
`One embedding per operation/action`

This block should branch into Actor and Critic.

## Actor Branch
Draw a teal block titled:
**Actor: Masked Dispatching Policy**

Inside:

```text
MLP_actor: 128 -> 64 -> 1
```

Then:

```text
all operation logits: [n_o]
```

Then red mask block:

```text
if o ∉ A_t: logit_o = -∞
```

Then output:

```text
π_θ(o | G_t) = softmax(masked logits)
```

Important wording:
`The actor scores all operations first; infeasible operations are masked before softmax.`

## Critic Branch
Draw an amber/brown block titled:
**Critic: Graph-level Value**

Inside:

```text
Select unscheduled operations U_t
Mean pooling:
h_G = mean({h_o^L | o ∈ U_t})
```

Then:

```text
MLP_critic: 128 -> 64 -> 1
V_φ(G_t)
```

Small note:
`Used by PPO / GAE advantage estimation`

## Bottom Methodological Separation Bar
At the bottom of the figure, draw a horizontal bar with four labeled segments:

1. **HGT + Actor**
   `learn dispatching preference`

2. **Feasible Mask**
   `enforce valid JSSP actions`

3. **Environment Decoder**
   `compute exact start / finish times`

4. **Low-level Controller**
   `track the planned schedule`

This bar should make clear that the neural network does not directly compute exact crossing times.

## Suggested Full SVG Text Content
Use the following exact English phrases where possible:

```text
Current Traffic State
Heterogeneous Graph G_t
Operation nodes: X_op ∈ R^{n_o × 6}
Vehicle nodes: X_veh ∈ R^{n_v × 4}
Zone nodes: X_zone ∈ R^{n_z × 5}

Type-specific Linear Projection
all node embeddings -> 128-dim

3 × HGTConv Encoder
4 attention heads, d_h = 32
Q/K/V: 128 -> 384 -> 3 × [4, 32]
Relation-aware message passing

Final Operation Embeddings
H_op^L ∈ R^{n_o × 128}

Actor
128 -> 64 -> 1 logits
Hard feasible-action mask
π_θ(o | G_t)

Critic
mean pool unscheduled operations
128 -> 64 -> 1 value
V_φ(G_t)
```

## Recommended Figure Caption
**Figure X. HGT-based actor-critic architecture for intersection scheduling.**  
At each decision step, the environment state is represented as a heterogeneous graph with operation, vehicle, and zone nodes. Type-specific linear projections map raw features into a shared 128-dimensional hidden space. A 3-layer HGT encoder with 4 attention heads performs relation-aware message passing over sequence, lane, conflict, ownership, and hosting relations. The actor scores all operation nodes and applies a hard feasible-action mask before the policy softmax. The critic mean-pools unscheduled operation embeddings to estimate the graph-level value used for PPO training.

## Negative Instructions for the Drawing Agent
- Do not show the actor as scoring only feasible operations. It scores all operations, then masks infeasible ones.
- Do not show `h_G` or global features as actor inputs. The actor uses only each operation embedding.
- Do not show explicit edge-feature attention bias `phi(e_uv)` in the HGTConv block.
- Do not draw a generic Transformer encoder without graph relations. The diagram must emphasize typed graph message passing.
- Do not let the figure imply that the neural network computes exact schedule times. Exact timing is computed by the environment decoder after action selection.

## Optional Academic Layout Variant
If a more paper-like figure is preferred, use a two-row layout:

Top row:
`Traffic State -> Heterogeneous Graph -> HGT Encoder -> Operation Embeddings`

Bottom row:
`Q/K/V detail inset -> Actor branch -> Critic branch -> PPO`

This variant is better for a wide conference-paper figure because the formulas can sit below the high-level blocks without overcrowding the main flow.
