# PPT Design Draft: HGT Structure

## Slide 1 — Motivation and Role in the Framework

**Title**
HGT-based Neural Scheduler for JSSP Intersection Control

**Main Message**
The HGT scheduler learns which vehicle-zone operation should be dispatched next from a typed graph representation of the current traffic state.

**Visual Design**
- Left: simple intersection/conflict-zone icon.
- Center: heterogeneous graph icon with three node colors.
- Right: actor-critic output block.
- Use a left-to-right pipeline:
  `Traffic State -> Heterogeneous Graph -> HGT Encoder -> Actor-Critic Decision`

**On-slide Bullets**
- Converts each decision state into a heterogeneous scheduling graph.
- Encodes operation, vehicle, and zone information with relation-aware message passing.
- Produces a masked dispatching policy and a PPO value estimate.

**Speaker Notes**
This slide positions HGT as the decision module. The environment and mask enforce feasibility, while HGT learns dispatching preferences among valid operation choices.

---

## Slide 2 — Heterogeneous Graph Input

**Title**
Typed Graph Representation of the Scheduling State

**Main Message**
The current traffic state is represented using three node types and five relation types.

**Visual Design**
- Show three vertical node groups:
  - blue: `operation`
  - green: `vehicle`
  - orange: `zone`
- Draw typed edges between groups:
  - `seq`, `lane`, `conflict`
  - `owns`
  - `hosts`

**On-slide Bullets**
- Operation nodes: vehicle-zone scheduling actions, feature size `[n_o, 6]`.
- Vehicle nodes: arrival and route-level context, feature size `[n_v, 4]`.
- Zone nodes: resource availability and spatial context, feature size `[n_z, 5]`.
- Typed relations encode precedence, lane order, conflicts, ownership, and zone hosting.

**Speaker Notes**
Each operation is the action unit selected by the actor. Vehicle and zone nodes provide context, while typed relations describe scheduling constraints and resource dependencies.

---

## Slide 3 — Input Projection and Tensor Shapes

**Title**
From Raw Features to Shared Hidden Embeddings

**Main Message**
Different node types have different raw feature dimensions, so each type is projected into the same 128-dimensional hidden space.

**Visual Design**
- Three input boxes on the left:
  - `X_op [n_o, 6]`
  - `X_veh [n_v, 4]`
  - `X_zone [n_z, 5]`
- Three arrows through type-specific Linear layers.
- Three output boxes on the right:
  - `H_op [n_o, 128]`
  - `H_veh [n_v, 128]`
  - `H_zone [n_z, 128]`

**On-slide Formula**
```text
H_tau^0 = sigma(X_tau W_in^tau + b_in^tau)
```

**Example Shape**
```text
2 vehicles, 6 operations, 5 zones:
operation: [6, 6] -> [6, 128]
vehicle:   [2, 4] -> [2, 128]
zone:      [5, 5] -> [5, 128]
```

**Speaker Notes**
This projection step makes all node types compatible with the HGTConv layers. The raw feature dimensions differ, but the hidden representation dimension is shared.

---

## Slide 4 — Q/K/V Computation inside HGTConv

**Title**
Multi-head Q/K/V Projection for Typed Nodes

**Main Message**
Each HGT layer computes query, key, and value vectors using type-specific linear projections, then reshapes them into four attention heads.

**Visual Design**
- Show one node embedding block: `[128]`.
- Arrow into `Linear(128, 384)`.
- Split into three blocks:
  - `K [128]`
  - `Q [128]`
  - `V [128]`
- Reshape each into:
  - `[4 heads, 32 dim/head]`

**On-slide Formula**
```text
[K_tau || Q_tau || V_tau] = H_tau W_KQV^tau + b_KQV^tau
128 -> 384 -> 3 x 128 -> 3 x [4, 32]
```

**Example Shape**
```text
operation Q/K/V: [6, 4, 32]
vehicle Q/K/V:   [2, 4, 32]
zone Q/K/V:      [5, 4, 32]
```

**Speaker Notes**
The Q/K/V computation is not a deep MLP. It is a learned linear projection. HGT differs from a standard Transformer because the projection depends on the node type.

---

## Slide 5 — Relation-aware Attention and Message Passing

**Title**
Relation-aware Message Passing over Typed Edges

**Main Message**
Operation nodes use their queries to attend to incoming typed messages from operations, vehicles, and zones.

**Visual Design**
- Center: one target operation node.
- Incoming arrows:
  - from operation via `seq`
  - from operation via `lane`
  - from operation via `conflict`
  - from vehicle via `owns`
  - from zone via `hosts`
- Add small labels:
  - `Q_target`
  - `K_source`
  - `V_source`

**On-slide Formula**
```text
score = Q_target · K_source / sqrt(32)
alpha = softmax(score)
message = alpha · V_source
```

**Key Point**
Relation-specific transformations are applied to source keys and values before attention aggregation.

**Speaker Notes**
The target operation asks what information it needs through Q. Neighboring source nodes provide K and V. Attention weights decide how much each relation-specific message contributes to the updated operation embedding.

---

## Slide 6 — Three HGT Layers and Output Embeddings

**Title**
Stacked HGTConv Layers Preserve Hidden Size

**Main Message**
The model stacks three HGTConv layers. Each layer updates operation embeddings while preserving the 128-dimensional hidden size.

**Visual Design**
- Three repeated blocks:
  - `HGTConv Layer 1`
  - `HGTConv Layer 2`
  - `HGTConv Layer 3`
- Under each block show:
  - `operation [n_o, 128]`
  - `vehicle [n_v, 128]`
  - `zone [n_z, 128]`
- Highlight final output:
  - `H_op^L [n_o, 128]`

**On-slide Bullets**
- Layer 1 aggregates direct typed neighbors.
- Layer 2 propagates higher-order route, lane, conflict, and resource context.
- Layer 3 produces final operation embeddings for decision making.

**Implementation Note**
All current typed relations point to operation nodes, so HGTConv updates operation embeddings. Vehicle and zone embeddings are carried forward as contextual sources.

**Speaker Notes**
The hidden dimension does not grow across layers. The representation becomes richer because each layer mixes more typed neighborhood information.

---

## Slide 7 — Actor: Masked Dispatching Policy

**Title**
Actor Head: Operation Scores with Hard Feasibility Mask

**Main Message**
The actor maps each final operation embedding to one logit and applies a hard feasible-action mask before softmax.

**Visual Design**
- Input: `H_op^L [n_o, 128]`
- MLP block: `128 -> 64 -> 1`
- Output logits: `[n_o]`
- Mask block:
  - feasible actions remain
  - infeasible actions become `-inf`
- Softmax over feasible operations.

**On-slide Formula**
```text
s_o = MLP_actor(h_o^L)
l_o = s_o if o in A_t, else -inf
pi(o | G_t) = softmax(l_o)
```

**Speaker Notes**
The actor learns preferences, but it does not guarantee feasibility by itself. The environment computes the feasible set, and the mask prevents illegal operations from being selected.

---

## Slide 8 — Critic: Graph-level Value Estimate

**Title**
Critic Head: Value from Unscheduled Operation Pooling

**Main Message**
The critic estimates state value by mean-pooling final embeddings of unscheduled operations.

**Visual Design**
- Show final operation embeddings.
- Select unscheduled operations only.
- Mean pooling block:
  - `[n_unscheduled, 128] -> [128]`
- Critic MLP:
  - `128 -> 64 -> 1`

**On-slide Formula**
```text
h_G = mean({h_o^L | o in U_t})
V(G_t) = MLP_critic(h_G)
```

**Speaker Notes**
The critic compresses the remaining scheduling problem into one graph-level value. This value is used by PPO to compute advantages and stabilize policy updates.

---

## Slide 9 — Methodological Separation

**Title**
What Each Component Is Responsible For

**Main Message**
The neural policy, feasibility mask, environment decoder, and low-level controller have separate roles.

**Visual Design**
- Four columns:
  1. HGT Encoder + Actor
  2. Feasibility Mask
  3. Environment Decoder
  4. SUMO/TraCI Controller

**On-slide Bullets**
- HGT predicts dispatching preferences over operation nodes.
- The mask enforces JSSP and traffic feasibility constraints.
- The environment computes exact start and finish times.
- The low-level controller tracks the resulting schedule.

**Speaker Notes**
This separation is important for the methodology section. The model is not replacing the constraint system; it is learning how to prioritize valid dispatching choices.

---

## Suggested Visual Style

**Color Encoding**
- Operation nodes: blue
- Vehicle nodes: green
- Zone nodes: orange
- HGTConv blocks: purple
- Actor: teal
- Critic: brown/gold
- Infeasible mask: red or gray

**Layout Recommendation**
- Use a left-to-right flow for the whole architecture.
- Use repeated tensor-shape labels consistently.
- Keep Q/K/V slides more mathematical and actor-critic slides more decision-oriented.

**One-slide Summary Version**
```text
Traffic state
  -> typed graph: operation [6], vehicle [4], zone [5]
  -> type-specific projection to 128 dimensions
  -> 3 HGTConv layers with 4-head Q/K/V attention
  -> actor: 128 -> 64 -> 1 logits + feasible mask
  -> critic: unscheduled operation mean pooling -> value
```
