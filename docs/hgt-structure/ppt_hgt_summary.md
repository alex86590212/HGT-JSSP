# PPT Mini Template: HGT Structure

## Slide Title
HGT-based Neural Scheduler for JSSP Intersection Control

## Diagram Asset
Use `hgt_architecture_flow.html` as the one-slide architecture diagram. Open it in a browser and capture the SVG flow for PPT.

## One-sentence Message
The scheduler represents the current traffic state as a heterogeneous graph and uses HGT to score which operation should be dispatched next.

## Visual Layout
- Left: heterogeneous graph input
  - operation nodes `[6]`: dispatchable vehicle-zone operations
  - vehicle nodes `[4]`: arrival, route length, velocity, progress
  - zone nodes `[5]`: occupancy, free time, competition, position
- Middle: HGT encoder
  - typed relations: `seq`, `lane`, `conflict`, `owns`, `hosts`
  - node-type input projection + stacked HGTConv layers
- Right: actor-critic output
  - actor: operation logits + hard feasible-action mask
  - critic: mean-pool unscheduled operation embeddings for value estimate

## Speaker Notes
- HGT learns dispatching preference, not feasibility by itself.
- Feasibility is computed by the environment and enforced by masking invalid logits.
- The environment decoder computes exact start and finish times after an action is selected.
- Current code constructs edge attributes, but the HGT forward pass uses edge type identity and node features rather than explicit edge-attribute attention bias.

## 3 Bullet Version
- Environment state is encoded as a typed graph: operation, vehicle, and zone nodes.
- HGTConv performs relation-aware message passing over `seq`, `lane`, `conflict`, `owns`, and `hosts` edges.
- The actor predicts dispatching preference; the mask enforces legal actions; the critic estimates PPO value from unscheduled operations.


