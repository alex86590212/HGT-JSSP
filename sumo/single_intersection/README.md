# SUMO Single Unsignalized Intersection

This folder contains a simple four-way, signal-free SUMO network for later
mapping of vehicle routes to manually defined conflict-zone sequences.

## Network Structure

- `J0` is the only intersection node and has no traffic light.
- `J0` is `unregulated`, so SUMO does not impose a right-of-way rule after the
  external scheduler releases a vehicle.
- Each cardinal approach has one inbound edge and one outbound edge:
  `N_in`/`N_out`, `S_in`/`S_out`, `E_in`/`E_out`, `W_in`/`W_out`.
- Inbound roads are about `200` m long.
- Outbound roads are about `100` m long.
- `additional.add.xml` adds optional E1 detectors on inbound lanes about 50 m
  before the intersection.

## Route Naming

Route IDs use `<origin>_to_<destination>`:

- Straight: `N_to_S`, `S_to_N`, `E_to_W`, `W_to_E`
- Left turns: `N_to_E`, `S_to_W`, `E_to_S`, `W_to_N`
- Right turns: `N_to_W`, `S_to_E`, `E_to_N`, `W_to_S`

This naming is intentionally simple so an external scheduler can map a SUMO
route ID directly to a conflict-zone sequence.

## Files

- `single_intersection.nod.xml`: plain node definitions
- `single_intersection.edg.xml`: plain edge definitions
- `single_intersection.con.xml`: allowed turning connections
- `single_intersection.net.xml`: compiled SUMO network
- `routes.rou.xml`: vehicle types and static route definitions
- `demand.rou.xml`: random deterministic demand (`seed=42`,
  `vehicles=80`)
- `hard_demand.rou.xml`: bursty stress-test demand (`vehicles=160`)
- `additional.add.xml`: optional inbound lane detectors
- `single_intersection.sumocfg`: runnable SUMO configuration
- `single_intersection_hard.sumocfg`: denser hard scenario for scheduler tests

## Controller Stop Line

The TraCI controllers stop held vehicles close to the intersection by default,
about 8 m before the end of each inbound lane. The hold command is still issued
farther upstream so vehicles have enough room to brake. Because vehicles are
held very near the junction, the controller includes a small lane-end fallback:
if a released vehicle is clipped exactly at the inbound lane end, it is moved a
few centimeters onto the open internal link that SUMO reports for its route.

## Hard Scenario

The hard scenario sends repeated four-approach bursts through the intersection,
with more delivery vehicles and trucks than the baseline. It is intended to
produce queues and many Type-2/Type-3 scheduling constraints for testing FCFS
and later PPO-GNN schedulers.

## Run

From this repository root:

```powershell
sumo-gui -c sumo\single_intersection\single_intersection.sumocfg
```

Hard scenario:

```powershell
sumo-gui -c sumo\single_intersection\single_intersection_hard.sumocfg
```

Hard scenario with the live FCFS/JSSP debugger:

```powershell
.venv\Scripts\python.exe scripts\run_visual_debug_controller.py --config sumo\single_intersection\single_intersection_hard.sumocfg
```

Or from this folder:

```powershell
sumo-gui -c single_intersection.sumocfg
```

Regenerate the files with a different seed or demand size:

```powershell
python scripts\generate_single_intersection.py --seed 7 --vehicle-count 200
```

Regenerate a larger hard scenario:

```powershell
python scripts\generate_single_intersection.py --hard-vehicle-count 240
```

## TraCI Start Example

```python
import traci

traci.start([
    "sumo-gui",
    "-c",
    "sumo/single_intersection/single_intersection.sumocfg",
])
```
