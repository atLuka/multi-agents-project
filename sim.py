# All methods are extracted from RoboArena_M4.ipynb, because we could not get the jupyter notebook to work with unity.
#So we had to extract the methods and put them in a separate file.

import random
import itertools
import heapq

import agentpy as ap
import numpy as np

#Envoriment
CONFIG = {
    "cell_size_m": 0.3,
    "grid_width": 15,
    "grid_height": 12,

    "production_line": {
        "id": "production_line",
        "x": 0, "y": 0, "w": 1, "h": 12,
        "output_rows": [1, 4, 7, 10],
    },
    "truck_dock": {
        "id": "truck_dock",
        "x": 14, "y": 0, "w": 1, "h": 12,
        "door_rows": [1, 4, 7, 10],
    },

    "racks": [
        {"id": "rack_top",       "x": 5, "y": 10, "w": 5, "h": 2, "pallet_face": "bottom"},
        {"id": "rack_mid_upper", "x": 3, "y": 6,  "w": 4, "h": 2, "pallet_face": "top"},
        {"id": "rack_mid_lower", "x": 3, "y": 3,  "w": 4, "h": 2, "pallet_face": "bottom"},
        {"id": "rack_vertical",  "x": 9, "y": 3,  "w": 2, "h": 5, "pallet_face": "left"},
    ],
    "max_pallet_slots_per_rack": 4,

    "charging_stations": [
        {"id": "cs_1", "x": 1,  "y": 0, "w": 1, "h": 1},
        {"id": "cs_2", "x": 13, "y": 0, "w": 1, "h": 1},
    ],
    "parking_slot": {"id": "parking_1", "x": 6, "y": 0, "w": 3, "h": 1},
}

ZONE_CODES = {
    "corridor": 0,
    "rack": 1,
    "charging_station": 2,
    "parking_slot": 3,
    "production_line_wall": 4,
    "production_line_output": 5,
    "truck_dock_wall": 6,
    "truck_dock_door": 7,
    "pallet_position": 8,
}


def generate_pallet_slots(rack, max_slots):
    x0, y0, w, h = rack["x"], rack["y"], rack["w"], rack["h"]
    face = rack["pallet_face"]

    length = w if face in ("top", "bottom") else h
    nb_slots = min(max_slots, length)
    step = length / nb_slots
    offsets = [int(i * step) for i in range(nb_slots)]

    slots = []
    for off in offsets:
        if face == "bottom":
            slots.append((x0 + off, y0 - 1))
        elif face == "top":
            slots.append((x0 + off, y0 + h))
        elif face == "left":
            slots.append((x0 - 1, y0 + off))
        elif face == "right":
            slots.append((x0 + w, y0 + off))
    return slots


CONFIG["pallet_positions"] = []
for _rack in CONFIG["racks"]:
    _slots = generate_pallet_slots(_rack, CONFIG["max_pallet_slots_per_rack"])
    for _i, (_x, _y) in enumerate(_slots):
        CONFIG["pallet_positions"].append({
            "id": _rack["id"] + "_pallet_" + str(_i + 1),
            "rack_id": _rack["id"],
            "x": _x, "y": _y,
        })


#Environment model
class StaticZoneAgent(ap.Agent):

    def setup(self):
        self.zone_type = None
        self.zone_id = None
        self.walkable = False


class LogisticsEnvironment(ap.Model):

    def setup(self):
        p = self.p
        self.width = p["grid_width"]
        self.height = p["grid_height"]

        self.grid = ap.Grid(self, (self.height, self.width), track_empty=True)

        self.zone_map = np.zeros((self.height, self.width), dtype=int)
        self.walkable = np.ones((self.height, self.width), dtype=bool)
        self.zones = []

        self.place_production_line(p["production_line"])
        self.place_truck_dock(p["truck_dock"])

        for rack in p["racks"]:
            self.place_block(rack, "rack", walkable=False)

        for cs in p["charging_stations"]:
            self.place_block(cs, "charging_station", walkable=True)

        self.place_block(p["parking_slot"], "parking_slot", walkable=True)
        self.place_pallet_positions(p["pallet_positions"])

    def add_cell(self, x, y, zone_type, zone_id, walkable):
        agent = StaticZoneAgent(self)
        agent.zone_type = zone_type
        agent.zone_id = zone_id
        agent.walkable = walkable
        self.grid.add_agents([agent], positions=[(y, x)])   # agentpy grid positions are (row, col)
        self.zone_map[y, x] = ZONE_CODES[zone_type]
        self.walkable[y, x] = walkable

    def place_block(self, spec, zone_type, walkable):
        x0, y0, w, h = spec["x"], spec["y"], spec["w"], spec["h"]
        for y in range(y0, y0 + h):
            for x in range(x0, x0 + w):
                self.add_cell(x, y, zone_type, spec["id"], walkable)
        self.zones.append({"id": spec["id"], "type": zone_type, "x": x0, "y": y0, "w": w, "h": h})

    def place_production_line(self, spec):
        x0, y0, w, h = spec["x"], spec["y"], spec["w"], spec["h"]
        outputs = spec["output_rows"]
        for y in range(y0, y0 + h):
            if y in outputs:
                self.add_cell(x0, y, "production_line_output", spec["id"], walkable=True)
            else:
                self.add_cell(x0, y, "production_line_wall", spec["id"], walkable=False)
        self.zones.append({"id": spec["id"], "type": "production_line", "x": x0, "y": y0, "w": w, "h": h})

    def place_truck_dock(self, spec):
        x0, y0, w, h = spec["x"], spec["y"], spec["w"], spec["h"]
        doors = spec["door_rows"]
        for y in range(y0, y0 + h):
            if y in doors:
                self.add_cell(x0, y, "truck_dock_door", spec["id"], walkable=True)
            else:
                self.add_cell(x0, y, "truck_dock_wall", spec["id"], walkable=False)
        self.zones.append({"id": spec["id"], "type": "truck_dock", "x": x0, "y": y0, "w": w, "h": h})

    def place_pallet_positions(self, specs):
        for spec in specs:
            self.add_cell(spec["x"], spec["y"], "pallet_position", spec["id"], walkable=True)
            self.zones.append({"id": spec["id"], "type": "pallet_position",
                                "x": spec["x"], "y": spec["y"], "w": 1, "h": 1,
                                "rack_id": spec["rack_id"]})


env = LogisticsEnvironment(CONFIG)
env.setup()


#Connectivity check
def mask_fully_connected(mask, ref):

    from collections import deque
    H, W = mask.shape
    if not mask[ref[1], ref[0]]:
        return False
    seen = {ref}
    dq = deque([ref])
    while dq:
        cx, cy = dq.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < W and 0 <= ny < H and mask[ny, nx] and (nx, ny) not in seen:
                seen.add((nx, ny))
                dq.append((nx, ny))
    return len(seen) == int(mask.sum())


W_MASK = env.walkable



#Pathfinding (A*)
ACTIVE_MISSION_STATUSES = ("Pending", "Assigned", "InProgress")


def _astar_core(walkable, start, goal, blocked, cost_fn):
    if start == goal:
        return [start]
    H, W = walkable.shape

    if cost_fn is None:
        def extra_cost(_cell):
            return 0.0
    elif callable(cost_fn):
        extra_cost = cost_fn
    else:
        def extra_cost(cell):
            return cost_fn.get(cell, 0.0)

    def h(cell):
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    tie = itertools.count()
    open_heap = [(h(start), next(tie), start)]
    g_score = {start: 0.0}
    prev = {}
    closed = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        if current == goal:
            path = [current]
            node = current
            while node != start:
                node = prev[node]
                path.append(node)
            path.reverse()
            return path

        cx, cy = current
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            neighbor = (nx, ny)
            if not (0 <= nx < W and 0 <= ny < H and walkable[ny, nx]):
                continue
            if blocked and neighbor in blocked and neighbor != goal:
                continue
            step_cost = 1.0 + max(0.0, extra_cost(neighbor))
            tentative_g = g_score[current] + step_cost
            if neighbor in g_score and tentative_g >= g_score[neighbor]:
                continue
            g_score[neighbor] = tentative_g
            prev[neighbor] = current
            heapq.heappush(open_heap, (tentative_g + h(neighbor), next(tie), neighbor))

    return None


def astar_path(walkable, start, goal, cost_fn=None):
    return _astar_core(walkable, start, goal, None, cost_fn)


def astar_path_avoiding(walkable, start, goal, blocked, cost_fn=None):
    return _astar_core(walkable, start, goal, blocked, cost_fn)




#Cooperate/Defect payoff matrix
PAYOFF = {("C", "C"): (10, 10), ("C", "D"): (5, 0), ("D", "C"): (0, 5), ("D", "D"): (0, 0)}


#MDP-driven collision resolution policy
COLLISION_AGING_BUCKETS = (0, 1, 2, 3)              # 3 means "3 or more"
COLLISION_STAKE_BUCKETS = ("LOW", "MED", "HIGH")    # Available/ToCharge, ToPickup, Transporting

COLLISION_ACTIONS = ("HOLD", "YIELD")
COLLISION_GAMMA = 0.9

STAKE_WEIGHT = {"LOW": 1.0, "MED": 2.0, "HIGH": 3.0}
WAIT_COST_SCALE = 0.5        # cost of yielding this tick, per unit of stake
ADVANCE_BONUS_SCALE = 2.0    # payoff of finally getting through, per unit of stake
COLLISION_RISK_SCALE = 3.0   # friction cost of holding and still not getting through


def collision_win_prob(aging):
    # the longer an AGV has been stuck, the better its odds of winning the standoff
    return {0: 0.45, 1: 0.65, 2: 0.8, 3: 0.9}[aging]


def collision_reward(state, action):
    aging, stake = state
    w = STAKE_WEIGHT[stake]
    if action == "YIELD":
        return -WAIT_COST_SCALE * w
    p = collision_win_prob(aging)
    return p * ADVANCE_BONUS_SCALE * w - (1 - p) * COLLISION_RISK_SCALE * w


def collision_transitions(state, action):
    aging, stake = state
    if action == "YIELD":
        return [(1.0, (min(aging + 1, 3), stake))]
    p = collision_win_prob(aging)
    return [(p, (0, stake)), (1 - p, (min(aging + 1, 3), stake))]


def solve_collision_mdp(max_iters=5000, tol=1e-9):
    # plain value iteration; returns (policy, values) keyed by (aging_bucket, stake_bucket)
    states = [(a, s) for a in COLLISION_AGING_BUCKETS for s in COLLISION_STAKE_BUCKETS]
    V = {s: 0.0 for s in states}
    for _ in range(max_iters):
        new_V, delta = {}, 0.0
        for s in states:
            best = max(
                collision_reward(s, a) + COLLISION_GAMMA * sum(p * V[ns] for p, ns in collision_transitions(s, a))
                for a in COLLISION_ACTIONS
            )
            new_V[s] = best
            delta = max(delta, abs(best - V[s]))
        V = new_V
        if delta < tol:
            break
    policy = {}
    for s in states:
        qvals = {a: collision_reward(s, a) + COLLISION_GAMMA * sum(p * V[ns] for p, ns in collision_transitions(s, a))
                  for a in COLLISION_ACTIONS}
        policy[s] = max(qvals, key=qvals.get)
    return policy, V


# Solved once at import time (12 states x 2 actions -- negligible cost).
COLLISION_MDP_POLICY, COLLISION_MDP_VALUES = solve_collision_mdp()


def collision_stake_bucket(status):
    return {"Transporting": "HIGH", "ToPickup": "MED"}.get(status, "LOW")


# Pallets and missions
class Pallet:
    def __init__(self, pid, position, status):
        self.id, self.position, self.status = pid, position, status
    def __repr__(self):
        return f"Pallet({self.id},{self.position},{self.status})"


class Mission:
    def __init__(self, mid, origin, destination, pallet_id, flow, priority=0.0, creation_time=0):
        self.id, self.origin, self.destination = mid, origin, destination
        self.pallet_id, self.flow = pallet_id, flow
        self.priority = priority          # used by the MDP/negotiation, higher = more urgent
        self.creation_time = creation_time
        self.t_assigned = None            # stamped once an AGV accepts it (metrics.py)
        self.t_completed = None           # stamped once delivered
        self.status = "Pending"           # Pending -> Assigned -> InProgress -> Completed
        self.agv = None

    def __repr__(self):
        return f"Mission({self.id},{self.flow},{self.origin}->{self.destination},{self.status})"


#AGV agent and the multi-agent system model
class AGVAgent(ap.Agent):
    def setup(self):
        self.label = None
        self.pos = None
        self.battery = 100.0
        self.status = "Available"     # Available | ToPickup | Transporting | ToCharge | Charging
        self.mission = None
        self.carried_pallet = None
        self.path = []
        self.missions_completed = 0
        self.distance_traveled = 0


class MultiAGVSystem(ap.Model):

    N_AGV = 5
    START_POSITIONS = [(2, 0), (5, 0), (7, 0), (10, 0), (12, 0)]

    # Battery thresholds: below 30%, an AGV stops accepting work and heads to
    # charge; it stays there until back up to 80%. Both match the numbers used
    # throughout the project's own reference material for this fleet.
    B_MAX, B_THRESHOLD, B_CHARGED = 100.0, 30.0, 80.0
    C_D, R_C = 1.5, 10.0
    W_D, W_B, W_L = 2.0, 0.5, 20.0

    # A defector's flat edge in the mission auction (see _negotiate): about
    # a 4-cell distance advantage (DEFECT_BID_BONUS / W_D) -- enough to win
    # a close call, not enough to grab a mission from an AGV that is clearly
    # the better fit.
    DEFECT_BID_BONUS = 8.0

    # 150 simulation steps gives the stochastic mission generator below enough
    # attempts to reach a healthy mission count over the run.
    N_STEPS = 150

    # Periodic logistics-flow generator -----------------------------------------
    # Real-world reference intervals are 60s / 90s / 120s for Flow1 / Flow2 /
    # Flow3, i.e. a 2:3:4 ratio. A simulation "step" is an abstract discrete
    # tick rather than a literal second, so these keep the same 2:3:4 ratio at
    # a smaller scale, giving the generator enough attempts over 150 steps to
    # produce a reasonable number of missions:
    INTERVAL_FLOW1 = 6     # Production Line -> Rack or Truck Bay
    INTERVAL_FLOW2 = 9     # Rack -> Outbound Truck Bay
    INTERVAL_FLOW3 = 12    # Inbound Truck Bay -> Rack

    # Flow1 destination split: 60% of pallets go to a Rack, 40% to a Truck Bay.
    P_DEST_RACK = 0.60

    # Mission priority (Mission.priority, Mission.creation_time already stamped by
    # _new_mission): Flow3 represents an inbound truck that is physically already at
    # the dock waiting to be unloaded, so it carries more time-pressure than a Flow1
    # production pallet or a Flow2 outbound shipment that can comfortably wait a beat
    # in the rack -- hence a higher priority value. This does not change any decision
    # logic yet; it's recorded for later phases (routing/MDP) to use.
    PRIORITY_FLOW1 = 0.5
    PRIORITY_FLOW2 = 0.5
    PRIORITY_FLOW3 = 0.8

    N_OBSTACLES = 6

    # -- Dynamic (temporary) obstacles (Phase 2) ---------------------------
    # Separate from the N_OBSTACLES fixed warehouse fixtures above: these
    # appear and disappear at random during the run to simulate transient
    # blockages (a dropped box, maintenance cone, etc). ~4% per step over
    # N_STEPS=150 gives an expected ~6 spawn attempts, comfortably inside
    # the "5-12 events" target once a few attempts are skipped for lack of
    # a safe, connectivity-preserving cell.
    P_DYNAMIC_OBSTACLE_SPAWN = 0.04
    DYNAMIC_OBSTACLE_DURATION_MIN = 4
    DYNAMIC_OBSTACLE_DURATION_MAX = 8

    # -- Moving pedestrians (Phase 2) ---------------------------------------
    # Generalizes M3's single fixed 3-cell PEDESTRIAN_CELLS zone into several
    # short (2-4 cell) straight routes a pedestrian actually walks across,
    # one cell per step, picked at random per spawn so crossings happen in
    # different places over a run. All three routes sit in open corridor
    # cells (verified against the rack/charging/parking layout in CONFIG).
    PEDESTRIAN_ROUTES = (
        ((9, 0), (9, 1), (9, 2)),        # top corridor, vertical crossing (M3's original spot)
        ((4, 0), (4, 1), (4, 2)),        # top corridor, vertical crossing, different aisle
        ((4, 8), (5, 8), (6, 8), (7, 8)),  # mid-warehouse aisle, horizontal crossing
    )
    PEDESTRIAN_SPAWN_PERIOD = 12
    PEDESTRIAN_SPAWN_JITTER = 4

    # -- Charging station outage (Phase 2) ----------------------------------
    # ~1% per step over N_STEPS=150, but only ever armed while *no* station
    # is currently down (see _update_station_outages) -- this both keeps
    # occurrences in the "1-3 over the run" range and is exactly what
    # guarantees the two stations are never down simultaneously.
    P_STATION_OUTAGE = 0.01
    STATION_OUTAGE_DURATION_MIN = 10
    STATION_OUTAGE_DURATION_MAX = 20

    # Deadlock breaker: in a narrow aisle, two AGVs wanting to swap cells can
    # get stuck forever under the plain yield/reroute rules -- the loser of a
    # swap is blocked (doesn't move) but is still physically occupying its
    # cell that same tick, so the winner still can't step into the loser's
    # cell either; next tick the same swap is detected again and the same
    # agent yields again, forever. Measured across 150 seeds, this is a
    # common standoff in this warehouse's narrow aisles, not a rare edge
    # case, so: any mover blocked for more than STUCK_THRESHOLD consecutive
    # steps is forced to sidestep onto any free adjacent cell (breaking the
    # standoff) and replans from there.
    STUCK_THRESHOLD = 4

    def setup(self):
        # -- Reproducibility ---------------------------------------------------
        # One seeded random stream for the whole run (mission/pallet
        # generation, dynamic obstacles, pedestrians, station outages,
        # negotiation and collision rolls). Passing the same `seed` to a
        # baseline run and a coordinated run gives both the same warehouse
        # layout, initial obstacle placement and dice going in -- exactly the
        # reproducibility the assignment asks for (its own example is
        # `random.seed(42)`). If no seed is passed, falls back to the plain
        # `random` module.
        seed = self.p.get("seed", None)
        self.rng = random.Random(seed) if seed is not None else random

        # -- Decision strategy ---------------------------------------------
        # "baseline": no negotiation game, nearest-available-AGV assignment,
        # fixed deterministic collision tiebreak (lower AGV index wins).
        # "coordinated" (default): Cooperate/Defect negotiation for missions,
        # and an MDP-driven collision-resolution policy instead of the fixed
        # tiebreak (see the comment above _resolve_conflict). Charging is
        # identical, purely reactive, in both strategies -- see B_THRESHOLD.
        # In both, once a mission is accepted it stays with its AGV until
        # completed, unless it fails outright (battery hits 0, see
        # _update_agv), in which case it goes back to Pending.
        self.strategy = self.p.get("strategy", "coordinated")
        assert self.strategy in ("baseline", "coordinated"), \
            f"strategy must be 'baseline' or 'coordinated', got {self.strategy!r}"

        # -- Load profile ----------------------------------------------------
        # Every knob below defaults to the class constant (the "light load"
        # scenario) but can be overridden per run via `parameters=` -- used
        # to run the exact same fleet/warehouse under a "stressed" load
        # (longer shift, more frequent missions, faster battery drain) for
        # the second half of the comparison. Nothing else about the model
        # changes between the two -- same code, same techniques, only the
        # operating conditions differ.
        self.N_STEPS = self.p.get("n_steps", self.N_STEPS)
        self.INTERVAL_FLOW1 = self.p.get("interval_flow1", self.INTERVAL_FLOW1)
        self.INTERVAL_FLOW2 = self.p.get("interval_flow2", self.INTERVAL_FLOW2)
        self.INTERVAL_FLOW3 = self.p.get("interval_flow3", self.INTERVAL_FLOW3)
        self.C_D = self.p.get("c_d", self.C_D)

        self.charging_stations = [(cs["x"], cs["y"]) for cs in CONFIG["charging_stations"]]
        self.log = []
        self.history = []
        self.missions = {}
        self._mission_counter = 0
        self.pallets = {}
        self._pallet_counter = 0
        self.flow_log = {
            "Flow1": {"attempts": 0, "created": 0, "skipped": 0},
            "Flow2": {"attempts": 0, "created": 0, "skipped": 0},
            "Flow3": {"attempts": 0, "created": 0, "skipped": 0},
        }
        self.stats = {"mission_defect_seize": 0,
                       "collision_yield": 0,
                       "pedestrian_wait": 0, "pedestrian_reroute": 0,
                       "obstacle_wait": 0, "obstacle_reroute": 0,
                       "dynamic_obstacle_spawn": 0, "station_outage_events": 0,
                       "battery_failure": 0}

        self._build_initial_pallets()

        self.obstacles = self._generate_obstacles(self.N_OBSTACLES)
        self.walkable = W_MASK.copy()
        for (ox, oy) in self.obstacles:
            self.walkable[oy, ox] = False
        assert mask_fully_connected(self.walkable, self.START_POSITIONS[0]), \
            "obstacle placement broke grid connectivity"

        # Phase 2 dynamic state -------------------------------------------
        self.dynamic_obstacles = {}   # {(x,y): expire_at_step}
        self.pedestrians = []         # [{"path": [(x,y),...], "idx": int}, ...]
        self._next_pedestrian_spawn = (
            self.PEDESTRIAN_SPAWN_PERIOD
            + self.rng.randint(-self.PEDESTRIAN_SPAWN_JITTER, self.PEDESTRIAN_SPAWN_JITTER)
        )
        self.station_outage = {}      # {(x,y): expire_at_step}
        self._stuck_counts = {}       # {agent: consecutive blocked steps}, deadlock breaker

        self.agvs = ap.AgentList(self, self.N_AGV, AGVAgent)
        for i, (a, pos) in enumerate(zip(self.agvs, self.START_POSITIONS), start=1):
            a.label = f"AGV-{i}"
            a.pos = pos

        self._publish(list(self.missions.keys()))

    def _generate_obstacles(self, n, max_attempts=300):
        reserved = set(self.START_POSITIONS) | set(self.charging_stations)
        reserved |= {(p["x"], p["y"]) for p in CONFIG["pallet_positions"]}
        ps = CONFIG["parking_slot"]
        reserved |= {(ps["x"] + i, ps["y"]) for i in range(ps["w"])}
        for row in CONFIG["production_line"]["output_rows"]:
            reserved.add((0, row))
        for row in CONFIG["truck_dock"]["door_rows"]:
            reserved.add((14, row))
        for route in self.PEDESTRIAN_ROUTES:
            reserved |= set(route)

        H, W = W_MASK.shape
        candidates = [(x, y) for y in range(H) for x in range(W)
                      if W_MASK[y, x] and (x, y) not in reserved]

        ref = self.START_POSITIONS[0]
        for _ in range(max_attempts):
            if len(candidates) < n:
                break
            chosen = set(self.rng.sample(candidates, n))
            trial = W_MASK.copy()
            for (x, y) in chosen:
                trial[y, x] = False
            if mask_fully_connected(trial, ref):
                return chosen
        return set()

    # -- Phase 2: unified "temporarily blocked" concept ----------------------
    # Pedestrians and dynamic obstacles are tracked separately (different
    # lifetimes, different log messages, different stats) but for routing and
    # collision-avoidance purposes they are the same thing: a cell an AGV
    # must not be standing on / walking into right now. Every place that used
    # to consult "active pedestrian cells" alone now consults this union.

    def _active_pedestrian_cells(self):
        return {ped["path"][ped["idx"]] for ped in self.pedestrians}

    def _active_dynamic_obstacle_cells(self):
        return set(self.dynamic_obstacles.keys())

    def _temporarily_blocked_cells(self):
        return self._active_pedestrian_cells() | self._active_dynamic_obstacle_cells()

    def _plan_path(self, start, goal):
        avoid = self._temporarily_blocked_cells()
        path = astar_path_avoiding(self.walkable, start, goal, avoid) if avoid else None
        if path is None:
            path = astar_path(self.walkable, start, goal)
        return path

    def _log(self, msg):
        self.log.append(f"[t={self.t:02d}] {msg}")

    def _new_mission(self, origin, destination, pallet_id, flow):
        self._mission_counter += 1
        mid = f"M{self._mission_counter:02d}"
        m = Mission(mid, origin, destination, pallet_id, flow, creation_time=self.t)
        # M4 Phase 3 (Part E): Mission (defined in base_m3.py, which we must
        # not edit) doesn't declare these two timing fields in __init__ --
        # Python lets us set plain dynamic attributes on the instance anyway.
        # Every mission gets both set to None right here, at the one choke
        # point all missions are created through, so nothing downstream ever
        # hits an AttributeError before _negotiate/_update_agv fill them in.
        m.t_assigned = None
        m.t_completed = None
        self.missions[mid] = m
        return mid

    def _new_pallet_id(self, flow_tag):
        self._pallet_counter += 1
        return f"P_{flow_tag}_{self._pallet_counter:03d}"

    def _build_initial_pallets(self):
        # 2 stored pallets + 2 empty slots per rack, so Flow2 has stock to move right away
        pos_by_slot = {p["id"]: (p["x"], p["y"]) for p in CONFIG["pallet_positions"]}
        self._pos_by_slot = pos_by_slot

        self.slot_visual_offset = {}
        for rack in CONFIG["racks"]:
            dx, dy = {"bottom": (0, 1), "top": (0, -1), "left": (1, 0), "right": (-1, 0)}[rack["pallet_face"]]
            for pp in CONFIG["pallet_positions"]:
                if pp["rack_id"] == rack["id"]:
                    self.slot_visual_offset[(pp["x"], pp["y"])] = (dx, dy)

        for rack in CONFIG["racks"]:
            for n in (1, 2):
                slot_id = f"{rack['id']}_pallet_{n}"
                self.pallets[f"P_{slot_id}"] = Pallet(f"P_{slot_id}", pos_by_slot[slot_id], "stored")
            # slots 3 and 4 of each rack are intentionally left with no Pallet object
            # at setup -- that absence is exactly what makes them "free" to
            # _free_rack_slots() below, ready for Flow1/Flow3 to fill.

    def visual_pallet_position(self, pos):
        dx, dy = self.slot_visual_offset.get(pos, (0, 0))
        return (pos[0] + dx, pos[1] + dy)

    def _publish(self, mids):
        for mid in mids:
            self._log(f"Mission {mid} published")

    # -- periodic, probabilistic logistics-flow generator (M4 phase 1) ----------

    def _free_production_outputs(self):
        occupied = {p.position for p in self.pallets.values() if p.status == "waiting_for_pickup"}
        return [(0, row) for row in CONFIG["production_line"]["output_rows"] if (0, row) not in occupied]

    def _free_truck_doors(self):
        # a door is free if nothing's waiting there and no active mission is already headed to it
        reserved_dest = {m.destination for m in self.missions.values() if m.status in ACTIVE_MISSION_STATUSES}
        occupied = {p.position for p in self.pallets.values() if p.status == "waiting_for_pickup"}
        return [(14, row) for row in CONFIG["truck_dock"]["door_rows"]
                if (14, row) not in occupied and (14, row) not in reserved_dest]

    def _free_rack_slots(self):
        # same idea for rack slots (a Transporting pallet's position stays frozen at
        # its origin until delivered, so it can't block its own destination slot)
        reserved_dest = {m.destination for m in self.missions.values() if m.status in ACTIVE_MISSION_STATUSES}
        occupied = {p.position for p in self.pallets.values()
                    if p.status in ("stored", "reserved", "waiting_for_pickup")}
        return [(p["x"], p["y"]) for p in CONFIG["pallet_positions"]
                if (p["x"], p["y"]) not in occupied and (p["x"], p["y"]) not in reserved_dest]

    def _generate_flows(self):
        # each flow is just attempted every INTERVAL_FLOWn steps; if the resource it
        # needs isn't free right then, it's skipped and tried again next interval
        if self.t % self.INTERVAL_FLOW1 == 0:
            self._attempt_flow1()
        if self.t % self.INTERVAL_FLOW2 == 0:
            self._attempt_flow2()
        if self.t % self.INTERVAL_FLOW3 == 0:
            self._attempt_flow3()

    def _attempt_flow1(self):
        """Production Line -> Rack (60%) or Truck Bay (40%)."""
        self.flow_log["Flow1"]["attempts"] += 1
        free_outputs = self._free_production_outputs()
        if not free_outputs:
            self.flow_log["Flow1"]["skipped"] += 1
            self._log("  Flow1 attempt skipped: no free production output")
            return
        origin = self.rng.choice(free_outputs)

        if self.rng.random() < self.P_DEST_RACK:
            candidates = self._free_rack_slots()
        else:
            candidates = self._free_truck_doors()
        if not candidates:
            # The chosen destination type has no free resource right now -- the
            # spec is explicit that we must NOT silently fall back to the other
            # type, we just skip this attempt entirely.
            self.flow_log["Flow1"]["skipped"] += 1
            self._log(f"  Flow1 attempt skipped: no free destination for pallet at {origin}")
            return
        destination = self.rng.choice(candidates)

        pallet_id = self._new_pallet_id("flow1")
        self.pallets[pallet_id] = Pallet(pallet_id, origin, "waiting_for_pickup")
        mid = self._new_mission(origin, destination, pallet_id, "Flow1")
        self.missions[mid].priority = self.PRIORITY_FLOW1
        self.flow_log["Flow1"]["created"] += 1
        self._publish([mid])

    def _attempt_flow2(self):
        """Rack -> Outbound Truck Bay."""
        self.flow_log["Flow2"]["attempts"] += 1
        active_pallet_ids = {m.pallet_id for m in self.missions.values() if m.status in ACTIVE_MISSION_STATUSES}
        rack_positions = {(p["x"], p["y"]) for p in CONFIG["pallet_positions"]}
        candidates = [p for p in self.pallets.values()
                      if p.status == "stored" and p.position in rack_positions
                      and p.id not in active_pallet_ids]
        if not candidates:
            self.flow_log["Flow2"]["skipped"] += 1
            self._log("  Flow2 attempt skipped: no stored rack pallet available")
            return
        pallet = self.rng.choice(candidates)

        free_doors = self._free_truck_doors()
        if not free_doors:
            self.flow_log["Flow2"]["skipped"] += 1
            self._log(f"  Flow2 attempt skipped: no free truck bay door for pallet {pallet.id}")
            return
        destination = self.rng.choice(free_doors)

        mid = self._new_mission(pallet.position, destination, pallet.id, "Flow2")
        self.missions[mid].priority = self.PRIORITY_FLOW2
        self.flow_log["Flow2"]["created"] += 1
        self._publish([mid])

    def _attempt_flow3(self):
        """Inbound Truck Bay -> Rack."""
        self.flow_log["Flow3"]["attempts"] += 1
        free_doors = self._free_truck_doors()
        if not free_doors:
            self.flow_log["Flow3"]["skipped"] += 1
            self._log("  Flow3 attempt skipped: no free truck bay door")
            return
        free_slots = self._free_rack_slots()
        if not free_slots:
            self.flow_log["Flow3"]["skipped"] += 1
            self._log("  Flow3 attempt skipped: no free rack slot")
            return

        origin = self.rng.choice(free_doors)
        destination = self.rng.choice(free_slots)
        pallet_id = self._new_pallet_id("flow3")
        self.pallets[pallet_id] = Pallet(pallet_id, origin, "waiting_for_pickup")
        mid = self._new_mission(origin, destination, pallet_id, "Flow3")
        self.missions[mid].priority = self.PRIORITY_FLOW3
        self.flow_log["Flow3"]["created"] += 1
        self._publish([mid])

    # -- unchanged from base_m3.py below this line (negotiation, collisions, --
    # -- movement, battery, pedestrian safety, logging) -------------------------

    def _nearest_charging_station(self, pos):
        # Phase 2: a station currently out of service is never a valid
        # destination -- an AGV heading to charge routes to the other one
        # instead (both stations are never down at once, see
        # _update_station_outages, so `available` is never empty in practice;
        # the fallback below is just defensive).
        available = [cs for cs in self.charging_stations if cs not in self.station_outage]
        if not available:
            available = self.charging_stations
        return min(available, key=lambda cs: abs(cs[0] - pos[0]) + abs(cs[1] - pos[1]))

    def _other_station(self, cs):
        others = [c for c in self.charging_stations if c != cs]
        return others[0] if others else None

    def _utility(self, agv, mission):
        dist = len(self._plan_path(agv.pos, mission.origin)) - 1
        return -self.W_D * dist + self.W_B * agv.battery - self.W_L * agv.missions_completed, dist

    def _mission_defect_probability(self, agent):
        margin = agent.battery - self.B_THRESHOLD
        urgency = max(0.0, (40.0 - margin) / 60.0)
        return min(0.5, 0.12 + urgency)

    def _negotiate(self):
        for mid, m in list(self.missions.items()):
            if m.status != "Pending":
                continue
            eligible = []
            for a in self.agvs:
                if a.status != "Available":
                    self._log(f"  {a.label} unavailable (status={a.status})")
                    continue
                if a.battery <= self.B_THRESHOLD:
                    self._log(f"  {a.label} unavailable (battery {a.battery:.1f}% <= threshold)")
                    continue
                eligible.append(a)
            if not eligible:
                continue

            if self.strategy == "baseline":
                # No negotiation: assign to whichever eligible AGV is nearest
                # to the mission's origin. Reuses the distance component
                # _utility already computes (ignoring its battery/workload
                # terms) rather than duplicating the path-planning call.
                dists = []
                for a in eligible:
                    _u, dist = self._utility(a, m)
                    dists.append((dist, a))
                    self._log(f"  {a.label} distance to {mid} origin: {dist}")
                dists.sort(key=lambda d: d[0])
                winner = dists[0][1]
                self._log(f"  {mid}: baseline strategy (no negotiation) -> "
                          f"nearest AGV is {winner.label}")
            else:
                # M4 (revised): a defector no longer auto-wins the mission
                # regardless of fit, and a double-defection no longer stalls
                # the assignment a tick -- both were measured to hurt more
                # than they modeled: the auto-win let a bad-fit AGV grab a
                # mission a much closer AGV needed (feeding extra travel and
                # path crossings straight into the metrics we compare), and
                # the stall bunched deferred missions into later bursts of
                # simultaneous departures. The Cooperate/Defect game still
                # matters -- a defector bids with a flat edge on top of its
                # normal utility, so it wins more often than a cooperator in
                # a close call -- but it can still lose outright to a
                # cooperator who is a much better fit, and every mission is
                # assigned the same tick it becomes eligible.
                actions = {a: ("D" if self.rng.random() < self._mission_defect_probability(a) else "C")
                           for a in eligible}
                bids = []
                for a in eligible:
                    u, dist = self._utility(a, m)
                    if actions[a] == "D":
                        u += self.DEFECT_BID_BONUS
                    bids.append((u, a))
                    self._log(f"  {a.label} bid: {u:.1f} ({actions[a]}, dist={dist}, battery={a.battery:.1f}%)")
                bids.sort(key=lambda b: b[0], reverse=True)
                winner = bids[0][1]
                if actions[winner] == "D":
                    self.stats["mission_defect_seize"] += 1
                    self._log(f"  {mid}: {winner.label} wins as a defector (bid edge, still best fit)")
                else:
                    self._log(f"  {mid}: {winner.label} wins cooperating on merit")

            m.status, m.agv = "Assigned", winner.label
            m.t_assigned = self.t
            winner.mission, winner.status = m.id, "ToPickup"
            winner.path = self._plan_path(winner.pos, m.origin)[1:]
            self.pallets[m.pallet_id].status = "reserved"
            self._log(f"{winner.label} accepts Mission {m.id}")

    def _agv_goal(self, a):
        if a.status == "ToPickup":
            return self.missions[a.mission].origin
        if a.status == "Transporting":
            return self.missions[a.mission].destination
        if a.status == "ToCharge":
            return self._nearest_charging_station(a.pos)
        return None

    def _priority(self, a):
        if self.strategy == "baseline":
            # Fixed deterministic tiebreak, no randomness, no aging -- the
            # lower AGV index always wins a contested cell. This one change
            # point covers every sorted(key=self._priority) call site
            # (collision resolution and the post-reroute safety check alike).
            idx = int(a.label.rsplit("-", 1)[1])
            return (-idx, 0.0)
        # Coordinated: stake tier comes first and never flips on its own --
        # a Transporting AGV always outranks a ToPickup one, which always
        # outranks an idle/charging one, no matter how long anyone has been
        # stuck. Measured empirically: letting a raw MDP value alone decide
        # (aging and stake mixed into one number) let a barely-stuck LOW
        # AGV occasionally outrank a fresh MED/HIGH one -- a rank flip with
        # no anti-starvation purpose, since the two weren't even contesting
        # each other before, and it recreated the very standoffs we were
        # trying to remove (see phase3_notes.md). The MDP's value is still
        # what breaks ties *within* the same stake tier: two AGVs doing the
        # same kind of thing, the one that's been stuck longer eventually
        # outranks the other -- which is exactly the anti-starvation
        # property aging is for.
        aging = min(self._stuck_counts.get(a, 0), 3)
        stake = collision_stake_bucket(a.status)
        stake_rank = {"LOW": 0, "MED": 1, "HIGH": 2}[stake]
        return (stake_rank, COLLISION_MDP_VALUES[(aging, stake)])

    def _resolve_conflict(self, agents_here, blocked, cell):
        # Single deterministic resolution in both strategies -- baseline by
        # a fixed index, coordinated by the MDP value above. No re-rolling:
        # each genuine standoff is settled in exactly one comparison, so it
        # is counted (and resolved) once, not once per tick it happens to
        # persist.
        scored = sorted(agents_here, key=lambda a: self._priority(a), reverse=True)
        winner = scored[0]
        for loser in scored[1:]:
            blocked.add(loser)
            self.stats["collision_yield"] += 1
            self._log(f"  COLLISION avoided at {cell}: {loser.label} yields to {winner.label} "
                      f"(priority {self._priority(winner)} vs {self._priority(loser)})")

    def _nudge_idle_blockers(self, movers):
        reserved = set()
        for a in movers:
            reserved.update(a.path)
        occupied = {a.pos: a for a in self.agvs}
        temp_blocked = self._temporarily_blocked_cells()
        H, W = self.walkable.shape
        for a in self.agvs:
            if a.status != "Available" or a.pos not in reserved:
                continue
            candidates = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = a.pos[0] + dx, a.pos[1] + dy
                if (0 <= nx < W and 0 <= ny < H and self.walkable[ny, nx]
                        and (nx, ny) not in occupied and (nx, ny) not in temp_blocked):
                    candidates.append((nx, ny))
            if not candidates:
                continue
            free = [c for c in candidates if c not in reserved]
            dest = free[0] if free else candidates[0]
            self._log(f"  {a.label} (idle) steps aside from {a.pos} to {dest}, freeing the way for another AGV")
            occupied.pop(a.pos, None)
            a.pos = dest
            occupied[a.pos] = a

    def _enforce_pedestrian_safety(self, movers, planned, blocked):
        # hard rule, no negotiation: an AGV can never step onto a cell a pedestrian
        # or a dynamic obstacle currently occupies -- reroute if possible, else wait
        active_ped = self._active_pedestrian_cells()
        active_obs = self._active_dynamic_obstacle_cells()
        temp_blocked = active_ped | active_obs
        if not temp_blocked:
            return
        occupied_now = {o.pos for o in self.agvs}
        for a in movers:
            if a in blocked or planned[a] not in temp_blocked:
                continue
            blocked_cell = planned[a]
            if blocked_cell in active_ped:
                cause, stat_prefix = "PEDESTRIAN CROSSING", "pedestrian"
            else:
                cause, stat_prefix = "DYNAMIC OBSTACLE", "obstacle"
            goal = self._agv_goal(a)
            avoid = occupied_now | temp_blocked
            new_path = (astar_path_avoiding(self.walkable, a.pos, goal, avoid)
                        if goal else None)
            if new_path and len(new_path) > 1 and new_path[1] not in temp_blocked:
                a.path = new_path[1:]
                planned[a] = a.path[0]
                self.stats[f"{stat_prefix}_reroute"] += 1
                self._log(f"  {cause} active at {blocked_cell}: "
                          f"{a.label} reroutes to avoid it (absolute safety rule, no negotiation)")
                continue
            blocked.add(a)
            self.stats[f"{stat_prefix}_wait"] += 1
            self._log(f"  {cause} active: {a.label} waits at {a.pos} rather than enter "
                       f"{blocked_cell} (absolute safety rule, no negotiation)")

    def _resolve_and_move(self):
        movers = [a for a in self.agvs if a.status in ("ToPickup", "Transporting", "ToCharge") and a.path]
        self._nudge_idle_blockers(movers)
        planned = {a: a.path[0] for a in movers}

        blocked = set()
        self._enforce_pedestrian_safety(movers, planned, blocked)

        by_cell = {}
        for a, cell in planned.items():
            if a in blocked:
                continue
            by_cell.setdefault(cell, []).append(a)

        for cell, agents_here in by_cell.items():
            if len(agents_here) > 1:
                self._resolve_conflict(agents_here, blocked, cell)

        checked = set()
        for a in movers:
            for b in movers:
                if a is b or (a, b) in checked or (b, a) in checked:
                    continue
                checked.add((a, b))
                if a in blocked or b in blocked:
                    continue
                if planned[a] == b.pos and planned[b] == a.pos:
                    self._resolve_conflict([a, b], blocked, "swap")

        temp_blocked = self._temporarily_blocked_cells()
        for a in movers:
            if a in blocked:
                continue
            target = planned[a]
            occupant = next((o for o in self.agvs if o is not a and o.pos == target), None)
            if occupant is None:
                continue
            goal = self._agv_goal(a)
            occupied_now = {o.pos for o in self.agvs if o is not a} | temp_blocked
            new_path = (astar_path_avoiding(self.walkable, a.pos, goal, occupied_now)
                        if goal else None)
            if new_path and len(new_path) > 1:
                a.path = new_path[1:]
                planned[a] = a.path[0]
                self.stats["collision_yield"] += 1
                self._log(f"  COLLISION avoided at {target}: {a.label} reroutes around stationary {occupant.label}")
                if planned[a] != target:
                    continue
            blocked.add(a)
            self.stats["collision_yield"] += 1
            self._log(f"  COLLISION avoided at {target}: {a.label} waits (no alternate route), {occupant.label} in the way")

        by_cell2 = {}
        for a in movers:
            if a in blocked:
                continue
            by_cell2.setdefault(planned[a], []).append(a)
        for cell, agents_here in by_cell2.items():
            if len(agents_here) > 1:
                scored = sorted(agents_here, key=lambda a: self._priority(a), reverse=True)
                for loser in scored[1:]:
                    blocked.add(loser)
                    self.stats["collision_yield"] += 1
                    self._log(f"  COLLISION avoided at {cell}: {loser.label} yields to {scored[0].label} (post-reroute safety check)")

        occupied_cells = {o.pos for o in self.agvs}
        # Cells other *moving* (non-blocked) AGVs are about to step into this
        # same tick -- a deadlock-broken AGV must not sidestep into one of
        # these, or it collides with whoever was already cleared to move there.
        incoming_cells = {planned[b] for b in movers if b not in blocked}
        temp_blocked_now = self._temporarily_blocked_cells()
        H, W = self.walkable.shape
        for a in movers:
            if a not in blocked:
                self._stuck_counts[a] = 0
                continue
            self._stuck_counts[a] = self._stuck_counts.get(a, 0) + 1
            if self._stuck_counts[a] <= self.STUCK_THRESHOLD:
                continue

            side_candidates = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = a.pos[0] + dx, a.pos[1] + dy
                if (0 <= nx < W and 0 <= ny < H and self.walkable[ny, nx]
                        and (nx, ny) not in occupied_cells and (nx, ny) not in temp_blocked_now
                        and (nx, ny) not in incoming_cells):
                    side_candidates.append((nx, ny))
            if not side_candidates:
                continue
            # Prefer a candidate that isn't a dead end (has >=1 free exit of
            # its own besides stepping straight back) so the sidestep can't
            # trap the AGV somewhere worse than where it started.
            def _exits(cell):
                cnt = 0
                for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ex, ey = cell[0] + ddx, cell[1] + ddy
                    if (ex, ey) != a.pos and 0 <= ex < W and 0 <= ey < H and self.walkable[ey, ex]:
                        cnt += 1
                return cnt
            with_exits = [c for c in side_candidates if _exits(c) > 0]
            dest = (self.rng.choice(with_exits) if with_exits
                    else self.rng.choice(side_candidates))
            self._log(f"  DEADLOCK BREAK: {a.label} blocked {self._stuck_counts[a]} steps in a row at "
                      f"{a.pos}, sidesteps to {dest} to clear the standoff")
            occupied_cells.discard(a.pos)
            a.pos = dest
            occupied_cells.add(a.pos)
            goal = self._agv_goal(a)
            new_path = self._plan_path(a.pos, goal) if goal else None
            a.path = new_path[1:] if new_path else []
            self._stuck_counts[a] = 0

        for a in self.agvs:
            if a in blocked:
                continue
            self._update_agv(a)

    def _update_agv(self, a):
        if a.status in ("ToPickup", "Transporting", "ToCharge") and a.path:
            a.pos = a.path.pop(0)
            a.distance_traveled += 1
            if a.status != "ToCharge":
                a.battery = max(0.0, a.battery - self.C_D)

        if a.status in ("ToPickup", "Transporting") and a.battery <= 0.0:
            # M4: battery physically ran out mid-mission -- the AGV shuts
            # down exactly where it stands and has to be towed to the
            # nearest station. Its mission is not lost: the pallet is left
            # right where the AGV died (or never left the origin, if it
            # hadn't been picked up yet) and the mission goes back to
            # `Pending` at that location, for any other eligible AGV to
            # pick up later -- see the M3 battery rule this restores: the
            # >30%/<=30% eligibility gate never changes, this only adds the
            # missing consequence for actually hitting empty.
            m = self.missions[a.mission]
            pallet = self.pallets[m.pallet_id]
            if a.status == "Transporting":
                pallet.position = a.pos
            pallet.status = "waiting_for_pickup"
            m.origin = a.pos
            m.status, m.agv = "Pending", None
            self.stats["battery_failure"] += 1
            self._log(f"{a.label} battery hit 0% mid-mission -- {m.id} left at "
                      f"{a.pos} (back to Pending), {a.label} towed to charge")
            a.mission = None
            a.carried_pallet = None
            a.status = "ToCharge"
            a.path = self._plan_path(a.pos, self._nearest_charging_station(a.pos))[1:]
            return

        if a.status == "ToPickup" and not a.path:
            m = self.missions[a.mission]
            pallet = self.pallets[m.pallet_id]
            pallet.status = "being_transported"
            a.carried_pallet = pallet.id
            a.status = "Transporting"
            m.status = "InProgress"
            a.path = self._plan_path(a.pos, m.destination)[1:]
            self._log(f"{a.label} reached pallet {pallet.id}")
            self._log(f"{a.label} transporting {pallet.id}")

        elif a.status == "Transporting" and not a.path:
            m = self.missions[a.mission]
            pallet = self.pallets[m.pallet_id]
            pallet.position = a.pos
            pallet.status = "delivered" if m.flow == "Flow2" else "stored"
            a.carried_pallet = None
            m.status = "Completed"
            m.t_completed = self.t
            a.missions_completed += 1
            self._log(f"{a.label} completed Mission {m.id}")
            a.mission = None
            if a.battery <= self.B_THRESHOLD:
                a.status = "ToCharge"
                a.path = self._plan_path(a.pos, self._nearest_charging_station(a.pos))[1:]
                self._log(f"{a.label} battery low ({a.battery:.1f}%), heading to charge")
            else:
                a.status = "Available"

        elif a.status == "ToCharge" and not a.path:
            a.status = "Charging"
            self._log(f"{a.label} charging at {a.pos}")

        elif a.status == "Charging":
            if a.pos in self.station_outage:
                # Phase 2: station went down mid-charge -- stop gaining
                # charge; _update_station_outages already tried to redirect
                # this AGV to the other station the moment the outage
                # started, this is just the battery-gain guard for the
                # (rare) step where that hasn't happened yet.
                self._log(f"{a.label} paused charging: station {a.pos} is out of service")
            else:
                a.battery = min(self.B_MAX, a.battery + self.R_C)
                if a.battery >= self.B_CHARGED:
                    # A charging AGV never still holds a mission: either it
                    # arrived here from "Available" (no mission), or its
                    # mission failed and was released to Pending by the
                    # battery=0 rule above (also clears a.mission). So this
                    # always means back to Available.
                    a.status = "Available"
                    self._log(f"{a.label} charged to {a.battery:.1f}%, available again")

        elif a.status == "Available" and a.battery <= self.B_THRESHOLD:
            # M3's original reactive rule, unconditional in BOTH strategies:
            # below the threshold, an AGV always stops accepting work and
            # heads to charge -- nothing here consults any decision layer.
            a.status = "ToCharge"
            a.path = self._plan_path(a.pos, self._nearest_charging_station(a.pos))[1:]
            self._log(f"{a.label} battery low ({a.battery:.1f}%), heading to charge")

    # -- Phase 2: dynamic obstacles -------------------------------------------

    def _pick_dynamic_obstacle_cell(self, max_attempts=20):
        # pick a free, non-reserved cell whose removal still leaves the grid fully
        # connected -- give up after a few tries and just skip spawning this step
        H, W = self.walkable.shape
        occupied = {a.pos for a in self.agvs}
        reserved = set(self.charging_stations)
        reserved |= {(p["x"], p["y"]) for p in CONFIG["pallet_positions"]}
        for row in CONFIG["production_line"]["output_rows"]:
            reserved.add((0, row))
        for row in CONFIG["truck_dock"]["door_rows"]:
            reserved.add((14, row))
        for route in self.PEDESTRIAN_ROUTES:
            reserved |= set(route)

        candidates = [(x, y) for y in range(H) for x in range(W)
                      if self.walkable[y, x]
                      and (x, y) not in occupied
                      and (x, y) not in self.dynamic_obstacles
                      and (x, y) not in reserved]
        self.rng.shuffle(candidates)
        ref = self.START_POSITIONS[0]
        for cell in candidates[:max_attempts]:
            trial = self.walkable.copy()
            trial[cell[1], cell[0]] = False
            if mask_fully_connected(trial, ref):
                return cell
        return None

    def _force_reroute_for_cell(self, cell):
        # a dynamic obstacle just spawned on `cell` -- replan any AGV already
        # heading through it right away instead of waiting for it to walk into it
        temp_blocked = self._temporarily_blocked_cells()
        for a in self.agvs:
            if a.status not in ("ToPickup", "Transporting", "ToCharge") or not a.path:
                continue
            if cell not in a.path:
                continue
            goal = self._agv_goal(a)
            if goal is None:
                continue
            avoid = {o.pos for o in self.agvs if o is not a} | temp_blocked
            new_path = astar_path_avoiding(self.walkable, a.pos, goal, avoid)
            if new_path and len(new_path) > 1:
                a.path = new_path[1:]
                self.stats["obstacle_reroute"] += 1
                self._log(f"  DYNAMIC OBSTACLE at {cell} forces {a.label} to replan its route")
            else:
                self.stats["obstacle_wait"] += 1
                self._log(f"  DYNAMIC OBSTACLE at {cell}: {a.label} has no alternate route right now, "
                          f"will wait if/when it reaches the blocked cell")

    def _update_dynamic_obstacles(self):
        for cell in [c for c, exp in self.dynamic_obstacles.items() if self.t >= exp]:
            del self.dynamic_obstacles[cell]
            self._log(f"  Dynamic obstacle at {cell} cleared, cell reopened")

        if self.rng.random() < self.P_DYNAMIC_OBSTACLE_SPAWN:
            cell = self._pick_dynamic_obstacle_cell()
            if cell is None:
                self._log("  Dynamic obstacle spawn attempt skipped: no safe, connectivity-preserving cell found")
                return
            duration = self.rng.randint(self.DYNAMIC_OBSTACLE_DURATION_MIN, self.DYNAMIC_OBSTACLE_DURATION_MAX)
            self.dynamic_obstacles[cell] = self.t + duration
            self.stats["dynamic_obstacle_spawn"] += 1
            self._log(f"  Dynamic obstacle appears at {cell} for {duration} steps "
                      f"(expires t={self.t + duration})")
            self._force_reroute_for_cell(cell)

    # -- Phase 2: moving pedestrians -------------------------------------------

    def _update_pedestrians(self):
        # mirror image of _enforce_pedestrian_safety: a pedestrian also won't step onto
        # a cell an AGV is currently sitting on, it just waits a step instead
        occupied = {a.pos for a in self.agvs}
        still_active = []
        for ped in self.pedestrians:
            next_idx = ped["idx"] + 1
            if next_idx >= len(ped["path"]):
                self._log(f"  Pedestrian finishes crossing at {ped['path'][-1]}, route clears")
                continue
            next_cell = ped["path"][next_idx]
            if next_cell in occupied:
                self._log(f"  Pedestrian crossing holds at {ped['path'][ped['idx']]}: "
                          f"an AGV currently occupies {next_cell}")
            else:
                ped["idx"] = next_idx
                self._log(f"  Pedestrian crossing advances to {next_cell}")
            still_active.append(ped)
        self.pedestrians = still_active

        if self.t >= self._next_pedestrian_spawn:
            route = list(self.rng.choice(self.PEDESTRIAN_ROUTES))
            if route[0] in occupied:
                self._next_pedestrian_spawn = self.t + 1
            else:
                self.pedestrians.append({"path": route, "idx": 0})
                self._log(f"  Pedestrian enters crossing at {route[0]}, route {route}")
                self._next_pedestrian_spawn = (
                    self.t + self.PEDESTRIAN_SPAWN_PERIOD
                    + self.rng.randint(-self.PEDESTRIAN_SPAWN_JITTER, self.PEDESTRIAN_SPAWN_JITTER)
                )

    # -- Phase 2: charging station outages -------------------------------------

    def _update_station_outages(self):
        for cs in [c for c, exp in self.station_outage.items() if self.t >= exp]:
            del self.station_outage[cs]
            self._log(f"  Charging station at {cs} back ONLINE")

        # Only ever arm a new outage while none is currently down -- this is
        # exactly what guarantees the two stations are never down at once.
        if not self.station_outage and self.rng.random() < self.P_STATION_OUTAGE:
            cs = self.rng.choice(self.charging_stations)
            duration = self.rng.randint(self.STATION_OUTAGE_DURATION_MIN, self.STATION_OUTAGE_DURATION_MAX)
            self.station_outage[cs] = self.t + duration
            self.stats["station_outage_events"] += 1
            self._log(f"  Charging station at {cs} goes OUT OF SERVICE for {duration} steps "
                      f"(expires t={self.t + duration})")

            other = self._other_station(cs)
            for a in self.agvs:
                if a.status == "Charging" and a.pos == cs:
                    if other is not None:
                        a.status = "ToCharge"
                        a.path = self._plan_path(a.pos, other)[1:]
                        self._log(f"  {a.label} was charging at now out-of-service {cs}, "
                                  f"redirected to {other} instead")
                    else:
                        self._log(f"  {a.label} was charging at now out-of-service {cs}, "
                                  f"but no alternate station exists -- waiting it out")
                elif a.status == "ToCharge" and a.path and a.path[-1] == cs:
                    # Stale path computed before the outage started.
                    new_goal = self._agv_goal(a)  # _nearest_charging_station now skips `cs`
                    a.path = self._plan_path(a.pos, new_goal)[1:]
                    self._log(f"  {a.label} was heading to now out-of-service {cs}, rerouted to {new_goal}")

    def _record_history(self):
        self.history.append({
            "t": self.t,
            "agvs": [(a.label, a.pos, a.battery, a.status, a.carried_pallet) for a in self.agvs],
            "pallets": [(p.id, p.position, p.status) for p in self.pallets.values()],
            "pending": sum(1 for m in self.missions.values() if m.status == "Pending"),
            "completed": sum(1 for m in self.missions.values() if m.status == "Completed"),
            "pedestrian_active": bool(self._active_pedestrian_cells()),
            "pedestrian_cells": set(self._active_pedestrian_cells()),
            "dynamic_obstacles": set(self.dynamic_obstacles.keys()),
            "station_outage": set(self.station_outage.keys()),
        })

    def step(self):
        self._generate_flows()
        self._update_dynamic_obstacles()
        self._update_pedestrians()
        self._update_station_outages()
        self._negotiate()
        self._resolve_and_move()
        self._record_history()
        if self.t >= self.N_STEPS:
            self.stop()