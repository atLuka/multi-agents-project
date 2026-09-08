

import random
from collections import deque

import agentpy as ap
import numpy as np

# ---------------------------------------------------------------------------
# Step 2 - Layout parameters
# ---------------------------------------------------------------------------
CONFIG = {
    "cell_size_m": 0.3,     # ~ AGV footprint
    "grid_width": 15,       # 15 * 0.3 = 4.5 m
    "grid_height": 12,      # 12 * 0.3 = 3.6 m (real platform is ~3.5 m)

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


# ---------------------------------------------------------------------------
# Step 3 - Environment model (AgentPy)
# ---------------------------------------------------------------------------
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
        self.grid.add_agents([agent], positions=[(y, x)])
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


# ---------------------------------------------------------------------------
# Step 6 - Pathfinding (BFS)
# ---------------------------------------------------------------------------
def bfs_path(walkable, start, goal):
    if start == goal:
        return [start]
    H, W = walkable.shape
    visited = {start}
    prev = {}
    q = deque([start])
    while q:
        cx, cy = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < W and 0 <= ny < H and walkable[ny, nx] and (nx, ny) not in visited:
                visited.add((nx, ny))
                prev[(nx, ny)] = (cx, cy)
                if (nx, ny) == goal:
                    path = [(nx, ny)]
                    node = (nx, ny)
                    while node != start:
                        node = prev[node]
                        path.append(node)
                    path.reverse()
                    return path
                q.append((nx, ny))
    return None


def bfs_path_avoiding(walkable, start, goal, blocked):
    if start == goal:
        return [start]
    H, W = walkable.shape
    visited = {start}
    prev = {}
    q = deque([start])
    while q:
        cx, cy = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < W and 0 <= ny < H and walkable[ny, nx]):
                continue
            if (nx, ny) in visited:
                continue
            if (nx, ny) in blocked and (nx, ny) != goal:
                continue
            visited.add((nx, ny))
            prev[(nx, ny)] = (cx, cy)
            if (nx, ny) == goal:
                path = [(nx, ny)]
                node = (nx, ny)
                while node != start:
                    node = prev[node]
                    path.append(node)
                path.reverse()
                return path
            q.append((nx, ny))
    return None


def mask_fully_connected(mask, ref):
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


# ---------------------------------------------------------------------------
# Step 7 - Cooperate/Defect payoff matrix
# ---------------------------------------------------------------------------
PAYOFF = {("C", "C"): (10, 10), ("C", "D"): (5, 0), ("D", "C"): (0, 5), ("D", "D"): (0, 0)}


# ---------------------------------------------------------------------------
# Step 8 - Pallets and missions
# ---------------------------------------------------------------------------
class Pallet:
    def __init__(self, pid, position, status):
        self.id, self.position, self.status = pid, position, status

    def __repr__(self):
        return f"Pallet({self.id},{self.position},{self.status})"


class Mission:
    def __init__(self, mid, origin, destination, pallet_id, flow):
        self.id, self.origin, self.destination = mid, origin, destination
        self.pallet_id, self.flow = pallet_id, flow
        self.status = "Pending"
        self.agv = None

    def __repr__(self):
        return f"Mission({self.id},{self.flow},{self.origin}->{self.destination},{self.status})"


# ---------------------------------------------------------------------------
# Step 9 - AGV agent and the multi-agent system model
# ---------------------------------------------------------------------------
class AGVAgent(ap.Agent):
    def setup(self):
        self.label = None
        self.pos = None
        self.battery = 100.0
        self.status = "Available"
        self.mission = None
        self.carried_pallet = None
        self.path = []
        self.missions_completed = 0
        self.distance_traveled = 0


class MultiAGVSystem(ap.Model):

    N_AGV = 5
    START_POSITIONS = [(2, 0), (5, 0), (7, 0), (10, 0), (12, 0)]
    B_MAX, B_THRESHOLD, B_CHARGED = 100.0, 30.0, 80.0
    C_D, R_C = 1.5, 10.0
    W_D, W_B, W_L = 2.0, 0.5, 20.0
    N_STEPS = 500
    DYNAMIC_MISSIONS_AT = (15, 30, 45)

    N_OBSTACLES = 6
    PEDESTRIAN_CELLS = frozenset({(9, 0), (9, 1), (9, 2)})
    PEDESTRIAN_PERIOD = 8
    PEDESTRIAN_ACTIVE_DURATION = 3

    def setup(self):
        self.charging_stations = [(cs["x"], cs["y"]) for cs in CONFIG["charging_stations"]]
        self.log = []
        self.history = []
        self.missions = {}
        self._mission_counter = 0
        self.pallets = {}
        self.stats = {"mission_defect_seize": 0, "mission_mutual_defect": 0,
                       "collision_yield": 0, "collision_mutual_defect": 0,
                       "pedestrian_wait": 0, "pedestrian_reroute": 0}
        self._build_pallets_and_missions()

        self.obstacles = self._generate_obstacles(self.N_OBSTACLES)
        self.walkable = W_MASK.copy()
        for (ox, oy) in self.obstacles:
            self.walkable[oy, ox] = False
        assert mask_fully_connected(self.walkable, self.START_POSITIONS[0]), \
            "obstacle placement broke grid connectivity"

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
        reserved |= set(self.PEDESTRIAN_CELLS)

        H, W = W_MASK.shape
        candidates = [(x, y) for y in range(H) for x in range(W)
                      if W_MASK[y, x] and (x, y) not in reserved]

        ref = self.START_POSITIONS[0]
        for _ in range(max_attempts):
            if len(candidates) < n:
                break
            chosen = set(random.sample(candidates, n))
            trial = W_MASK.copy()
            for (x, y) in chosen:
                trial[y, x] = False
            if mask_fully_connected(trial, ref):
                return chosen
        return set()

    def _pedestrian_active(self, t=None):
        t = self.t if t is None else t
        return (t % self.PEDESTRIAN_PERIOD) < self.PEDESTRIAN_ACTIVE_DURATION

    def _active_pedestrian_cells(self, t=None):
        return set(self.PEDESTRIAN_CELLS) if self._pedestrian_active(t) else set()

    def _plan_path(self, start, goal):
        avoid = self._active_pedestrian_cells()
        path = bfs_path_avoiding(self.walkable, start, goal, avoid) if avoid else None
        if path is None:
            path = bfs_path(self.walkable, start, goal)
        return path

    def _log(self, msg):
        self.log.append(f"[t={self.t:02d}] {msg}")

    def _use_empty_slot(self):
        return self._empty_slots.pop(0)

    def _new_mission(self, origin, destination, pallet_id, flow):
        self._mission_counter += 1
        mid = f"M{self._mission_counter:02d}"
        self.missions[mid] = Mission(mid, origin, destination, pallet_id, flow)
        return mid

    def _build_pallets_and_missions(self):
        pos_by_slot = {p["id"]: (p["x"], p["y"]) for p in CONFIG["pallet_positions"]}
        self._pos_by_slot = pos_by_slot
        self._empty_slots = []

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
            for n in (3, 4):
                slot_id = f"{rack['id']}_pallet_{n}"
                self._empty_slots.append((slot_id, pos_by_slot[slot_id]))

        self.pallets["P_prod_1"] = Pallet("P_prod_1", (0, 1), "waiting_for_pickup")
        self.pallets["P_prod_2"] = Pallet("P_prod_2", (0, 10), "waiting_for_pickup")
        self.pallets["P_in_1"] = Pallet("P_in_1", (14, 7), "waiting_for_pickup")

        self._new_mission(pos_by_slot["rack_top_pallet_1"], (14, 1), "P_rack_top_pallet_1", "Flow2")
        self._new_mission(pos_by_slot["rack_mid_upper_pallet_1"], (14, 4), "P_rack_mid_upper_pallet_1", "Flow2")
        _, dest = self._use_empty_slot()
        self._new_mission((0, 1), dest, "P_prod_1", "Flow1")
        _, dest = self._use_empty_slot()
        self._new_mission((14, 7), dest, "P_in_1", "Flow3")
        self._new_mission(pos_by_slot["rack_mid_lower_pallet_1"], (14, 10), "P_rack_mid_lower_pallet_1", "Flow2")
        _, dest = self._use_empty_slot()
        self._new_mission((0, 10), dest, "P_prod_2", "Flow1")

    def visual_pallet_position(self, pos):
        dx, dy = self.slot_visual_offset.get(pos, (0, 0))
        return (pos[0] + dx, pos[1] + dy)

    def _publish(self, mids):
        for mid in mids:
            self._log(f"Mission {mid} published")

    def _dynamic_mission(self, step):
        mid = None
        if step == 15:
            self.pallets["P_prod_3"] = Pallet("P_prod_3", (0, 4), "waiting_for_pickup")
            _, dest = self._use_empty_slot()
            mid = self._new_mission((0, 4), dest, "P_prod_3", "Flow1")
        elif step == 30:
            dest = self._pos_by_slot["rack_top_pallet_2"]
            mid = self._new_mission(dest, (14, 1), "P_rack_top_pallet_2", "Flow2")
        elif step == 45:
            self.pallets["P_in_2"] = Pallet("P_in_2", (14, 10), "waiting_for_pickup")
            _, dest = self._use_empty_slot()
            mid = self._new_mission((14, 10), dest, "P_in_2", "Flow3")
        if mid is not None:
            self._publish([mid])

    def _nearest_charging_station(self, pos):
        return min(self.charging_stations, key=lambda cs: abs(cs[0] - pos[0]) + abs(cs[1] - pos[1]))

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

            actions = {a: ("D" if random.random() < self._mission_defect_probability(a) else "C")
                       for a in eligible}
            defectors = [a for a in eligible if actions[a] == "D"]

            if len(defectors) >= 2:
                names = ", ".join(a.label for a in defectors)
                self._log(f"  {mid}: mutual defection ({names}) -> reconsidering next step (payoff D,D=0,0)")
                self.stats["mission_mutual_defect"] += 1
                continue

            if len(defectors) == 1:
                winner = defectors[0]
                self.stats["mission_defect_seize"] += 1
                self._log(f"  {mid}: {winner.label} DEFECTS and seizes the mission out of turn "
                          f"(payoff if it had cooperated instead: {PAYOFF['C','C'][0]})")
            else:
                bids = []
                for a in eligible:
                    u, dist = self._utility(a, m)
                    bids.append((u, a))
                    self._log(f"  {a.label} bid: {u:.1f} (dist={dist}, battery={a.battery:.1f}%)")
                bids.sort(key=lambda b: b[0], reverse=True)
                winner = bids[0][1]
                self._log(f"  {mid}: all agents cooperate (C strongly dominates D) "
                          f"-> negotiation winner is {winner.label}")

            m.status, m.agv = "Assigned", winner.label
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
        proximity = 1.0 / (1 + len(a.path))
        battery = a.battery / 100.0
        stake = {"Transporting": 1.0, "ToPickup": 0.5, "ToCharge": 0.2}.get(a.status, 0.0)
        return 0.4 * proximity + 0.3 * battery + 0.3 * stake

    def _collision_defect_probability(self, a):
        p = 0.12
        if a.status == "Transporting":
            p += 0.2
        if a.battery < 50:
            p += 0.2
        return min(0.5, p)

    def _resolve_conflict(self, agents_here, blocked, cell):
        scored = sorted(agents_here, key=lambda a: self._priority(a), reverse=True)
        winner = scored[0]
        for loser in scored[1:]:
            if random.random() < self._collision_defect_probability(loser):
                self._log(f"  COLLISION at {cell}: {loser.label} defects against {winner.label} "
                          f"-> both yield this step (mutual defection avoided)")
                self.stats["collision_mutual_defect"] += 1
                blocked.add(winner)
                blocked.add(loser)
                return
        for loser in scored[1:]:
            blocked.add(loser)
            self.stats["collision_yield"] += 1
            self._log(f"  COLLISION avoided at {cell}: {loser.label} yields to {winner.label} "
                      f"(priority {self._priority(winner):.2f} vs {self._priority(loser):.2f})")

    def _nudge_idle_blockers(self, movers):
        reserved = set()
        for a in movers:
            reserved.update(a.path)
        occupied = {a.pos: a for a in self.agvs}
        active_ped = self._active_pedestrian_cells()
        H, W = self.walkable.shape
        for a in self.agvs:
            if a.status != "Available" or a.pos not in reserved:
                continue
            candidates = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = a.pos[0] + dx, a.pos[1] + dy
                if (0 <= nx < W and 0 <= ny < H and self.walkable[ny, nx]
                        and (nx, ny) not in occupied and (nx, ny) not in active_ped):
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
        active_ped = self._active_pedestrian_cells()
        if not active_ped:
            return
        occupied_now = {o.pos for o in self.agvs}
        for a in movers:
            if a in blocked or planned[a] not in active_ped:
                continue
            goal = self._agv_goal(a)
            avoid = occupied_now | active_ped
            new_path = bfs_path_avoiding(self.walkable, a.pos, goal, avoid) if goal else None
            if new_path and len(new_path) > 1 and new_path[1] not in active_ped:
                blocked_cell = a.path[0] if a.path else None
                a.path = new_path[1:]
                planned[a] = a.path[0]
                self.stats["pedestrian_reroute"] += 1
                self._log(f"  PEDESTRIAN CROSSING active at {blocked_cell}: "
                          f"{a.label} reroutes to avoid it (absolute safety rule, no negotiation)")
                continue
            blocked.add(a)
            self.stats["pedestrian_wait"] += 1
            self._log(f"  PEDESTRIAN CROSSING active: {a.label} waits at {a.pos} rather than enter "
                       f"{sorted(active_ped)} (absolute safety rule, no negotiation)")

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

        active_ped = self._active_pedestrian_cells()
        for a in movers:
            if a in blocked:
                continue
            target = planned[a]
            occupant = next((o for o in self.agvs if o is not a and o.pos == target), None)
            if occupant is None:
                continue
            goal = self._agv_goal(a)
            occupied_now = {o.pos for o in self.agvs if o is not a} | active_ped
            new_path = bfs_path_avoiding(self.walkable, a.pos, goal, occupied_now) if goal else None
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
            a.battery = min(self.B_MAX, a.battery + self.R_C)
            if a.battery >= self.B_CHARGED:
                a.status = "Available"
                self._log(f"{a.label} charged to {a.battery:.1f}%, available again")

        elif a.status == "Available" and a.battery <= self.B_THRESHOLD:
            a.status = "ToCharge"
            a.path = self._plan_path(a.pos, self._nearest_charging_station(a.pos))[1:]
            self._log(f"{a.label} battery low ({a.battery:.1f}%), heading to charge")

    def _record_history(self):
        self.history.append({
            "t": self.t,
            "agvs": [(a.label, a.pos, a.battery, a.status, a.carried_pallet) for a in self.agvs],
            "pallets": [(p.id, p.position, p.status) for p in self.pallets.values()],
            "pending": sum(1 for m in self.missions.values() if m.status == "Pending"),
            "completed": sum(1 for m in self.missions.values() if m.status == "Completed"),
            "pedestrian_active": self._pedestrian_active(),
        })

    def step(self):
        if self.t in self.DYNAMIC_MISSIONS_AT:
            self._dynamic_mission(self.t)
        self._negotiate()
        self._resolve_and_move()
        self._record_history()
        if self.t >= self.N_STEPS:
            self.stop()