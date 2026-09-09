"""
push_client.py
---------------
Drives the RoboArena AGV simulation (sim_core.MultiAGVSystem) step by step and
pushes it to Unity over TCP, matching the protocol TCPIPServerAsync.cs expects:

    - connect to (host, port)                      [default 127.0.0.1:1126]
    - read the server's one-time greeting bytes     ("I will send key")
    - send ONE "environment" message describing the static layout (racks,
      charging stations, parking slot, production line, truck dock, pallet
      slots) so Unity can spawn the matching prefabs once
    - then, for every simulation step, send one "agents" message with the
      current AGV positions
    Every message is JSON, immediately followed by the literal bytes
    b"<EOF>" (the frame delimiter TCPIPServerAsync.cs looks for -- see
    EOF_MARKER in that script). A "type" field on each message tells Unity
    which parser/spawner to use.

Coordinates: the sim grid is (x, y) with cell_size_m = 0.3 m/cell (CONFIG in
sim_core.py). Unity's ground plane is X/Z with Y up, so grid x -> Unity x,
grid y -> Unity z, and Unity y is left at 0 (flat floor). Adjust
CELL_SIZE_M / Y_HEIGHT below, or use the `positionScale` field already exposed
on TCPIPServerAsync in the Inspector, if your scene uses a different scale.

Usage:
    pip install agentpy numpy --break-system-packages
    python push_client.py                      # one run, then disconnect
    python push_client.py --loop                # auto-reconnect and rerun forever
    python push_client.py --step-delay 0.25      # slower playback
    python push_client.py --host 127.0.0.1 --port 1126
"""

import argparse
import json
import socket
import time

from sim import CONFIG, MultiAGVSystem

CELL_SIZE_M = CONFIG["cell_size_m"]   # 0.3 m/cell, matches sim_core's CONFIG
Y_HEIGHT = 0.0                        # flat floor; bump this if a prefab's pivot needs it
EOF_MARKER = b"<EOF>"



def _cell_center(x, y):
    """World-space (x, z) for the center of a single grid cell (x, y)."""
    return round((x + 0.5) * CELL_SIZE_M, 4), round((y + 0.5) * CELL_SIZE_M, 4)


def _point(x, y):
    wx, wz = _cell_center(x, y)
    return {"x": wx, "y": Y_HEIGHT, "z": wz}


def _arena_wall_cells():
    w, h = CONFIG["grid_width"], CONFIG["grid_height"]
    cells = []
    for x in range(-1, w + 1):
        cells.append(_point(x, -1))
        cells.append(_point(x, h))
    return cells


# Charging station id,
CS_ID_BY_POS = {(cs["x"], cs["y"]): cs["id"] for cs in CONFIG["charging_stations"]}


def environment_payload(obstacles=()):
    """Build the one-time 'environment' message describing every static
    element, straight from CONFIG -- the same source of truth the matplotlib
    plot and the AgentPy grid use, so Unity's layout always matches the
    Python layout exactly.
    """

    racks = []
    for r in CONFIG["racks"]:
        cells = [_point(x, y)
                 for y in range(r["y"], r["y"] + r["h"])
                 for x in range(r["x"], r["x"] + r["w"])]
        racks.append({"id": r["id"], "pallet_face": r["pallet_face"], "cells": cells})

    charging_stations = [
        {"id": cs["id"], **_point(cs["x"], cs["y"])}
        for cs in CONFIG["charging_stations"]
    ]

    ps = CONFIG["parking_slot"]
    parking_slot = {
        "id": ps["id"],
        "cells": [_point(x, ps["y"]) for x in range(ps["x"], ps["x"] + ps["w"])],
    }

    pl = CONFIG["production_line"]
    production_line = {
        "id": pl["id"],
        "wall_cells": [_point(pl["x"], y) for y in range(pl["y"], pl["y"] + pl["h"])
                       if y not in pl["output_rows"]],
        "outputs": [_point(pl["x"], y) for y in pl["output_rows"]],
    }

    td = CONFIG["truck_dock"]
    truck_dock = {
        "id": td["id"],
        "wall_cells": [_point(td["x"], y) for y in range(td["y"], td["y"] + td["h"])
                       if y not in td["door_rows"]],
        "doors": [_point(td["x"], y) for y in td["door_rows"]],
    }

    pallet_positions = [
        {"id": p["id"], "rack_id": p["rack_id"], **_point(p["x"], p["y"])}
        for p in CONFIG["pallet_positions"]
    ]

    return {
        "type": "environment",
        "cell_size_m": CELL_SIZE_M,
        "racks": racks,
        "charging_stations": charging_stations,
        "parking_slot": parking_slot,
        "production_line": production_line,
        "truck_dock": truck_dock,
        "pallet_positions": pallet_positions,
        "walls": _arena_wall_cells(),
        "static_obstacles": [_point(ox, oy) for (ox, oy) in obstacles],
    }


def agv_positions_payload(model):
    data = []
    for a in model.agvs:
        gx, gy = a.pos
        data.append(_point(gx, gy))

    # sim.py only updates Pallet.position once, at delivery (see
    # MultiAGVSystem._update_agv: "elif a.status == 'Transporting' and not
    # a.path: pallet.position = a.pos"). While a pallet is mid-transport,
    # p.position is still frozen at its pickup point -- it's the AGV's own
    # a.pos that's actually moving. So for a carried pallet we look up its
    # carrier's live grid position instead of using p.position directly;
    # this is purely a visualization fix (which coordinate we report to
    # Unity), not a change to the simulation's own state.
    carrier_pos_by_pallet_id = {
        a.carried_pallet: a.pos for a in model.agvs if a.carried_pallet is not None
    }

    pallets_data = []
    for p in model.pallets.values():
        if p.status == "being_transported" and p.id in carrier_pos_by_pallet_id:
            vx, vy = carrier_pos_by_pallet_id[p.id]
        elif p.status != "being_transported":
            vx, vy = model.visual_pallet_position(p.position)
        else:
            # Fallback for the one-tick edge case where a pallet is marked
            # being_transported but its carrier isn't found (shouldn't
            # normally happen) -- better to show it at its last known spot
            # than to crash.
            vx, vy = p.position

        pt = _point(vx, vy)
        pt["id"] = p.id

        # Elevate the pallet slightly so it visibly sits ON the AGV, and
        # nudge it along +X so it isn't dead-center on top of it.
        if p.status == "being_transported":
            pt["x"] += 0.15

        pallets_data.append(pt)

    # Pedestrians come and go (spawn, walk their route, despawn), so each one
    # needs a stable id for as long as it exists. `id(ped)` (Python's
    # built-in object identity) works here because sim.py's
    # _update_pedestrians() keeps reusing the same dict for an active
    # pedestrian across steps -- it's only ever appended once and dropped,
    # never rebuilt -- so the id stays constant for that pedestrian's whole
    # lifetime and simply stops appearing once it's gone.
    pedestrians_data = []
    for ped in model.pedestrians:
        px, py = ped["path"][ped["idx"]]
        pt = _point(px, py)
        pt["id"] = f"ped_{id(ped)}"
        pedestrians_data.append(pt)

    # Only currently-down stations are listed; Unity treats any charging
    # station id NOT in this list as back online.
    station_outages_data = []
    for (sx, sy) in model.station_outage:
        pt = _point(sx, sy)
        pt["id"] = CS_ID_BY_POS.get((sx, sy), f"cs_{sx}_{sy}")
        station_outages_data.append(pt)

    return {
        "type": "agents",
        "data": data,
        "pallets": pallets_data,
        "pedestrians": pedestrians_data,
        "station_outages": station_outages_data,
    }

def send_json(sock, payload):
    message = json.dumps(payload).encode("utf-8") + EOF_MARKER
    sock.sendall(message)


def run_once(sock, step_delay, verbose):
    model = MultiAGVSystem()
    model.setup()
    model.t = 0

    # Environment first: Unity spawns racks/CS/parking/production line/truck
    # dock/pallets/static obstacles exactly once per connection, before any
    # AGV moves. Has to come after model.setup() -- model.obstacles is
    # chosen randomly (per the model's seed) inside setup(), it isn't part
    # of the static CONFIG.
    send_json(sock, environment_payload(model.obstacles))
    if verbose:
        print("sent environment layout (racks, charging stations, parking slot, "
              "production line, truck dock, pallet slots, static obstacles)")

    for _ in range(model.N_STEPS):
        model.t += 1
        model.step()

        payload = agv_positions_payload(model)
        send_json(sock, payload)

        if verbose:
            print(f"[t={model.t:02d}] sent {len(model.agvs)} AGV positions -> {payload['data']}")
            if payload["station_outages"]:
                print(f"    STATION OUTAGE active: {payload['station_outages']}")
            if payload["pedestrians"]:
                print(f"    PEDESTRIAN active: {payload['pedestrians']}")

        if step_delay > 0:
            time.sleep(step_delay)

    print(f"Simulation finished after {model.N_STEPS} steps "
          f"({sum(a.missions_completed for a in model.agvs)} missions completed).")


def connect_and_greet(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    greeting = sock.recv(4096)
    print(f"Connected to Unity at {host}:{port}, server said: {greeting!r}")
    return sock


def main():
    parser = argparse.ArgumentParser(description="Push RoboArena AGV positions to Unity over TCP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1126)
    parser.add_argument("--step-delay", type=float, default=0.15,
                         help="seconds to sleep between simulation steps (0 = as fast as possible)")
    parser.add_argument("--loop", action="store_true",
                         help="after one run finishes, reconnect and run a fresh simulation again")
    parser.add_argument("--quiet", action="store_true", help="don't print each step's payload")
    args = parser.parse_args()

    while True:
        try:
            sock = connect_and_greet(args.host, args.port)
        except ConnectionRefusedError:
            print(f"Could not connect to {args.host}:{args.port}. "
                  f"Make sure the Unity scene is in Play mode (TCPIPServerAsync listens on that port).")
            return

        try:
            run_once(sock, args.step_delay, verbose=not args.quiet)
        except (BrokenPipeError, ConnectionResetError):
            print("Unity closed the connection.")
        finally:
            sock.close()

        if not args.loop:
            break
        print("--loop is set, restarting the simulation and reconnecting...\n")


if __name__ == "__main__":
    main()