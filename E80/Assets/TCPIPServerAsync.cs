using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

// TC2008B. Push strategy: this script listens for a TCP connection from the
// Python multi-agent simulation (push_client.py). The Python client sends,
// once per connection, an "environment" message describing the static
// layout (racks, charging stations, parking slot, production line, truck
// dock, pallet slots), followed by one "agents" message per simulation step
// with the current AGV positions. Every message is JSON terminated by
// "<EOF>". Networking happens on a background thread; parsed messages are
// queued and applied to GameObjects on Unity's main thread in Update().
public class TCPIPServerAsync : MonoBehaviour
{
    [Header("Network")]
    public string ip = "127.0.0.1";
    public int port = 1126;

    [Header("AGV Visualization")]
    public GameObject agentPrefab;      // optional; falls back to a sphere primitive
    public float positionScale = 1f;

    [Header("Environment Prefabs")]
    public GameObject rackPrefab;               // one instance per rack, stretched to its footprint
    public GameObject chargingStationPrefab;    // one instance per charging station cell
    public GameObject parkingSlotPrefab;        // optional; one instance for the parking slot
    public GameObject productionLineWallPrefab; // stretched along the whole production line column
    public GameObject productionOutputPrefab;   // one instance per output cell
    public GameObject truckDockWallPrefab;       // stretched along the whole truck dock column
    public GameObject truckDoorPrefab;           // one instance per dock door cell
    public GameObject palletPrefab;              // one instance per pallet slot

    private Thread socketThread;
    private volatile bool keepReading = false;

    private Socket listener;
    private Socket handler;

    // Thread-safe hand-off from the network thread to the main thread.
    private readonly ConcurrentQueue<string> incomingMessages = new ConcurrentQueue<string>();
    private readonly List<GameObject> agentObjects = new List<GameObject>();
    private readonly List<GameObject> environmentObjects = new List<GameObject>();
    private Dictionary<string, GameObject> activePallets = new Dictionary<string, GameObject>();
    private const string EOF_MARKER = "<EOF>";

    void Start()
    {
        Application.runInBackground = true;
        StartServer();
    }

    void Update()
    {
        // Apply every queued message in order (environment message must be
        // processed before the agent messages that follow it), but if
        // several agent-position frames piled up, only the most recent one
        // actually needs to move anything (older frames are already stale).
        string latestAgents = null;
        while (incomingMessages.TryDequeue(out string msg))
        {
            MessageEnvelope envelope = null;
            try
            {
                envelope = JsonUtility.FromJson<MessageEnvelope>(msg);
            }
            catch (Exception e)
            {
                Debug.LogWarning("Could not read message type: " + e.Message + "\nPayload: " + msg);
                continue;
            }

            if (envelope == null || string.IsNullOrEmpty(envelope.type))
            {
                // Older clients that don't send "type" always meant AGV positions.
                latestAgents = msg;
                continue;
            }

            if (envelope.type == "environment")
            {
                // Flush any pending agent frame first: it was queued before this
                // fresher environment message shouldn't be, but just in case.
                if (latestAgents != null)
                {
                    ApplyPositions(latestAgents);
                    latestAgents = null;
                }
                BuildEnvironment(msg);
            }
            else if (envelope.type == "agents")
            {
                latestAgents = msg;
            }
        }

        if (latestAgents != null)
        {
            ApplyPositions(latestAgents);
        }
    }

    void StartServer()
    {
        socketThread = new Thread(NetworkCode);
        socketThread.IsBackground = true;
        socketThread.Start();
    }

    void NetworkCode()
    {
        IPAddress ipAddress = IPAddress.Parse(ip);
        IPEndPoint localEndPoint = new IPEndPoint(ipAddress, port);

        listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);

        try
        {
            listener.Bind(localEndPoint);
            listener.Listen(10);

            while (true)
            {
                keepReading = true;
                Debug.Log("Waiting for connection on " + ip + ":" + port);

                handler = listener.Accept();
                Debug.Log("Client connected");

                string buffer = "";
                byte[] sendBytes = Encoding.UTF8.GetBytes("I will send key");
                handler.Send(sendBytes);

                while (keepReading)
                {
                    byte[] bytes = new byte[4096];
                    int bytesRec = handler.Receive(bytes);

                    if (bytesRec <= 0)
                    {
                        keepReading = false;
                        handler.Disconnect(true);
                        break;
                    }

                    buffer += Encoding.UTF8.GetString(bytes, 0, bytesRec);

                    // A single Receive() call may contain zero, one, or
                    // several complete "<EOF>"-terminated messages.
                    int eofIndex;
                    while ((eofIndex = buffer.IndexOf(EOF_MARKER)) > -1)
                    {
                        string message = buffer.Substring(0, eofIndex);
                        Debug.Log("Received from Python: " + message.Substring(0, Math.Min(80, message.Length)));
                        incomingMessages.Enqueue(message);
                        buffer = buffer.Substring(eofIndex + EOF_MARKER.Length);
                    }
                }

                Debug.Log("Client disconnected");
            }
        }
        catch (Exception e)
        {
            Debug.Log(e.ToString());
        }
    }

    // -----------------------------------------------------------------
    // Environment: racks, charging stations, parking slot, production
    // line + outputs, truck dock + doors, pallet slots.
    // -----------------------------------------------------------------
    void BuildEnvironment(string json)
    {
        EnvironmentPayload env;
        try
        {
            env = JsonUtility.FromJson<EnvironmentPayload>(json);
        }
        catch (Exception e)
        {
            Debug.LogWarning("Could not parse environment data: " + e.Message);
            return;
        }
        if (env == null) return;

        ClearEnvironment();

        if (env.racks != null)
        {
            foreach (var r in env.racks)
            {
                if (r.cells == null) continue;
                int i = 0;
                foreach (var c in r.cells)
                {
                    GameObject go = SpawnPoint(rackPrefab, c.x, c.y, c.z, "Rack_" + r.id + "_" + (i++));
                    if (go != null) environmentObjects.Add(go);
                }
            }
        }

        if (env.charging_stations != null)
        {
            foreach (var cs in env.charging_stations)
            {
                GameObject go = SpawnPoint(chargingStationPrefab, cs.x, cs.y, cs.z, "Cs_" + cs.id);
                if (go != null) environmentObjects.Add(go);
            }
        }

        if (env.parking_slot != null && env.parking_slot.cells != null)
        {
            int i = 0;
            foreach (var c in env.parking_slot.cells)
            {
                GameObject go = SpawnPoint(parkingSlotPrefab, c.x, c.y, c.z, "Parking_" + env.parking_slot.id + "_" + (i++));
                if (go != null) environmentObjects.Add(go);
            }
        }

        if (env.production_line != null)
        {
            var pl = env.production_line;
            if (pl.wall_cells != null)
            {
                int i = 0;
                foreach (var c in pl.wall_cells)
                {
                    GameObject go = SpawnPoint(productionLineWallPrefab, c.x, c.y, c.z, "ProductionLine_" + (i++));
                    if (go != null) environmentObjects.Add(go);
                }
            }
            if (pl.outputs != null)
            {
                int i = 0;
                foreach (var o in pl.outputs)
                {
                    GameObject go = SpawnPoint(productionOutputPrefab, o.x, o.y, o.z, "ProductionOutput_" + (i++));
                    if (go != null) environmentObjects.Add(go);
                }
            }
        }

        if (env.truck_dock != null)
        {
            var td = env.truck_dock;
            if (td.wall_cells != null)
            {
                int i = 0;
                foreach (var c in td.wall_cells)
                {
                    GameObject go = SpawnPoint(truckDockWallPrefab, c.x, c.y, c.z, "TruckWall_" + (i++));
                    if (go != null) environmentObjects.Add(go);
                }
            }
            if (td.doors != null)
            {
                int i = 0;
                foreach (var d in td.doors)
                {
                    GameObject go = SpawnPoint(truckDoorPrefab, d.x, d.y, d.z, "Truck_" + (i++));
                    if (go != null) environmentObjects.Add(go);
                }
            }
        }

        

        Debug.Log("Environment built: " + environmentObjects.Count + " objects spawned.");
    }

    // Instantiates `prefab` at a single cell center, unscaled. Returns null
    // (and logs once) if no prefab was assigned in the Inspector for that
    // element -- everything here is tiled one real-size prefab per grid
    // cell rather than one prefab stretched to fit a multi-cell footprint,
    // since these prefabs are already modeled at their intended size.
    GameObject SpawnPoint(GameObject prefab, float x, float y, float z, string name)
    {
        if (prefab == null)
        {
            Debug.LogWarning("No prefab assigned for " + name + " -- skipping. Assign one in the Inspector.");
            return null;
        }
        GameObject go = Instantiate(prefab, new Vector3(x, y, z) * positionScale, Quaternion.identity);
        go.name = name;
        return go;
    }

    void ClearEnvironment()
    {
        foreach (var go in environmentObjects)
        {
            if (go != null) Destroy(go);
        }
        environmentObjects.Clear();
    }

    // -----------------------------------------------------------------
    // AGV positions (per simulation step)
    // -----------------------------------------------------------------
    void ApplyPositions(string json)
    {
        AgentData agentData;
        try
        {
            agentData = JsonUtility.FromJson<AgentData>(json);
        }
        catch (Exception e)
        {
            Debug.LogWarning("Could not parse agent data: " + e.Message + "\nPayload: " + json);
            return;
        }

        if (agentData == null || agentData.data == null) return;

        // Instantiate one GameObject per agent the first time we see them.
        while (agentObjects.Count < agentData.data.Count)
        {
            GameObject go = agentPrefab != null
                ? Instantiate(agentPrefab)
                : GameObject.CreatePrimitive(PrimitiveType.Sphere);
            agentObjects.Add(go);
        }

        for (int i = 0; i < agentData.data.Count; i++)
        {
            AgentPosition p = agentData.data[i];
            agentObjects[i].transform.position = new Vector3(p.x, p.y, p.z) * positionScale;
        }
        if (agentData.pallets != null)
    {
        foreach (var p in agentData.pallets)
        {
            // Spawn missing pallets on the fly
            if (!activePallets.ContainsKey(p.id))
            {
                if (palletPrefab != null)
                {
                    GameObject newPallet = Instantiate(palletPrefab);
                    newPallet.name = "Dynamic_" + p.id;
                    activePallets[p.id] = newPallet;
                }
            }

            // Move the pallet to its live position
            if (activePallets.ContainsKey(p.id))
            {
                activePallets[p.id].transform.position = new Vector3(p.x, p.y, p.z) * positionScale;
            }
        }
    }
    }

    void StopServer()
    {
        keepReading = false;

        if (handler != null && handler.Connected)
        {
            handler.Disconnect(false);
            Debug.Log("Disconnected!");
        }

        if (socketThread != null)
        {
            socketThread.Abort();
        }
    }

    void OnDisable()
    {
        StopServer();
    }
}

// ---------------------------------------------------------------------
// JSON shapes sent by push_client.py. Keep these in sync with that file.
// ---------------------------------------------------------------------

// Every message starts with a "type" field; parsed first to decide which
// full struct to parse the same JSON into next.
[Serializable]
public class MessageEnvelope
{
    public string type; // "environment" | "agents"
}

[Serializable]
public class AgentPosition
{
    public float x;
    public float y;
    public float z;
}

// {"type":"agents","data":[{"x":..,"y":..,"z":..}, ...]}
[Serializable]
public class PalletUpdate
{
    public string id;
    public float x;
    public float y;
    public float z;
}

[Serializable]
public class AgentData
{
    public string type;
    public List<AgentPosition> data;
    public List<PalletUpdate> pallets; // NEW
}

[Serializable]
public class RackSpec
{
    public string id;
    public string pallet_face;
    public List<AgentPosition> cells; // one entry per grid cell the rack covers, reuses {x,y,z}
}

[Serializable]
public class ChargingStationSpec
{
    public string id;
    public float x, y, z;
}

[Serializable]
public class ParkingSlotSpec
{
    public string id;
    public List<AgentPosition> cells;
}

[Serializable]
public class ProductionLineSpec
{
    public string id;
    public List<AgentPosition> wall_cells; // every column cell that isn't an output
    public List<AgentPosition> outputs;
}

[Serializable]
public class TruckDockSpec
{
    public string id;
    public List<AgentPosition> wall_cells; // every column cell that isn't a door
    public List<AgentPosition> doors;
}

[Serializable]
public class PalletSpec
{
    public string id;
    public string rack_id;
    public float x, y, z;
}

// {"type":"environment", "cell_size_m":.., "racks":[...], "charging_stations":[...],
//  "parking_slot":{...}, "production_line":{...}, "truck_dock":{...}, "pallet_positions":[...]}
[Serializable]
public class EnvironmentPayload
{
    public string type;
    public float cell_size_m;
    public List<RackSpec> racks;
    public List<ChargingStationSpec> charging_stations;
    public ParkingSlotSpec parking_slot;
    public ProductionLineSpec production_line;
    public TruckDockSpec truck_dock;
    public List<PalletSpec> pallet_positions;
}
