using System.Collections.Generic;
using System.IO;
using UnityEngine;

[System.Serializable]
public class DetectedObjectData
{
    public string id;
    public string @class;
    public float[] position;
    public float[] size;
    public int point_count;
    public float confidence;
    public float mass;
    public float drag;
    public string mesh_file;
}

[System.Serializable]
public class DetectedObjectList
{
    public List<DetectedObjectData> objects;
}

public class SceneObjectSpawner : MonoBehaviour
{
    [Header("Settings")]
    public string jsonPath = @"C:\Users\Arni\UnityProjects\video_Capture\segmentation\objects_3d.json";
    public bool showDebugVisuals = true; // kept for cases where mesh_file fails to load -- see fallback below

    [Header("Filtering (leave classFilter empty to spawn everything)")]
    public string classFilter = "chair";
    public float minConfidence = 0f; // build_object_meshes.py's confidence is a GLOBAL average across
                                     // frames, not a real per-instance score -- don't set this too
                                     // strict, it won't discriminate between clusters meaningfully.

    [Header("Transform to match splat")]
    [Tooltip("The GameObject rendering the Gaussian Splat. Objects are parented under this so position/rotation/scale all line up automatically -- do not hand-flip any axes.")]
    public Transform splatTransform;

    [Header("Coordinate Fix")]
    public bool flipX = false;
    public bool flipY = false;
    public bool flipZ = false;

    private readonly Dictionary<string, Color> classColors = new Dictionary<string, Color>
    {
        {"chair",   Color.blue},
        {"table",   Color.green},
        {"bag",     Color.yellow},
        {"bottle",  Color.cyan},
        {"cup",     Color.white},
        {"laptop",  new Color(0.5f, 0f, 1f)},
        {"tv",      Color.red},
        {"phone",   Color.magenta},
        {"bicycle", new Color(1f, 0.5f, 0f)},
        {"couch",   new Color(0.6f, 0.3f, 0f)},
        {"bench",   new Color(0.4f, 0.2f, 0f)},
        {"object",  Color.gray},
    };

    void Start()
    {
        LoadAndSpawn();
    }

    void LoadAndSpawn()
    {
        if (!File.Exists(jsonPath))
        {
            Debug.LogError("objects_3d.json not found at: " + jsonPath);
            return;
        }

        string json = "{\"objects\":" + File.ReadAllText(jsonPath) + "}";
        DetectedObjectList data = JsonUtility.FromJson<DetectedObjectList>(json);

        int spawned = 0;
        foreach (var obj in data.objects)
        {
            if (!string.IsNullOrEmpty(classFilter) && obj.@class != classFilter)
                continue;
            if (obj.confidence < minConfidence)
                continue;

            SpawnObject(obj);
            spawned++;
        }
        Debug.Log($"Spawned {spawned} objects (filter='{classFilter}', minConf={minConfidence}) out of {data.objects.Count} total.");
    }

    void SpawnObject(DetectedObjectData obj)
    {
        GameObject go = new GameObject(obj.id);

        if (splatTransform != null)
            go.transform.SetParent(splatTransform, false);

        float px = flipX ? -obj.position[0] : obj.position[0];
        float py = flipY ? -obj.position[1] : obj.position[1];
        float pz = flipZ ? -obj.position[2] : obj.position[2];

        go.transform.localPosition = new Vector3(px, py, pz);
        go.transform.localRotation = Quaternion.identity;
        go.transform.localScale = Vector3.one;

        Color classColor = classColors.ContainsKey(obj.@class) ? classColors[obj.@class] : Color.gray;

        // ---- Try to load the real chair-shaped mesh (mesh_file, from
        // build_object_meshes.py -- Open3D alpha-shape reconstruction,
        // object-local i.e. already centered on its own origin, which is
        // why "position" above is applied separately via transform). ----
        bool loadedMesh = false;
        if (!string.IsNullOrEmpty(obj.mesh_file) && File.Exists(obj.mesh_file))
        {
            Mesh mesh = RuntimeObjImporter.LoadOBJ(obj.mesh_file);
            if (mesh != null && mesh.vertexCount > 0)
            {
                var mf = go.AddComponent<MeshFilter>();
                mf.mesh = mesh;
                var mr = go.AddComponent<MeshRenderer>();
                var mat = new Material(Shader.Find("Standard"));
                mat.color = classColor;
                mr.material = mat;

                // Non-convex, NO Rigidbody: this is the whole point of
                // switching away from ObjectPusher -- a convex hull would
                // erase the gaps between chair legs/under the seat that
                // the alpha-shape reconstruction exists to capture.
                // DraggableObject moves the transform directly, so
                // convexity is never required.
                var mc = go.AddComponent<MeshCollider>();
                mc.sharedMesh = mesh;
                mc.convex = false;

                RegionObject region0 = go.AddComponent<RegionObject>();
                region0.visualRenderer = mr;
                AttachRegionData(region0, obj, classColor);

                go.AddComponent<DraggableObject>();
                loadedMesh = true;
            }
            else
            {
                Debug.LogWarning($"mesh_file for {obj.id} loaded but had 0 vertices, "
                                  + $"falling back to box: {obj.mesh_file}");
            }
        }
        else if (!string.IsNullOrEmpty(obj.mesh_file))
        {
            Debug.LogWarning($"mesh_file not found for {obj.id}: {obj.mesh_file}, falling back to box");
        }

        // ---- Fallback: old box-collider debug-cube path, only used if
        // mesh_file is missing/failed to load, so a bad mesh reconstruction
        // doesn't silently drop the object from the scene entirely. ----
        if (!loadedMesh && showDebugVisuals)
        {
            Vector3 size = new Vector3(
                Mathf.Max(obj.size[0], 0.05f),
                Mathf.Max(obj.size[1], 0.05f),
                Mathf.Max(obj.size[2], 0.05f));

            BoxCollider col = go.AddComponent<BoxCollider>();
            col.size = size;

            GameObject visual = GameObject.CreatePrimitive(PrimitiveType.Cube);
            visual.transform.SetParent(go.transform, false);
            visual.transform.localScale = size;
            Destroy(visual.GetComponent<BoxCollider>());
            var mat = new Material(Shader.Find("Standard"));
            Color c = classColor; c.a = 0.5f;
            mat.color = c;
            visual.GetComponent<Renderer>().material = mat;

            RegionObject region1 = go.AddComponent<RegionObject>();
            region1.visualRenderer = visual.GetComponent<Renderer>();
            AttachRegionData(region1, obj, classColor);

            go.AddComponent<DraggableObject>();
        }

        go.layer = 6; // Selectable -- same layer SelectionRaycaster already filters on

        Debug.Log($"Spawned {obj.id} ({obj.@class}, conf {obj.confidence:F2}, "
                  + $"mesh={(loadedMesh ? "yes" : "fallback box")}) at {go.transform.localPosition}");
    }

    void AttachRegionData(RegionObject region, DetectedObjectData obj, Color color)
    {
        RegionData regionData = ScriptableObject.CreateInstance<RegionData>();
        regionData.regionName = FormatClassName(obj.@class) + " " + obj.id.Split('_')[1];
        regionData.description = $"Detected {obj.@class} - {obj.point_count} pts, confidence {obj.confidence:F2}";
        regionData.tags = new string[] { obj.@class, "auto-detected" };
        regionData.highlightColor = color;
        region.data = regionData;
    }

    string FormatClassName(string className)
    {
        if (string.IsNullOrEmpty(className)) return className;
        return char.ToUpper(className[0]) + className.Substring(1);
    }
}