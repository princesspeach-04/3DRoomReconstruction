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
    public bool showDebugVisuals = true;
    public float pushForce = 8f;

    [Header("Filtering (leave classFilter empty to spawn everything)")]
    public string classFilter = "chair";
    public float minConfidence = 1f;

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

        Vector3 size = new Vector3(
            Mathf.Max(obj.size[0], 0.05f),
            Mathf.Max(obj.size[1], 0.05f),
            Mathf.Max(obj.size[2], 0.05f)
        );

        BoxCollider col = go.AddComponent<BoxCollider>();
        col.size = size;

        Rigidbody rb = go.AddComponent<Rigidbody>();
        rb.mass = obj.mass > 0 ? obj.mass : 1f;
        rb.drag = obj.drag;
        rb.useGravity = true;

        ObjectPusher pusher = go.AddComponent<ObjectPusher>();
        pusher.pushForce = pushForce;

        RegionObject region = go.AddComponent<RegionObject>();
        RegionData regionData = ScriptableObject.CreateInstance<RegionData>();
        regionData.regionName = FormatClassName(obj.@class) + " " + obj.id.Split('_')[1];
        regionData.description = $"Detected {obj.@class} - {obj.point_count} pts, confidence {obj.confidence:F2}";
        regionData.tags = new string[] { obj.@class, "auto-detected" };
        regionData.highlightColor = classColors.ContainsKey(obj.@class) ? classColors[obj.@class] : Color.gray;
        region.data = regionData;

        if (showDebugVisuals)
        {
            GameObject visual = GameObject.CreatePrimitive(PrimitiveType.Cube);
            visual.transform.SetParent(go.transform, false);
            visual.transform.localPosition = Vector3.zero;
            visual.transform.localScale = size;
            Destroy(visual.GetComponent<BoxCollider>());

            Material mat = new Material(Shader.Find("Standard"));
            Color c = regionData.highlightColor;
            c.a = 0.5f;
            mat.color = c;
            mat.SetFloat("_Mode", 3);
            mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            mat.SetInt("_ZWrite", 0);
            mat.DisableKeyword("_ALPHATEST_ON");
            mat.EnableKeyword("_ALPHABLEND_ON");
            mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            mat.renderQueue = 3000;
            visual.GetComponent<Renderer>().material = mat;

            region.visualRenderer = visual.GetComponent<Renderer>();
        }

        go.layer = 6; // Selectable

        Debug.Log($"Spawned {obj.id} ({obj.@class}, conf {obj.confidence:F2}) at {go.transform.localPosition}, size {size}");
    }

    string FormatClassName(string className)
    {
        if (string.IsNullOrEmpty(className)) return className;
        return char.ToUpper(className[0]) + className.Substring(1);
    }
}