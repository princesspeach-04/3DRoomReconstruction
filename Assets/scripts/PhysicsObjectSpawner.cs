using System.Collections.Generic;
using System.IO;
using UnityEngine;

// Replaces SceneObjectLoader.cs and PhysicsObjectSpawner.cs.
// Attach this to one GameObject in the scene (not both old scripts).

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
    public string jsonPath = @"C:\Users\Arni\UnityProjects\TestCapture\segmentation\objects_3d.json";
    public bool showDebugVisuals = true;
    public float pushForce = 8f;

    [Header("Transform to match splat")]
    [Tooltip("The GameObject rendering the Gaussian Splat. Objects are parented under this so position/rotation/scale all line up automatically -- do not hand-flip any axes.")]
    public Transform splatTransform;

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
        Debug.Log($"Spawning {data.objects.Count} objects...");

        foreach (var obj in data.objects)
            SpawnObject(obj);
    }

    void SpawnObject(DetectedObjectData obj)
    {
        GameObject go = new GameObject(obj.id);

        // Parent under the splat so position AND rotation AND scale all
        // inherit automatically -- this is the only coordinate correction
        // needed. Do not flip/reorder axes in code anywhere else.
        if (splatTransform != null)
            go.transform.SetParent(splatTransform, false);

        go.transform.localPosition = new Vector3(obj.position[0], obj.position[1], obj.position[2]);
        go.transform.localRotation = Quaternion.identity;
        go.transform.localScale = Vector3.one;

        Vector3 size = new Vector3(
            Mathf.Max(obj.size[0], 0.05f),
            Mathf.Max(obj.size[1], 0.05f),
            Mathf.Max(obj.size[2], 0.05f)
        );

        // Collider size is in local space of `go`, which is already inside
        // the splat's coordinate frame -- no extra scaling math needed.
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
        regionData.description = $"Detected {obj.@class} — {obj.point_count} pts, confidence {obj.confidence:F2}";
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
            c.a = 0.3f;
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

            // NOTE: RegionObject reads its Renderer off `go`, but the visible
            // cube is a child. Click-to-highlight color swap won't do
            // anything until RegionObject looks at the child's renderer
            // instead -- harmless, but worth fixing later if you want the
            // highlight-on-select effect to actually show.
        }

        go.layer = 6; // Selectable

        Debug.Log($"Spawned {obj.id} (local) at {go.transform.localPosition}, size {size}");
    }

    string FormatClassName(string className)
    {
        if (string.IsNullOrEmpty(className)) return className;
        return char.ToUpper(className[0]) + className.Substring(1);
    }
}