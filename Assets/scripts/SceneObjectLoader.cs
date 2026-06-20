using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

[System.Serializable]
public class DetectedObject
{
    public string id;
    public string @class;
    public float[] position;
    public float[] size;
    public int observation_count;
}

[System.Serializable]
public class DetectedObjectList
{
    public List<DetectedObject> objects;
}

public class SceneObjectLoader : MonoBehaviour
{
    [Header("Settings")]
    public string jsonPath = @"C:\Users\Arni\UnityProjects\TestCapture\segmentation\objects_3d.json";
    public bool showDebugVisuals = true;

    [Header("Transform to match splat")]
    public Transform splatTransform;

    void Start()
    {
        LoadObjects();
    }

    void LoadObjects()
    {
        if (!File.Exists(jsonPath))
        {
            Debug.LogError("objects_3d.json not found at: " + jsonPath);
            return;
        }

        string json = File.ReadAllText(jsonPath);

        // Wrap in a list wrapper since JSON is an array
        string wrappedJson = "{\"objects\":" + json + "}";
        DetectedObjectList data = JsonUtility.FromJson<DetectedObjectList>(wrappedJson);

        Debug.Log($"Loading {data.objects.Count} detected objects...");

        foreach (var obj in data.objects)
        {
            SpawnObject(obj);
        }
    }

    void SpawnObject(DetectedObject obj)
    {
        GameObject go = new GameObject(obj.id);

        // Position — apply splat parent transform if assigned
        Vector3 pos = new Vector3(obj.position[0], obj.position[1], obj.position[2]);
        if (splatTransform != null)
            pos = splatTransform.TransformPoint(pos);
        go.transform.position = pos;

        // Size — use class-based defaults for minimum size
        Vector3 size = GetClassSize(obj.@class, obj.size);
        go.transform.localScale = size;

        // Add collider
        BoxCollider col = go.AddComponent<BoxCollider>();

        // Add RegionObject
        RegionObject region = go.AddComponent<RegionObject>();

        // Create RegionData on the fly
        RegionData data = ScriptableObject.CreateInstance<RegionData>();
        data.regionName = FormatClassName(obj.@class) + " " + obj.id.Split('_')[1];
        data.description = $"Detected {obj.@class} — observed {obj.observation_count}x across photos";
        data.tags = new string[] { obj.@class, "auto-detected" };
        data.highlightColor = GetClassColor(obj.@class);
        region.data = data;

        // Debug visual
        if (showDebugVisuals)
        {
            GameObject visual = GameObject.CreatePrimitive(PrimitiveType.Cube);
            visual.transform.SetParent(go.transform);
            visual.transform.localPosition = Vector3.zero;
            visual.transform.localScale = Vector3.one;
            Destroy(visual.GetComponent<BoxCollider>());

            Material mat = new Material(Shader.Find("Standard"));
            Color c = GetClassColor(obj.@class);
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
        }

        // Set layer to Selectable (layer 6)
        go.layer = 6;

        Debug.Log($"Spawned: {obj.id} at {pos}");
    }

    Vector3 GetClassSize(string className, float[] rawSize)
    {
        // Class-based reasonable sizes in meters
        switch (className)
        {
            case "chair": return new Vector3(0.6f, 1.0f, 0.6f);
            case "bed": return new Vector3(1.5f, 0.5f, 2.0f);
            case "person": return new Vector3(0.5f, 1.7f, 0.5f);
            case "suitcase": return new Vector3(0.4f, 0.6f, 0.3f);
            case "backpack": return new Vector3(0.3f, 0.5f, 0.2f);
            case "cup": return new Vector3(0.1f, 0.15f, 0.1f);
            case "bottle": return new Vector3(0.1f, 0.3f, 0.1f);
            case "bowl": return new Vector3(0.2f, 0.1f, 0.2f);
            case "bench": return new Vector3(1.5f, 0.5f, 0.5f);
            case "umbrella": return new Vector3(1.0f, 0.1f, 1.0f);
            case "handbag": return new Vector3(0.3f, 0.3f, 0.2f);
            default: return new Vector3(0.5f, 0.5f, 0.5f);
        }
    }

    Color GetClassColor(string className)
    {
        switch (className)
        {
            case "chair": return Color.blue;
            case "bed": return Color.green;
            case "person": return Color.red;
            case "suitcase": return Color.yellow;
            case "backpack": return Color.cyan;
            case "cup": return Color.white;
            case "bottle": return new Color(1f, 0.5f, 0f);
            case "bowl": return Color.magenta;
            case "bench": return new Color(0.5f, 0.3f, 0f);
            default: return Color.gray;
        }
    }

    string FormatClassName(string className)
    {
        if (string.IsNullOrEmpty(className)) return className;
        return char.ToUpper(className[0]) + className.Substring(1);
    }
}