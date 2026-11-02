using System.IO;
using UnityEngine;

// Loads output/room_mesh.obj (from build_room_mesh.py) as an INVISIBLE
// collision-only mesh. The splat still renders every visual.
//
// ATTACH: empty GameObject "RoomCollision", CHILD of GaussianSplats
// (so it inherits the same axis-fix transform), with this component on it.
public class RoomMeshImporter : MonoBehaviour
{
    [Tooltip("Full path to output/room_mesh.obj")]
    public string roomMeshPath;

    [Header("Debug")]
    [Tooltip("Shows the collider as a translucent orange mesh. Turn OFF for the exported asset.")]
    public bool showDebugMesh = true;

    void Start()
    {
        if (!File.Exists(roomMeshPath))
        {
            Debug.LogError($"RoomMeshImporter: no mesh at '{roomMeshPath}' -- "
                          + "run build_room_mesh.py first and set the path in the Inspector.");
            return;
        }

        Mesh mesh = RuntimeObjImporter.LoadOBJ(roomMeshPath);
        if (mesh == null || mesh.vertexCount == 0)
        {
            Debug.LogError("RoomMeshImporter: loaded mesh has 0 vertices.");
            return;
        }

        var mc = gameObject.AddComponent<MeshCollider>();
        mc.sharedMesh = mesh;
        mc.convex = false;

        if (showDebugMesh)
        {
            var mf = gameObject.AddComponent<MeshFilter>();
            mf.sharedMesh = mesh;

            var mr = gameObject.AddComponent<MeshRenderer>();
            var mat = new Material(Shader.Find("Standard"));
            mat.color = new Color(1f, 0.4f, 0f, 0.35f);
            mat.SetFloat("_Mode", 3); // transparent
            mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            mat.SetInt("_ZWrite", 0);
            mat.EnableKeyword("_ALPHABLEND_ON");
            mat.renderQueue = 3000;
            mr.material = mat;
        }

        Debug.Log($"RoomMeshImporter: loaded collision mesh "
                  + $"({mesh.vertexCount} verts, {mesh.triangles.Length / 3} tris)");
    }
}