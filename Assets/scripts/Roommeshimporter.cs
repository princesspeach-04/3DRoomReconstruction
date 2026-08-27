using System.IO;
using UnityEngine;

// Loads output/room_mesh.obj (from build_room_mesh.py) as an INVISIBLE
// collision-only mesh -- this is the hybrid approach: the splat still
// renders every visual, this only provides real room-shaped collision
// for RoomWalker to walk against, instead of RoomColliderBuilder's 6
// hardcoded box walls.
//
// ATTACH: create an empty GameObject named "RoomCollision", make it a
// CHILD of GaussianSplats (same axis-fix inheritance reason as every
// other spawner in this project), and put this component on it.
public class RoomMeshImporter : MonoBehaviour
{
    [Tooltip("Full path to output/room_mesh.obj")]
    public string roomMeshPath;

    void Start()
    {
        if (!File.Exists(roomMeshPath))
        {
            Debug.LogError($"RoomMeshImporter: no mesh at {roomMeshPath} -- "
                          + "run build_room_mesh.py first.");
            return;
        }

        Mesh mesh = RuntimeObjImporter.LoadOBJ(roomMeshPath);
        if (mesh == null || mesh.vertexCount == 0)
        {
            Debug.LogError("RoomMeshImporter: loaded mesh has 0 vertices.");
            return;
        }

        // No MeshFilter/MeshRenderer added on purpose -- this object is
        // invisible, collision-only. The splat is still what you see.
        var mc = gameObject.AddComponent<MeshCollider>();
        mc.sharedMesh = mesh;
        mc.convex = false; // room-scale, no Rigidbody needed here either

        Debug.Log($"RoomMeshImporter: loaded collision mesh "
                  + $"({mesh.vertexCount} verts, {mesh.triangles.Length / 3} tris)");
    }
}