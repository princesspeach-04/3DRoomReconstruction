using UnityEngine;

public class RoomColliderBuilder : MonoBehaviour
{
    [Header("Parent under the same splat transform as spawned objects")]
    public Transform splatTransform;

    [Header("Bounds from check.py output (local space)")]
    public Vector3 boundsMin = new Vector3(-4.881f, -2.84f, -1.365f);
    public Vector3 boundsMax = new Vector3(4.655f, 2.466f, 3.566f);

    public float wallThickness = 0.1f;

    void Start()
    {
        Vector3 size = boundsMax - boundsMin;
        Vector3 center = (boundsMin + boundsMax) * 0.5f;

        MakeWall("Floor", center - Vector3.up * size.y * 0.5f, new Vector3(size.x, wallThickness, size.z));
        MakeWall("Ceiling", center + Vector3.up * size.y * 0.5f, new Vector3(size.x, wallThickness, size.z));
        MakeWall("Wall_+X", center + Vector3.right * size.x * 0.5f, new Vector3(wallThickness, size.y, size.z));
        MakeWall("Wall_-X", center - Vector3.right * size.x * 0.5f, new Vector3(wallThickness, size.y, size.z));
        MakeWall("Wall_+Z", center + Vector3.forward * size.z * 0.5f, new Vector3(size.x, size.y, wallThickness));
        MakeWall("Wall_-Z", center - Vector3.forward * size.z * 0.5f, new Vector3(size.x, size.y, wallThickness));
    }

    void MakeWall(string name, Vector3 localPos, Vector3 size)
    {
        GameObject go = new GameObject(name);
        if (splatTransform != null)
            go.transform.SetParent(splatTransform, false);
        go.transform.localPosition = localPos;

        BoxCollider col = go.AddComponent<BoxCollider>();
        col.size = size;
        go.layer = 0; // Default, not Selectable -- so raycasts for chairs ignore it
    }
}