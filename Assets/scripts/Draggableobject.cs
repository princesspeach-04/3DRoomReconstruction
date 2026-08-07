using UnityEngine;

// Lets the user click-and-drag a reconstructed object to reposition it
// after scene load. Deliberately does NOT use Rigidbody/physics: Unity's
// physics engine requires a CONVEX collider for anything non-kinematic,
// and forcing convex on a chair mesh would erase exactly the shape detail
// (gaps under the seat, between legs) that build_object_meshes.py's
// alpha-shape reconstruction was built to preserve. Direct transform
// movement has no such restriction, so the non-convex MeshCollider from
// SceneObjectSpawner stays intact.
//
// ATTACH: added automatically by SceneObjectSpawner.cs to every spawned
// object -- no manual attachment needed. If you want to test dragging on
// a single object by hand, just Add Component > Draggable Object.
[RequireComponent(typeof(Collider))]
public class DraggableObject : MonoBehaviour
{
    [Tooltip("Height above the object's current Y to drag it, so it stays "
             + "level with the floor instead of following the camera in 3D.")]
    public bool constrainToHorizontalPlane = true;

    private Camera cam;
    private Plane dragPlane;
    private Vector3 dragOffset;
    private bool dragging;

    void Start()
    {
        cam = Camera.main;
        if (cam == null)
            Debug.LogError($"[DraggableObject] Camera.main is null on {gameObject.name} -- "
                            + "check your camera's Tag dropdown is set to 'MainCamera', "
                            + "not just named 'Main Camera'. Dragging can't work without this.");
    }

    void OnMouseDown()
    {
        Debug.Log($"[DraggableObject] OnMouseDown on {gameObject.name}");
        Vector3 planeNormal = constrainToHorizontalPlane ? Vector3.up : -cam.transform.forward;
        dragPlane = new Plane(planeNormal, transform.position);

        Ray ray = cam.ScreenPointToRay(Input.mousePosition);
        if (dragPlane.Raycast(ray, out float dist))
        {
            Vector3 hitPoint = ray.GetPoint(dist);
            dragOffset = transform.position - hitPoint;
            dragging = true;
        }
        else
        {
            Debug.LogWarning($"[DraggableObject] drag plane raycast MISSED on {gameObject.name} "
                              + "-- camera ray is near-parallel to the plane, drag won't start.");
        }
    }

    void OnMouseDrag()
    {
        if (!dragging) return;
        Ray ray = cam.ScreenPointToRay(Input.mousePosition);
        if (dragPlane.Raycast(ray, out float dist))
        {
            Vector3 hitPoint = ray.GetPoint(dist);
            transform.position = hitPoint + dragOffset;
        }
    }

    void OnMouseUp()
    {
        Debug.Log($"[DraggableObject] OnMouseUp on {gameObject.name}, moved to {transform.position}");
        dragging = false;
    }
}