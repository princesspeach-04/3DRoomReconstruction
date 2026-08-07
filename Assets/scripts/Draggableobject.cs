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
    }

    void OnMouseDown()
    {
        Vector3 planeNormal = constrainToHorizontalPlane ? Vector3.up : -cam.transform.forward;
        dragPlane = new Plane(planeNormal, transform.position);

        Ray ray = cam.ScreenPointToRay(Input.mousePosition);
        if (dragPlane.Raycast(ray, out float dist))
        {
            Vector3 hitPoint = ray.GetPoint(dist);
            dragOffset = transform.position - hitPoint;
            dragging = true;
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
        dragging = false;
    }
}