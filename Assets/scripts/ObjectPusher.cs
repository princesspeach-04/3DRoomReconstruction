using UnityEngine;

public class ObjectPusher : MonoBehaviour
{
    public float pushForce = 8f;
    private Rigidbody rb;
    private Camera mainCam;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        mainCam = Camera.main;
    }

    void Update()
    {
        // Left click to push, right click held = camera movement
        if (Input.GetMouseButtonDown(0) && !Input.GetMouseButton(1))
        {
            Ray ray = mainCam.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;
            if (Physics.Raycast(ray, out hit, 100f))
            {
                if (hit.collider.gameObject == gameObject)
                {
                    Vector3 dir = (ray.direction + Vector3.up * 0.2f).normalized;
                    rb.AddForce(dir * pushForce, ForceMode.Impulse);
                    rb.AddTorque(Random.insideUnitSphere * pushForce * 0.5f, ForceMode.Impulse);
                    Debug.Log($"Pushed {gameObject.name}!");
                }
            }
        }
    }
}