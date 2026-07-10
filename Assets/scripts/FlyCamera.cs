using UnityEngine;

public class FlyCamera : MonoBehaviour
{
    [Header("Movement")]
    public float moveSpeed = 3f;
    public float fastMultiplier = 3f;

    [Header("Look")]
    public float lookSpeed = 2f;

    [Header("Auto-start inside the splat")]
    public Transform splatTransform;
    public float startHeightAboveCenter = 2f;

    float yaw;
    float pitch;
    bool cursorLocked = false;

    void Start()
    {
        if (splatTransform != null)
        {
            Renderer rend = splatTransform.GetComponent<Renderer>();
            Vector3 center = rend != null ? rend.bounds.center : splatTransform.position;
            transform.position = center + Vector3.up * startHeightAboveCenter;
            transform.LookAt(center);
        }

        Vector3 e = transform.eulerAngles;
        yaw = e.y;
        pitch = e.x;
    }

    void Update()
    {
        // Right-click TOGGLES mouse-look, instead of requiring it held down
        if (Input.GetMouseButtonDown(1))
        {
            cursorLocked = !cursorLocked;
            Cursor.lockState = cursorLocked ? CursorLockMode.Locked : CursorLockMode.None;
            Cursor.visible = !cursorLocked;
        }

        if (cursorLocked)
        {
            yaw += lookSpeed * Input.GetAxis("Mouse X");
            pitch -= lookSpeed * Input.GetAxis("Mouse Y");
            pitch = Mathf.Clamp(pitch, -89f, 89f);
            transform.eulerAngles = new Vector3(pitch, yaw, 0f);
        }

        float speed = moveSpeed * (Input.GetKey(KeyCode.LeftShift) ? fastMultiplier : 1f);
        transform.position += transform.forward * speed * Input.GetAxis("Vertical") * Time.deltaTime;
        transform.position += transform.right * speed * Input.GetAxis("Horizontal") * Time.deltaTime;
        transform.position += transform.up * speed * (Input.GetKey(KeyCode.E) ? 1 : Input.GetKey(KeyCode.Q) ? -1 : 0) * Time.deltaTime;
    }
}