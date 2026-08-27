using UnityEngine;

// Real first-person walking with gravity and collision, replacing
// FlyCamera's noclip flying as the default movement mode. Requires a
// CharacterController on the SAME GameObject this script sits on -- put
// this on a "Player" GameObject, with your Main Camera as its CHILD
// (at roughly eye height), not directly on the camera itself, since
// CharacterController needs its own capsule separate from the camera.
//
// SCALE WARNING: gsplat's normalize step means your scene is NOT in
// real-world meters -- it's scaled so median camera distance from
// center is ~1.0. There's no reliable way to know the true meters-per-
// unit ratio without a known real-world reference measurement in the
// scene, so moveSpeed/controllerHeight below are starting points to
// TUNE BY EYE in Play mode, not physically accurate defaults.
[RequireComponent(typeof(CharacterController))]
public class RoomWalker : MonoBehaviour
{
    [Header("Movement (tune by eye -- scene is NOT in real meters)")]
    public float moveSpeed = 0.5f;
    public float sprintMultiplier = 2f;
    public float gravity = -1.5f;
    public float jumpHeight = 0.15f;

    [Header("Look")]
    public float lookSpeed = 2f;
    public Transform cameraTransform; // assign your Main Camera child here

    private CharacterController controller;
    private float yaw, pitch;
    private float verticalVelocity;
    private bool cursorLocked = false;

    void Start()
    {
        controller = GetComponent<CharacterController>();
        if (cameraTransform == null)
            Debug.LogError("RoomWalker: assign Camera Transform in the Inspector "
                          + "-- mouse look has nothing to rotate without it.");
    }

    void Update()
    {
        // Right-click TOGGLES mouse-look, matching FlyCamera's existing convention
        if (Input.GetMouseButtonDown(1))
        {
            cursorLocked = !cursorLocked;
            Cursor.lockState = cursorLocked ? CursorLockMode.Locked : CursorLockMode.None;
            Cursor.visible = !cursorLocked;
        }

        if (cursorLocked && cameraTransform != null)
        {
            yaw += lookSpeed * Input.GetAxis("Mouse X");
            pitch -= lookSpeed * Input.GetAxis("Mouse Y");
            pitch = Mathf.Clamp(pitch, -80f, 80f);

            transform.rotation = Quaternion.Euler(0f, yaw, 0f);        // body yaws
            cameraTransform.localRotation = Quaternion.Euler(pitch, 0f, 0f); // camera pitches
        }

        // Movement relative to look direction, flattened to the ground plane
        Vector3 forward = transform.forward;
        Vector3 right = transform.right;
        float speed = moveSpeed * (Input.GetKey(KeyCode.LeftShift) ? sprintMultiplier : 1f);

        Vector3 move = (forward * Input.GetAxis("Vertical")
                       + right * Input.GetAxis("Horizontal")) * speed;

        if (controller.isGrounded)
        {
            verticalVelocity = -0.01f; // small downward value keeps isGrounded reliable
            if (Input.GetButtonDown("Jump"))
                verticalVelocity = Mathf.Sqrt(jumpHeight * -2f * gravity);
        }
        else
        {
            verticalVelocity += gravity * Time.deltaTime;
        }

        move.y = verticalVelocity;
        controller.Move(move * Time.deltaTime);
    }
}