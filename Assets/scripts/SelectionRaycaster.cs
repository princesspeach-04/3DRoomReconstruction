using UnityEngine;

public class SelectionRaycaster : MonoBehaviour
{
    public Camera mainCamera;
    public LayerMask selectableLayer;

    void Update()
    {
        if (mainCamera == null)
            mainCamera = Camera.main;

        if (Input.GetMouseButtonDown(0) && !Input.GetMouseButton(1))
        {
            Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit, 1000f, selectableLayer))
            {
                Debug.Log("Raycast hit: " + hit.collider.gameObject.name);
                RegionObject region = hit.collider.GetComponent<RegionObject>();
                if (region == null)
                {
                    Debug.LogWarning("Hit object has no RegionObject component.");
                    return;
                }
                if (InspectionManager.Instance == null)
                {
                    Debug.LogError("InspectionManager.Instance is null -- is InspectionManager in the scene?");
                    return;
                }
                InspectionManager.Instance.SetSelected(region);
                region.Select();
            }
            else
            {
                Debug.Log("Raycast hit nothing on layer mask: " + selectableLayer.value);
                if (InspectionManager.Instance != null)
                    InspectionManager.Instance.HidePanel();
            }
        }
    }
}