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
                RegionObject region = hit.collider.GetComponent<RegionObject>();
                if (region != null && InspectionManager.Instance != null)
                {
                    InspectionManager.Instance.SetSelected(region);
                    region.Select();
                }
            }
            else
            {
                if (InspectionManager.Instance != null)
                    InspectionManager.Instance.HidePanel();
            }
        }
    }
}