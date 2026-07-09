using UnityEngine;

public class RegionObject : MonoBehaviour
{
    public RegionData data;

    [Tooltip("Assigned by the spawner. If left empty, falls back to searching children.")]
    public Renderer visualRenderer;

    private Color originalColor;

    void Start()
    {
        if (visualRenderer == null)
            visualRenderer = GetComponentInChildren<Renderer>();

        if (visualRenderer != null)
            originalColor = visualRenderer.material.color;
    }

    public void Select()
    {
        if (visualRenderer != null && data != null)
        {
            Color c = data.highlightColor;
            c.a = 0.8f;
            visualRenderer.material.color = c;
        }

        InspectionManager.Instance.ShowRegion(data);
        Debug.Log("Selected: " + (data != null ? data.regionName : "Unknown"));
    }

    public void Deselect()
    {
        if (visualRenderer != null)
            visualRenderer.material.color = originalColor;
    }
}