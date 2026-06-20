using UnityEngine;

public class RegionObject : MonoBehaviour
{
    public RegionData data;

    private Renderer rend;
    private Color originalColor;
    private bool isHighlighted = false;

    void Start()
    {
        rend = GetComponent<Renderer>();
        if (rend != null)
            originalColor = rend.material.color;
    }

    public void Select()
    {
        isHighlighted = true;
        if (rend != null && data != null)
            rend.material.color = data.highlightColor;

        InspectionManager.Instance.ShowRegion(data);
        Debug.Log("Selected: " + (data != null ? data.regionName : "Unknown"));
    }

    public void Deselect()
    {
        isHighlighted = false;
        if (rend != null)
            rend.material.color = originalColor;
    }
}