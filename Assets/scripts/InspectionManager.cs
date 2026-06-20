using UnityEngine;
using UnityEngine.UI;

public class InspectionManager : MonoBehaviour
{
    public static InspectionManager Instance;

    [Header("UI References")]
    public GameObject inspectionPanel;
    public Text regionNameText;
    public Text regionDescriptionText;
    public Text regionTagsText;

    private RegionObject currentSelected;

    void Awake()
    {
        Instance = this;
        if (inspectionPanel != null)
            inspectionPanel.SetActive(false);
    }

    public void ShowRegion(RegionData data)
    {
        if (data == null) return;

        inspectionPanel.SetActive(true);
        regionNameText.text = data.regionName;
        regionDescriptionText.text = data.description;
        regionTagsText.text = data.tags != null ? string.Join(", ", data.tags) : "";
    }

    public void HidePanel()
    {
        if (inspectionPanel != null)
            inspectionPanel.SetActive(false);
        if (currentSelected != null)
            currentSelected.Deselect();
        currentSelected = null;
    }

    public void SetSelected(RegionObject obj)
    {
        if (currentSelected != null && currentSelected != obj)
            currentSelected.Deselect();
        currentSelected = obj;
    }
}