using UnityEngine;

[CreateAssetMenu(fileName = "NewRegion", menuName = "GaussianApp/Region Data")]
public class RegionData : ScriptableObject
{
    public string regionName = "Unnamed Region";
    [TextArea] public string description = "";
    public string[] tags;
    public Color highlightColor = Color.yellow;
}