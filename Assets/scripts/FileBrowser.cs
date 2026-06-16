using UnityEngine;

public class FileBrowser : MonoBehaviour
{
    public ImportManager importManager;

    public void OnBrowseClicked()
    {
        // For testing: change this path to a real ZIP or image on your machine
        string testPath = @"C:\Users\Arni\Downloads\tree-scaniverse-3d-gaussian-splat-ply.zip";
        importManager.ProcessFile(testPath);
    }
}