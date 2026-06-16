using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using UnityEngine;
using UnityEngine.UI;

public class ImportManager : MonoBehaviour
{
    [Header("UI References")]
    public Text statusText;
    public Text dropZoneLabel;

    [Header("Settings")]
    public string workingFolderName = "ImportedFiles";

    private string workingFolderPath;
    private List<string> importedImagePaths = new List<string>();

    void Start()
    {
        workingFolderPath = Path.Combine(Application.persistentDataPath, workingFolderName);
        Directory.CreateDirectory(workingFolderPath);
        UpdateStatus("Ready. Use Browse to load files.");
    }

    public void ProcessFile(string filePath)
    {
        if (!File.Exists(filePath))
        {
            UpdateStatus("File not found: " + filePath);
            return;
        }

        string extension = Path.GetExtension(filePath).ToLower();
        UpdateStatus("Processing: " + Path.GetFileName(filePath));
        StartCoroutine(HandleFile(filePath, extension));
    }

    IEnumerator HandleFile(string filePath, string extension)
    {
        importedImagePaths.Clear();

        if (extension == ".zip")
        {
            yield return StartCoroutine(ExtractZip(filePath));
        }
        else if (extension == ".pdf")
        {
            UpdateStatus("PDF support coming in a later phase.");
            yield return null;
        }
        else if (IsImageExtension(extension))
        {
            string destPath = Path.Combine(workingFolderPath, Path.GetFileName(filePath));
            File.Copy(filePath, destPath, true);
            importedImagePaths.Add(destPath);
            UpdateStatus("Loaded 1 image.");
        }
        else
        {
            UpdateStatus("Unsupported file type: " + extension);
        }

        if (importedImagePaths.Count > 0)
        {
            UpdateStatus($"Ready! {importedImagePaths.Count} images loaded.");
            Debug.Log("Working folder: " + workingFolderPath);
        }
    }

    IEnumerator ExtractZip(string zipPath)
    {
        string extractPath = Path.Combine(workingFolderPath, "extracted");
        Directory.CreateDirectory(extractPath);

        UpdateStatus("Extracting ZIP...");
        yield return null;

        using (ZipArchive archive = ZipFile.OpenRead(zipPath))
        {
            int total = archive.Entries.Count;
            int current = 0;

            foreach (ZipArchiveEntry entry in archive.Entries)
            {
                string entryExt = Path.GetExtension(entry.Name).ToLower();
                if (IsImageExtension(entryExt) && entry.Name.Length > 0)
                {
                    string destFile = Path.Combine(extractPath, entry.Name);
                    entry.ExtractToFile(destFile, true);
                    importedImagePaths.Add(destFile);
                }
                current++;

                if (current % 10 == 0)
                {
                    UpdateStatus($"Extracting... {current}/{total}");
                    yield return null;
                }
            }
        }

        UpdateStatus($"Extracted {importedImagePaths.Count} images from ZIP.");
    }

    bool IsImageExtension(string ext)
    {
        return ext == ".jpg" || ext == ".jpeg" || ext == ".png"
               || ext == ".bmp" || ext == ".tiff";
    }

    void UpdateStatus(string message)
    {
        if (statusText != null)
            statusText.text = message;
        Debug.Log("[ImportManager] " + message);
    }

    public List<string> GetImportedImagePaths()
    {
        return importedImagePaths;
    }

    public string GetWorkingFolderPath()
    {
        return workingFolderPath;
    }
}