using System.Collections;
using System.Diagnostics;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

public class ReconstructionManager : MonoBehaviour
{
    [Header("UI")]
    public Text statusText;
    public Button reconstructButton;

    [Header("Paths")]
    public string colmapExePath = @"C:\Users\Arni\COLMAP\bin\colmap.exe";
    public string pythonExePath = @"C:\Users\Arni\UnityProjects\gaussian-splatting\venv\Scripts\python.exe";
    public string trainScriptPath = @"C:\Users\Arni\UnityProjects\gaussian-splatting\train.py";

    private ImportManager importManager;

    void Start()
    {
        importManager = FindObjectOfType<ImportManager>();
    }

    public void StartReconstruction()
    {
        string workingFolder = importManager.GetWorkingFolderPath();
        StartCoroutine(RunPipeline(workingFolder));
    }

    IEnumerator RunPipeline(string sourceFolder)
    {
        reconstructButton.interactable = false;

        // Step 1: Feature extraction
        UpdateStatus("Running COLMAP: Feature Extraction...");
        yield return StartCoroutine(RunProcess(colmapExePath,
            $"feature_extractor " +
            $"--database_path \"{sourceFolder}/database.db\" " +
            $"--image_path \"{sourceFolder}/extracted\" " +
            $"--ImageReader.single_camera 1"));

        // Step 2: Feature matching
        UpdateStatus("Running COLMAP: Feature Matching...");
        yield return StartCoroutine(RunProcess(colmapExePath,
            $"exhaustive_matcher " +
            $"--database_path \"{sourceFolder}/database.db\""));

        // Step 3: Sparse reconstruction
        Directory.CreateDirectory(sourceFolder + "/sparse");
        UpdateStatus("Running COLMAP: Sparse Reconstruction...");
        yield return StartCoroutine(RunProcess(colmapExePath,
            $"mapper " +
            $"--database_path \"{sourceFolder}/database.db\" " +
            $"--image_path \"{sourceFolder}/extracted\" " +
            $"--output_path \"{sourceFolder}/sparse\""));

        // Step 4: Gaussian Splatting training
        UpdateStatus("Training Gaussian Splat (this takes several minutes)...");
        yield return StartCoroutine(RunProcess(pythonExePath,
            $"\"{trainScriptPath}\" " +
            $"-s \"{sourceFolder}\" " +
            $"-m \"{sourceFolder}/output\" " +
            $"--iterations 7000"));

        UpdateStatus("Done! Loading result...");
        string plyPath = Path.Combine(sourceFolder,
            "output/point_cloud/iteration_7000/point_cloud.ply");

        if (File.Exists(plyPath))
            UpdateStatus("Reconstruction complete! PLY file ready at: " + plyPath);
        else
            UpdateStatus("Something went wrong — PLY file not found. Check Console.");

        reconstructButton.interactable = true;
    }

    IEnumerator RunProcess(string exe, string args)
    {
        Process process = new Process();
        process.StartInfo.FileName = exe;
        process.StartInfo.Arguments = args;
        process.StartInfo.UseShellExecute = false;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.CreateNoWindow = true;
        process.Start();

        while (!process.HasExited)
        {
            yield return new WaitForSeconds(1f);
        }

        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();

        if (!string.IsNullOrEmpty(output)) UnityEngine.Debug.Log(output);
        if (!string.IsNullOrEmpty(error)) UnityEngine.Debug.LogWarning(error);
    }

    void UpdateStatus(string message)
    {
        if (statusText != null) statusText.text = message;
        UnityEngine.Debug.Log("[Reconstruction] " + message);
    }
}