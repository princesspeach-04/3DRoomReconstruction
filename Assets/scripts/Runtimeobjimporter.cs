using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

// Loads a .obj file into a Unity Mesh AT RUNTIME (no editor import step --
// required because a built PC app can't use AssetDatabase).
// Supports: "v x y z", "v x y z r g b" (vertex colours, 0-1 floats, as written by
// generate_replacement.py), "vn", and faces with 3+ vertices (fan-triangulated).
// No textures/materials: collider meshes use a flat material, replacement
// meshes use the vertex-colour shader (Aerosense/VertexColorLit).
public static class RuntimeObjImporter
{
    public static Mesh LoadOBJ(string path)
    {
        var vertices = new List<Vector3>();
        var normals = new List<Vector3>();
        var colors = new List<Color>();
        var triangles = new List<int>();
        bool hasNormals = false;

        foreach (string rawLine in File.ReadAllLines(path))
        {
            string line = rawLine.Trim();
            if (line.Length == 0 || line[0] == '#') continue;
            string[] parts = line.Split(' ', System.StringSplitOptions.RemoveEmptyEntries);

            if (parts[0] == "v" && parts.Length >= 4)
            {
                vertices.Add(new Vector3(
                    ParseFloat(parts[1]), ParseFloat(parts[2]), ParseFloat(parts[3])));
                if (parts.Length >= 7)
                    colors.Add(new Color(
                        ParseFloat(parts[4]), ParseFloat(parts[5]), ParseFloat(parts[6]), 1f));
            }
            else if (parts[0] == "vn" && parts.Length >= 4)
            {
                hasNormals = true;
                normals.Add(new Vector3(
                    ParseFloat(parts[1]), ParseFloat(parts[2]), ParseFloat(parts[3])));
            }
            else if (parts[0] == "f" && parts.Length >= 4)
            {
                int n = vertices.Count;
                var idx = new List<int>(parts.Length - 1);
                for (int i = 1; i < parts.Length; i++)
                {
                    int k = int.Parse(parts[i].Split('/')[0], CultureInfo.InvariantCulture);
                    idx.Add(k > 0 ? k - 1 : n + k); // OBJ is 1-indexed, negatives are relative
                }
                for (int i = 1; i < idx.Count - 1; i++)
                {
                    triangles.Add(idx[0]);
                    triangles.Add(idx[i]);
                    triangles.Add(idx[i + 1]);
                }
            }
        }

        var mesh = new Mesh();
        mesh.indexFormat = vertices.Count > 65000
            ? UnityEngine.Rendering.IndexFormat.UInt32
            : UnityEngine.Rendering.IndexFormat.UInt16;
        mesh.SetVertices(vertices);
        mesh.SetTriangles(triangles, 0);
        if (colors.Count == vertices.Count)
            mesh.SetColors(colors);
        if (hasNormals && normals.Count == vertices.Count)
            mesh.SetNormals(normals);
        else
            mesh.RecalculateNormals();
        mesh.RecalculateBounds();
        return mesh;
    }

    private static float ParseFloat(string s) =>
        float.Parse(s, CultureInfo.InvariantCulture);
}