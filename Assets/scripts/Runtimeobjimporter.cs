using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

// Loads a .obj file into a Unity Mesh AT RUNTIME (no editor import step --
// required because a built PC app can't use AssetDatabase). Handles the
// plain vertex/normal/triangle-face output Open3D's write_triangle_mesh
// produces. Does not handle textures/materials -- these are collider/
// shape meshes, not rendered assets, so a flat material is enough.
public static class RuntimeObjImporter
{
    public static Mesh LoadOBJ(string path)
    {
        var vertices = new List<Vector3>();
        var normals = new List<Vector3>();
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
            }
            else if (parts[0] == "vn" && parts.Length >= 4)
            {
                hasNormals = true;
                normals.Add(new Vector3(
                    ParseFloat(parts[1]), ParseFloat(parts[2]), ParseFloat(parts[3])));
            }
            else if (parts[0] == "f" && parts.Length >= 4)
            {
                // Open3D writes triangulated faces already ("f v1 v2 v3"
                // or "f v1//n1 v2//n2 v3//n3") -- if you ever feed this a
                // quad-faced OBJ from elsewhere, add a fan-triangulation
                // loop here instead of the direct 3-index parse below.
                for (int i = 1; i <= 3; i++)
                {
                    string token = parts[i].Split('/')[0];
                    int idx = int.Parse(token, CultureInfo.InvariantCulture) - 1; // OBJ is 1-indexed
                    triangles.Add(idx);
                }
            }
        }

        var mesh = new Mesh();
        mesh.indexFormat = vertices.Count > 65000
            ? UnityEngine.Rendering.IndexFormat.UInt32
            : UnityEngine.Rendering.IndexFormat.UInt16;
        mesh.SetVertices(vertices);
        mesh.SetTriangles(triangles, 0);
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