// Built-in render pipeline. Shows per-vertex colours (TripoSR replacement meshes) with
// normal Standard lighting. Has a _Color tint so RegionObject's highlight (material.color) works.
// For BUILDS: put this file in a Resources folder (or add it to Always Included Shaders),
// otherwise Shader.Find cannot see it. In the Editor any location works.
Shader "Aerosense/VertexColorLit"
{
    Properties
    {
        _Color ("Tint", Color) = (1,1,1,1)
        _Glossiness ("Smoothness", Range(0,1)) = 0.15
        _Highlight ("Regenerated highlight", Range(0,1)) = 0
        _HighlightColor ("Highlight colour", Color) = (1,0.45,0,1)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 200
        CGPROGRAM
        #pragma surface surf Standard fullforwardshadows
        #pragma target 3.0

        struct Input { float4 color : COLOR; float3 viewDir; };
        fixed4 _Color;
        half _Glossiness;
        half _Highlight;
        fixed4 _HighlightColor;

        void surf (Input IN, inout SurfaceOutputStandard o)
        {
            float3 c = IN.color.rgb;
            #ifndef UNITY_COLORSPACE_GAMMA
            c = GammaToLinearSpace(c);   // vertex colours are authored in sRGB
            #endif
            o.Albedo = c * _Color.rgb;
            o.Metallic = 0;
            o.Smoothness = _Glossiness;
            o.Alpha = 1;
            // orange glow marking "this part was regenerated": faint overall + strong rim
            half rim = 1.0 - saturate(dot(normalize(IN.viewDir), o.Normal));
            o.Emission = _HighlightColor.rgb * _Highlight * (0.12 + 0.88 * pow(rim, 2.0));
        }
        ENDCG
    }
    FallBack "Diffuse"
}
