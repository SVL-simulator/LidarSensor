/**
 * Copyright (c) 2019 LG Electronics, Inc.
 *
 * This software contains code licensed as described in LICENSE.
 *
 */

using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;
using Simulator.Bridge;
using Simulator.Utilities;
using PointCloudData = Simulator.Bridge.Data.PointCloudData;

namespace Simulator.Sensors
{
    using UnityEngine.Rendering;

    [SensorType("Lidar", new[] { typeof(PointCloudData) })]
    public partial class LidarSensor : LidarSensorBase
    {
        private static class Properties
        {
            public static readonly int Input = Shader.PropertyToID("_Input");
            public static readonly int Output = Shader.PropertyToID("_Output");
            public static readonly int SinLatitudeAngles = Shader.PropertyToID("_SinLatitudeAngles");
            public static readonly int CosLatitudeAngles = Shader.PropertyToID("_CosLatitudeAngles");
            public static readonly int Index = Shader.PropertyToID("_Index");
            public static readonly int Count = Shader.PropertyToID("_Count");
            public static readonly int LaserCount = Shader.PropertyToID("_LaserCount");
            public static readonly int MeasuresPerRotation = Shader.PropertyToID("_MeasurementsPerRotation");
            public static readonly int Origin = Shader.PropertyToID("_Origin");
            public static readonly int Transform = Shader.PropertyToID("_Transform");
            public static readonly int CameraToWorld = Shader.PropertyToID("_CameraToWorld");
            public static readonly int ScaleDistance = Shader.PropertyToID("_ScaleDistance");
            public static readonly int TexSize = Shader.PropertyToID("_TexSize");
            public static readonly int LongitudeAngles = Shader.PropertyToID("_LongitudeAngles");
        }
        
        public void ApplyTemplate()
        {
            var values = Template.Templates[TemplateIndex];
            LaserCount = values.LaserCount;
            MinDistance = values.MinDistance;
            MaxDistance = values.MaxDistance;
            RotationFrequency = values.RotationFrequency;
            MeasurementsPerRotation = values.MeasurementsPerRotation;
            FieldOfView = values.FieldOfView;
            VerticalRayAngles = new List<float>(values.VerticalRayAngles);
            CenterAngle = values.CenterAngle;
        }

        public override void Reset()
        {
            Active.ForEach(req =>
            {
                req.TextureSet.Release();
            });
            Active.Clear();

            foreach (var tex in AvailableRenderTextures)
            {
                tex.Release();
            };
            AvailableRenderTextures.Clear();

            foreach (var tex in AvailableTextures)
            {
                Destroy(tex);
            };
            AvailableTextures.Clear();

            if (PointCloudBuffer != null)
            {
                PointCloudBuffer.Release();
                PointCloudBuffer = null;
            }

            if (CosLatitudeAnglesBuffer != null)
            {
                CosLatitudeAnglesBuffer.Release();
                CosLatitudeAnglesBuffer = null;
            }
            if (SinLatitudeAnglesBuffer != null)
            {
                SinLatitudeAnglesBuffer.Release();
                SinLatitudeAnglesBuffer = null;
            }

            AngleStart = 0.0f;
            // Assuming center of view frustum is horizontal, find the vertical FOV (of view frustum) that can encompass the tilted Lidar FOV.
            // "MaxAngle" is half of the vertical FOV of view frustum.
            if (VerticalRayAngles.Count == 0)
            {
                MaxAngle = Mathf.Abs(CenterAngle) + FieldOfView / 2.0f;

                StartLatitudeAngle = 90.0f + MaxAngle;
                //If the Lidar is tilted up, ignore lower part of the vertical FOV.
                if (CenterAngle < 0.0f)
                {
                    StartLatitudeAngle -= MaxAngle * 2.0f - FieldOfView;
                }
                EndLatitudeAngle = StartLatitudeAngle - FieldOfView;
            }
            else
            {
                LaserCount = VerticalRayAngles.Count;
                StartLatitudeAngle = 90.0f - VerticalRayAngles.Min();
                EndLatitudeAngle = 90.0f - VerticalRayAngles.Max();
                FieldOfView = StartLatitudeAngle - EndLatitudeAngle;
                MaxAngle = Mathf.Max(StartLatitudeAngle - 90.0f, 90.0f - EndLatitudeAngle);
            }

            float startLongitudeAngle = 90.0f + HorizontalAngleLimit / 2.0f;
            SinStartLongitudeAngle = Mathf.Sin(startLongitudeAngle * Mathf.Deg2Rad);
            CosStartLongitudeAngle = Mathf.Cos(startLongitudeAngle * Mathf.Deg2Rad);

            // The MaxAngle above is the calculated at the center of the view frustum.
            // Because the scan curve for a particular laser ray is a hyperbola (intersection of a conic surface and a vertical plane),
            // the vertical FOV should be enlarged toward left and right ends.
            float startFovAngle = CalculateFovAngle(StartLatitudeAngle, startLongitudeAngle);
            float endFovAngle = CalculateFovAngle(EndLatitudeAngle, startLongitudeAngle);
            MaxAngle = Mathf.Max(MaxAngle, Mathf.Max(startFovAngle, endFovAngle));

            // Calculate sin/cos of latitude angle of each ray.
            SinLatitudeAngles = new float[LaserCount];
            CosLatitudeAngles = new float[LaserCount];

            int totalCount = LaserCount * MeasurementsPerRotation;
            PointCloudBuffer = new ComputeBuffer(totalCount, UnsafeUtility.SizeOf<Vector4>());
            CosLatitudeAnglesBuffer = new ComputeBuffer(LaserCount, sizeof(float));
            SinLatitudeAnglesBuffer = new ComputeBuffer(LaserCount, sizeof(float));

            if (PointCloudMaterial != null)
                PointCloudMaterial.SetBuffer("_PointCloud", PointCloudBuffer);

            Points = new Vector4[totalCount];

            CurrentLaserCount = LaserCount;
            CurrentMeasurementsPerRotation = MeasurementsPerRotation;
            CurrentFieldOfView = FieldOfView;
            CurrentVerticalRayAngles = new List<float>(VerticalRayAngles);
            CurrentCenterAngle = CenterAngle;
            CurrentMinDistance = MinDistance;
            CurrentMaxDistance = MaxDistance;

            IgnoreNewRquests = 0;

            // If VerticalRayAngles array is not provided, use uniformly distributed angles.
            if (VerticalRayAngles.Count == 0)
            {
                float deltaLatitudeAngle = FieldOfView / LaserCount;
                int index = 0;
                float angle = StartLatitudeAngle;
                while (index < LaserCount)
                {
                    SinLatitudeAngles[index] = Mathf.Sin(angle * Mathf.Deg2Rad);
                    CosLatitudeAngles[index] = Mathf.Cos(angle * Mathf.Deg2Rad);
                    index++;
                    angle -= deltaLatitudeAngle;
                }
            }
            else
            {
                for (int index = 0; index < LaserCount; index++)
                {
                    SinLatitudeAngles[index] = Mathf.Sin((90.0f - VerticalRayAngles[index]) * Mathf.Deg2Rad);
                    CosLatitudeAngles[index] = Mathf.Cos((90.0f - VerticalRayAngles[index]) * Mathf.Deg2Rad);
                }
            }

            CosLatitudeAnglesBuffer.SetData(CosLatitudeAngles);
            SinLatitudeAnglesBuffer.SetData(SinLatitudeAngles);

            int count = Mathf.CeilToInt(HorizontalAngleLimit / (360.0f / MeasurementsPerRotation));
            DeltaLongitudeAngle = HorizontalAngleLimit / count;

            // Enlarged the texture by some factors to mitigate alias.
            RenderTextureHeight = 16 * Mathf.CeilToInt(2.0f * MaxAngle * LaserCount / FieldOfView);
            RenderTextureWidth = 8 * Mathf.CeilToInt(HorizontalAngleLimit / (360.0f / MeasurementsPerRotation));

            // View frustum size at the near plane.
            float frustumWidth = 2 * MinDistance * Mathf.Tan(HorizontalAngleLimit / 2.0f * Mathf.Deg2Rad);
            float frustumHeight = 2 * MinDistance * Mathf.Tan(MaxAngle * Mathf.Deg2Rad);
            XScale = frustumWidth / RenderTextureWidth;
            YScale = frustumHeight / RenderTextureHeight;

            // construct custom aspect ratio projection matrix
            // math from https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix

            float v = 1.0f / Mathf.Tan(MaxAngle * Mathf.Deg2Rad);
            float h = 1.0f / Mathf.Tan(HorizontalAngleLimit * Mathf.Deg2Rad / 2.0f);
            float a = (MaxDistance + MinDistance) / (MinDistance - MaxDistance);
            float b = 2.0f * MaxDistance * MinDistance / (MinDistance - MaxDistance);

            var projection = new Matrix4x4(
                new Vector4(h, 0, 0, 0),
                new Vector4(0, v, 0, 0),
                new Vector4(0, 0, a, -1),
                new Vector4(0, 0, b, 0));

            SensorCamera.nearClipPlane = MinDistance;
            SensorCamera.farClipPlane = MaxDistance;
            SensorCamera.projectionMatrix = projection;
        }

        protected override void EndReadRequest(CommandBuffer cmd, ReadRequest req)
        {
            EndReadMarker.Begin();

            var kernel = cs.FindKernel(Compensated ? "LidarComputeComp" : "LidarCompute");
            cmd.SetComputeTextureParam(cs, kernel, Properties.Input, req.TextureSet.ColorTexture);
            cmd.SetComputeBufferParam(cs, kernel, Properties.Output, PointCloudBuffer);
            cmd.SetComputeBufferParam(cs, kernel, Properties.SinLatitudeAngles, SinLatitudeAnglesBuffer);
            cmd.SetComputeBufferParam(cs, kernel, Properties.CosLatitudeAngles, CosLatitudeAnglesBuffer);
            cmd.SetComputeIntParam(cs, Properties.Index, req.Index);
            cmd.SetComputeIntParam(cs, Properties.Count, req.Count);
            cmd.SetComputeIntParam(cs, Properties.LaserCount, CurrentLaserCount);
            cmd.SetComputeIntParam(cs, Properties.MeasuresPerRotation, CurrentMeasurementsPerRotation);
            cmd.SetComputeVectorParam(cs, Properties.Origin, req.Origin);
            cmd.SetComputeMatrixParam(cs, Properties.Transform, req.Transform);
            cmd.SetComputeMatrixParam(cs, Properties.CameraToWorld, req.CameraToWorldMatrix);
            cmd.SetComputeVectorParam(cs, Properties.ScaleDistance, new Vector4(XScale, YScale, MinDistance, MaxDistance));
            cmd.SetComputeVectorParam(cs, Properties.TexSize, new Vector4(RenderTextureWidth, RenderTextureHeight, 1f / RenderTextureWidth, 1f / RenderTextureHeight));
            cmd.SetComputeVectorParam(cs, Properties.LongitudeAngles, new Vector4(SinStartLongitudeAngle, CosStartLongitudeAngle, DeltaLongitudeAngle, 0f));
            cmd.DispatchCompute(cs, kernel, HDRPUtilities.GetGroupSize(req.Count, 8), HDRPUtilities.GetGroupSize(LaserCount, 8), 1);

            EndReadMarker.End();
        }

        protected override void SendMessage()
        {
            if (Bridge != null && Bridge.Status == Status.Connected)
            {
                var worldToLocal = LidarTransform;
                if (Compensated)
                {
                    worldToLocal = worldToLocal * transform.worldToLocalMatrix;
                }

                PointCloudBuffer.GetData(Points);

                Task.Run(() =>
                {
                    Publish(new PointCloudData()
                    {
                        Name = Name,
                        Frame = Frame,
                        Time = SimulatorManager.Instance.CurrentTime,
                        Sequence = SendSequence++,

                        LaserCount = CurrentLaserCount,
                        Transform = worldToLocal,
                        Points = Points,
                    });
                });
            }
        }

        void OnValidate()
        {
            if (TemplateIndex != 0)
            {
                var values = Template.Templates[TemplateIndex];
                if (LaserCount != values.LaserCount ||
                    MinDistance != values.MinDistance ||
                    MaxDistance != values.MaxDistance ||
                    RotationFrequency != values.RotationFrequency ||
                    MeasurementsPerRotation != values.MeasurementsPerRotation ||
                    FieldOfView != values.FieldOfView ||
                    CenterAngle != values.CenterAngle ||
                    !Enumerable.SequenceEqual(VerticalRayAngles, values.VerticalRayAngles))
                {
                    TemplateIndex = 0;
                }
            }
        }
    }
}
