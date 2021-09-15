/**
 * Copyright (c) 2021 LG Electronics, Inc.
 *
 * This software contains code licensed as described in LICENSE.
 *
 */

namespace Simulator.Sensors
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using Bridge;
    using Bridge.Data;
    using Components;
    using PointCloud;
    using UI;
    using Unity.Collections.LowLevel.Unsafe;
    using Unity.Profiling;
    using UnityEngine;
    using UnityEngine.Rendering;
    using UnityEngine.Rendering.HighDefinition;
    using UnityEngine.Serialization;
    using Utilities;

    [SensorType("Lidar", new[] { typeof(PointCloudData) })]
    [RequireComponent(typeof(Camera))]
    public partial class LidarSensor : SensorBase, IPointCloudGenerator
    {
        private static readonly Matrix4x4 LidarTransform = new Matrix4x4(new Vector4(0, -1, 0, 0), new Vector4(0, 0, 1, 0), new Vector4(1, 0, 0, 0), Vector4.zero);

        private static class Properties
        {
            public static readonly int InputCubemapTexture = Shader.PropertyToID("_InputCubemapTexture");
            public static readonly int Output = Shader.PropertyToID("_Output");
            public static readonly int LatitudeAngles = Shader.PropertyToID("_LatitudeAngles");
            public static readonly int LaserCount = Shader.PropertyToID("_LaserCount");
            public static readonly int MeasuresPerRotation = Shader.PropertyToID("_MeasurementsPerRotation");
            public static readonly int Origin = Shader.PropertyToID("_Origin");
            public static readonly int Transform = Shader.PropertyToID("_Transform");
            public static readonly int RotMatrix = Shader.PropertyToID("_RotMatrix");
            public static readonly int PackedVec = Shader.PropertyToID("_PackedVec");
            public static readonly int PointCloud = Shader.PropertyToID("_PointCloud");
            public static readonly int LocalToWorld = Shader.PropertyToID("_LocalToWorld");
            public static readonly int Size = Shader.PropertyToID("_Size");
            public static readonly int Color = Shader.PropertyToID("_Color");
            public static readonly int AntialiasingThreshold = Shader.PropertyToID("_AntialiasingThreshold");
        }

        public enum MessageTypes
        {
            PointCloud2,
            LaserScan
        }

        [HideInInspector]
        public int TemplateIndex;

        [SensorParameter]
        public List<float> VerticalRayAngles;

        [SensorParameter]
        [Range(1, 128)]
        public int LaserCount = 32;

        [SensorParameter]
        [Range(1.0f, 45.0f)]
        public float FieldOfView = 40.0f;

        [SensorParameter]
        [Range(-45.0f, 45.0f)]
        public float CenterAngle = 10.0f;

        [SensorParameter]
        [Range(0.01f, 1000f)]
        public float MinDistance = 0.5f; // meters

        [SensorParameter]
        [Range(0.01f, 2000f)]
        public float MaxDistance = 100.0f; // meters

        [SensorParameter]
        [Range(1, 30)]
        public float RotationFrequency = 5.0f; // Hz

        [SensorParameter]
        [Range(18, 6000)]
        public int MeasurementsPerRotation = 1500; // for each ray

        [SensorParameter]
        public bool Compensated = true;

        [SensorParameter]
        [Range(1, 10)]
        public float PointSize = 2.0f;

        [SensorParameter]
        public Color PointColor = Color.red;

        [SensorParameter]
        public int CubemapSize = 1024;

        [SensorParameter]
        [Range(0f, 360f)]
        public float ForwardAngle = 0f;

        [SensorParameter]
        [Range(0f, 360f)]
        public float HorizontalAngle = 360f;

        [SensorParameter]
        public float AntialiasingThreshold = 0.1f;

        [SensorParameter]
        public MessageTypes MessageType = MessageTypes.PointCloud2;

        private readonly int faceMask = 1 << (int) CubemapFace.PositiveX | 1 << (int) CubemapFace.NegativeX |
                                        1 << (int) CubemapFace.PositiveY | 1 << (int) CubemapFace.NegativeY |
                                        1 << (int) CubemapFace.PositiveZ | 1 << (int) CubemapFace.NegativeZ;

        public ComputeShader computeShader;

        [FormerlySerializedAs("PerformanceLoad")]
        [SensorParameter]
        public float performanceLoad = 1.0f;

        public override float PerformanceLoad => performanceLoad;
        public override SensorDistributionType DistributionType => SensorDistributionType.ClientOnly;

        private SensorRenderTarget renderTarget;

        private Material PointCloudMaterial;
        private ShaderTagId passId;
        private ComputeShader cs;

        private BridgeInstance Bridge;
        private Publisher<PointCloudData> PointCloudPublish;
        private Publisher<LaserScanData> LaserScanPublish;

        uint Sequence;
        private float NextCaptureTime;

        private Camera sensorCamera;
        private HDAdditionalCameraData hdAdditionalCameraData;
        private Vector4[] Points;

        private ComputeBuffer PointCloudBuffer;
        private ComputeBuffer LatitudeAnglesBuffer;

        private GpuReadbackPool<GpuReadbackData<Vector4>, Vector4> ReadbackPool;

        private ProfilerMarker RenderMarker = new ProfilerMarker("Lidar.Render");
        private ProfilerMarker ComputeMarker = new ProfilerMarker("Lidar.Compute");
        private ProfilerMarker VisualizeMarker = new ProfilerMarker("Lidar.Visualzie");

        private float MaxAngle;
        private float MinAngle;
        private float DeltaLongitudeAngle;
        private float StartLongitudeOffset;
        private int CurrentLaserCount;
        private int CurrentUsedMeasurementsPerRotation;
        private int UsedMeasurementsPerRotation;
        private float CurrentFieldOfView;
        private List<float> CurrentVerticalRayAngles;
        private float CurrentCenterAngle;
        private float CurrentForwardAngle;
        private float CurrentHorizontalAngle;
        private float CurrentMinDistance;
        private float CurrentMaxDistance;

        private Camera SensorCamera
        {
            get
            {
                if (sensorCamera == null)
                    sensorCamera = GetComponent<Camera>();

                return sensorCamera;
            }
        }

        private HDAdditionalCameraData HDAdditionalCameraData
        {
            get
            {
                if (hdAdditionalCameraData == null)
                    hdAdditionalCameraData = GetComponent<HDAdditionalCameraData>();

                return hdAdditionalCameraData;
            }
        }

        protected override void Initialize()
        {
            SensorCamera.enabled = false;
            passId = new ShaderTagId("SimulatorLidarPass");
            cs = Instantiate(computeShader);
            PointCloudMaterial = new Material(RuntimeSettings.Instance.PointCloudShader);
            HDAdditionalCameraData.hasPersistentHistory = true;
            HDAdditionalCameraData.customRender += CustomRender;
            ReadbackPool = new GpuReadbackPool<GpuReadbackData<Vector4>, Vector4>();
            ReadbackPool.Initialize(GetTotalRayCount(), OnReadbackComplete);

            Reset();
        }

        protected override void Deinitialize()
        {
            renderTarget?.Release();
            renderTarget = null;

            PointCloudBuffer?.Release();
            PointCloudBuffer = null;

            LatitudeAnglesBuffer?.Release();
            LatitudeAnglesBuffer = null;

            ReadbackPool?.Dispose();
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

        private int GetTotalRayCount()
        {
            var usedMeasurementsPerRotation = (int) (MeasurementsPerRotation * HorizontalAngle / 360f);
            var totalCount = LaserCount * usedMeasurementsPerRotation;
            return totalCount;
        }

        private void Reset()
        {
            if (PointCloudBuffer != null)
            {
                PointCloudBuffer.Release();
                PointCloudBuffer = null;
            }

            if (LatitudeAnglesBuffer != null)
            {
                LatitudeAnglesBuffer.Release();
                LatitudeAnglesBuffer = null;
            }

            UsedMeasurementsPerRotation = (int) (MeasurementsPerRotation * HorizontalAngle / 360f);
            DeltaLongitudeAngle = HorizontalAngle / 360f / UsedMeasurementsPerRotation;
            StartLongitudeOffset = ForwardAngle / 360f - HorizontalAngle / 720f + 0.5f;
            MaxAngle = Mathf.Abs(CenterAngle) + FieldOfView / 2.0f;

            float startLatitudeAngle;
            // Assuming center of view frustum is horizontal, find the vertical FOV (of view frustum) that can encompass the tilted Lidar FOV.
            // "MaxAngle" is half of the vertical FOV of view frustum.
            if (VerticalRayAngles.Count == 0)
            {
                MaxAngle = Mathf.Abs(CenterAngle) + FieldOfView / 2.0f;
                MinAngle = Mathf.Abs(CenterAngle) - FieldOfView / 2.0f;

                startLatitudeAngle = 90.0f + MaxAngle;
                //If the Lidar is tilted up, ignore lower part of the vertical FOV.
                if (CenterAngle < 0.0f)
                {
                    startLatitudeAngle -= MaxAngle * 2.0f - FieldOfView;
                }
            }
            else
            {
                LaserCount = VerticalRayAngles.Count;
                startLatitudeAngle = 90.0f - VerticalRayAngles.Min();
                var endLatitudeAngle = 90.0f - VerticalRayAngles.Max();
                FieldOfView = startLatitudeAngle - endLatitudeAngle;
                MaxAngle = Mathf.Max(startLatitudeAngle - 90.0f, 90.0f - endLatitudeAngle);
                MinAngle = Mathf.Min(startLatitudeAngle - 90.0f, 90.0f - endLatitudeAngle);
            }

            CurrentVerticalRayAngles = new List<float>(VerticalRayAngles);
            CurrentLaserCount = LaserCount;
            CurrentUsedMeasurementsPerRotation = UsedMeasurementsPerRotation;
            CurrentHorizontalAngle = HorizontalAngle;
            CurrentForwardAngle = ForwardAngle;
            CurrentFieldOfView = FieldOfView;
            CurrentCenterAngle = CenterAngle;
            CurrentMinDistance = MinDistance;
            CurrentMaxDistance = MaxDistance;

            var latitudeAngles = new float[LaserCount];

            if (VerticalRayAngles.Count == 0)
            {
                if (LaserCount == 1)
                {
                    latitudeAngles[0] = 1f - (90f + CenterAngle) * Mathf.Deg2Rad / Mathf.PI;
                }
                else
                {
                    var deltaLatitudeAngle = FieldOfView / LaserCount;
                    var index = 0;
                    var angle = startLatitudeAngle;
                    while (index < LaserCount)
                    {
                        latitudeAngles[index] = 1f - angle * Mathf.Deg2Rad / Mathf.PI;
                        index++;
                        angle -= deltaLatitudeAngle;
                    }
                }
            }
            else
            {
                for (var index = 0; index < LaserCount; index++)
                {
                    latitudeAngles[index] = 1f - ((90.0f - VerticalRayAngles[index]) * Mathf.Deg2Rad / Mathf.PI);
                }
            }

            LatitudeAnglesBuffer = new ComputeBuffer(LaserCount, sizeof(float));
            LatitudeAnglesBuffer.SetData(latitudeAngles);

            var totalCount = LaserCount * UsedMeasurementsPerRotation;
            PointCloudBuffer = new ComputeBuffer(totalCount, UnsafeUtility.SizeOf<Vector4>());
            Points = new Vector4[totalCount];
            ReadbackPool.Resize(totalCount);

            if (PointCloudMaterial != null)
                PointCloudMaterial.SetBuffer(Properties.PointCloud, PointCloudBuffer);
        }

        private void CustomRender(ScriptableRenderContext context, HDCamera hd)
        {
            var cmd = CommandBufferPool.Get();

            void RenderPointCloud(CubemapFace face)
            {
                PointCloudManager.RenderLidar(context, cmd, hd, renderTarget.ColorHandle, renderTarget.DepthHandle, face);
            }

            RenderMarker.Begin();
            SensorPassRenderer.Render(context, cmd, hd, renderTarget, passId, Color.clear, RenderPointCloud);
            RenderMarker.End();

            ComputeMarker.Begin();
            var kernel = cs.FindKernel(Compensated ? "CubeComputeComp" : "CubeCompute");
            cmd.SetComputeTextureParam(cs, kernel, Properties.InputCubemapTexture, renderTarget.ColorHandle);
            cmd.SetComputeBufferParam(cs, kernel, Properties.Output, PointCloudBuffer);
            cmd.SetComputeBufferParam(cs, kernel, Properties.LatitudeAngles, LatitudeAnglesBuffer);
            cmd.SetComputeIntParam(cs, Properties.LaserCount, LaserCount);
            cmd.SetComputeIntParam(cs, Properties.MeasuresPerRotation, UsedMeasurementsPerRotation);
            cmd.SetComputeFloatParam(cs, Properties.AntialiasingThreshold, AntialiasingThreshold);
            cmd.SetComputeVectorParam(cs, Properties.Origin, SensorCamera.transform.position);
            cmd.SetComputeMatrixParam(cs, Properties.RotMatrix, Matrix4x4.Rotate(transform.rotation));
            cmd.SetComputeMatrixParam(cs, Properties.Transform, transform.worldToLocalMatrix);
            cmd.SetComputeVectorParam(cs, Properties.PackedVec, new Vector4(MaxDistance, DeltaLongitudeAngle, MinDistance, StartLongitudeOffset));
            cmd.DispatchCompute(cs, kernel, HDRPUtilities.GetGroupSize(UsedMeasurementsPerRotation, 8), HDRPUtilities.GetGroupSize(LaserCount, 8), 1);
            ComputeMarker.End();

            context.ExecuteCommandBuffer(cmd);
            cmd.Clear();
            CommandBufferPool.Release(cmd);
        }

        public override void OnBridgeSetup(BridgeInstance bridge)
        {
            Bridge = bridge;

            if (MessageType == MessageTypes.LaserScan)
            {
                if (LaserCount == 1)
                    LaserScanPublish = bridge.AddPublisher<LaserScanData>(Topic);
                else
                {
                    Debug.LogError("LaserScan message type is only valid when LaserCount is 1. Falling back to PointCloud2.");
                    PointCloudPublish = bridge.AddPublisher<PointCloudData>(Topic);
                }
            }
            else
                PointCloudPublish = bridge.AddPublisher<PointCloudData>(Topic);
        }

        private void Update()
        {
            if (LaserCount != CurrentLaserCount ||
                UsedMeasurementsPerRotation != CurrentUsedMeasurementsPerRotation ||
                !Mathf.Approximately(FieldOfView, CurrentFieldOfView) ||
                !Mathf.Approximately(CenterAngle, CurrentCenterAngle) ||
                !Mathf.Approximately(ForwardAngle, CurrentForwardAngle) ||
                !Mathf.Approximately(HorizontalAngle, CurrentHorizontalAngle) ||
                !Mathf.Approximately(MinDistance, CurrentMinDistance) ||
                !Mathf.Approximately(MaxDistance, CurrentMaxDistance) ||
                !VerticalRayAngles.SequenceEqual(CurrentVerticalRayAngles))
            {
                Reset();
            }

            SensorCamera.fieldOfView = FieldOfView;
            SensorCamera.nearClipPlane = MinDistance;
            SensorCamera.farClipPlane = MaxDistance;

            CheckTexture();
            CheckCapture();
        }

        private void CheckTexture()
        {
            if (renderTarget != null && (!renderTarget.IsCube || !renderTarget.IsValid(CubemapSize, CubemapSize)))
            {
                renderTarget.Release();
                renderTarget = null;
            }

            if (renderTarget == null)
            {
                renderTarget = SensorRenderTarget.CreateCube(CubemapSize, CubemapSize, faceMask);
                SensorCamera.targetTexture = null;
            }
        }

        private void RenderCamera()
        {
            SensorCamera.Render();
        }

        private void CheckCapture()
        {
            if (Time.time >= NextCaptureTime)
            {
                RenderCamera();
                ReadbackPool.StartReadback(PointCloudBuffer);

                if (NextCaptureTime < Time.time - Time.deltaTime)
                {
                    NextCaptureTime = Time.time + 1.0f / RotationFrequency;
                }
                else
                {
                    NextCaptureTime += 1.0f / RotationFrequency;
                }
            }

            ReadbackPool.Process();
        }

        private void OnReadbackComplete(GpuReadbackData<Vector4> data)
        {
            if (!(Bridge is {Status: Status.Connected}))
                return;

            var worldToLocal = LidarTransform;
            if (Compensated)
                worldToLocal = worldToLocal * transform.worldToLocalMatrix;

            if (PointCloudPublish != null)
            {
                var message = new PointCloudData()
                {
                    Name = Name,
                    Frame = Frame,
                    Time = data.captureTime,
                    Sequence = Sequence,

                    LaserCount = CurrentLaserCount,
                    Transform = worldToLocal,
                    NativePoints = data.gpuData,
                    PointCount = Points.Length
                };

                if (BridgeMessageDispatcher.Instance.TryQueueTask(PointCloudPublish, message))
                    Sequence++;
            }
            else if (LaserScanPublish != null)
            {
                var maxRad = (1f - StartLongitudeOffset - 0.5f) * 2 * Mathf.PI;
                var minRad = maxRad - HorizontalAngle * Mathf.Deg2Rad;

                // TODO: add support for native arrays to LaserScanData to skip this copy
                data.gpuData.CopyTo(Points);

                var message = new LaserScanData()
                {
                    Name = Name,
                    Frame = Frame,
                    Time = data.captureTime,
                    Sequence = Sequence,
                    MinAngle = minRad,
                    MaxAngle = maxRad,
                    AngleStep = DeltaLongitudeAngle * 2 * Mathf.PI,
                    RangeMin = MinDistance,
                    RangeMax = MaxDistance,
                    TimeIncrement = 0f,
                    ScanTime = 1f / RotationFrequency,
                    Transform = worldToLocal,
                    Points = Points,
                };

                if (BridgeMessageDispatcher.Instance.TryQueueTask(LaserScanPublish, message))
                    Sequence++;
            }
        }

        public bool Save(string path)
        {
            CheckTexture();
            RenderCamera();

            PointCloudBuffer.GetData(Points);

            var worldToLocal = LidarTransform;
            if (Compensated)
                worldToLocal = worldToLocal * transform.worldToLocalMatrix;

            try
            {
                using (var writer = new PcdWriter(path))
                {
                    foreach (var point in Points)
                    {
                        if (point != Vector4.zero)
                            writer.Write(worldToLocal.MultiplyPoint3x4(point), point.w);
                    }
                }

                return true;
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                return false;
            }
        }

        public override void OnVisualize(Visualizer visualizer)
        {
            VisualizeMarker.Begin();
            var lidarToWorld = Compensated ? Matrix4x4.identity : transform.localToWorldMatrix;
            PointCloudMaterial.SetMatrix(Properties.LocalToWorld, lidarToWorld);
            PointCloudMaterial.SetFloat(Properties.Size, PointSize * Utility.GetDpiScale());
            PointCloudMaterial.SetColor(Properties.Color, PointColor);
            Graphics.DrawProcedural(PointCloudMaterial, new Bounds(transform.position, MaxDistance * Vector3.one), MeshTopology.Points, PointCloudBuffer.count, layer: LayerMask.NameToLayer("Sensor"));

            VisualizeMarker.End();
        }

        public override void OnVisualizeToggle(bool state)
        {
            //
        }

        #region IPointCloudGenerator

        public void ApplySettings(LidarTemplate template)
        {
            Initialize();
            CubemapSize = 2048;
            AntialiasingThreshold = 0f;
            LaserCount = template.LaserCount;
            MinDistance = template.MinDistance;
            MaxDistance = template.MaxDistance;
            RotationFrequency = template.RotationFrequency;
            MeasurementsPerRotation = template.MeasurementsPerRotation;
            FieldOfView = template.FieldOfView;
            VerticalRayAngles = template.VerticalRayAngles == null ? new List<float>() : new List<float>(template.VerticalRayAngles);
            CenterAngle = template.CenterAngle;
            SensorCamera.fieldOfView = FieldOfView;
            SensorCamera.nearClipPlane = MinDistance;
            SensorCamera.farClipPlane = MaxDistance;
            Reset();
            CheckTexture();
        }

        public Vector4[] GeneratePoints(Vector3 position)
        {
            transform.position = position;
            Compensated = true;
            RenderCamera();
            PointCloudBuffer.GetData(Points);
            return Points;
        }

        public void Cleanup()
        {
            Deinitialize();
        }

        #endregion
    }
}