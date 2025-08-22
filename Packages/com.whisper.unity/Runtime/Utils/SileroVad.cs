using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

namespace Whisper.Utils
{
    /// <summary>
    /// Silero VAD implementation using ONNX runtime
    /// </summary>
    public class SileroVad : IDisposable
    {
        private InferenceSession _session;
        private float[,,] _state;
        private readonly int _sampleRate;
        private readonly int _windowSize;
        private readonly float _threshold;
        private bool _disposed;

        public SileroVad(string modelPath, int sampleRate = 16000, int windowSize = 512, float threshold = 0.5f)
        {
            _sampleRate = sampleRate;
            _windowSize = windowSize;
            _threshold = threshold;

            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"Silero VAD model not found at: {modelPath}");
            }

            try
            {
                var sessionOptions = new SessionOptions
                {
                    LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR,
                    InterOpNumThreads = 1,
                    IntraOpNumThreads = 1
                };

                _session = new InferenceSession(modelPath, sessionOptions);
                Reset();
                
                LogUtils.Verbose("Silero VAD model initialized successfully");
            }
            catch (Exception e)
            {
                LogUtils.Error($"Failed to initialize Silero VAD model: {e.Message}");
                throw;
            }
        }

        public bool IsSpeech(float[] samples)
        {
            if (_disposed || _session == null)
                return false;

            try
            {
                var probability = GetSpeechProbability(samples);
                return probability > _threshold;
            }
            catch (Exception e)
            {
                LogUtils.Error($"Silero VAD inference failed: {e.Message}");
                return false;
            }
        }

        public float GetSpeechProbability(float[] samples)
        {
            if (_disposed || _session == null || samples.Length != _windowSize)
            {
                return 0.0f;
            }

            try
            {
                // Create input tensors
                var inputTensor = new DenseTensor<float>(samples, new[] { 1, _windowSize });
                var stateTensor = new DenseTensor<float>(ToFlatArray(_state), new[] { 2, 1, 128 });
                var srTensor = new DenseTensor<long>(new long[] { _sampleRate }, new[] { 1 });

                // Prepare inputs
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input", inputTensor),
                    NamedOnnxValue.CreateFromTensor("sr", srTensor),
                    NamedOnnxValue.CreateFromTensor("state", stateTensor)
                };

                // Run inference
                using var results = _session.Run(inputs);

                // Get outputs
                var output = results.First(x => x.Name == "output").AsEnumerable<float>().ToArray();
                var newState = results.First(x => x.Name == "stateN").AsTensor<float>();

                // Update state
                UpdateState(newState.ToArray());

                return output[0];
            }
            catch (Exception e)
            {
                LogUtils.Error($"Silero VAD inference error: {e.Message}");
                return 0.0f;
            }
        }

        private float[] ToFlatArray(float[,,] array)
        {
            var result = new float[2 * 1 * 128];
            Buffer.BlockCopy(array, 0, result, 0, result.Length * sizeof(float));
            return result;
        }

        private void UpdateState(float[] flatState)
        {
            if (flatState.Length == 2 * 1 * 128)
            {
                _state = new float[2, 1, 128];
                Buffer.BlockCopy(flatState, 0, _state, 0, flatState.Length * sizeof(float));
            }
        }

        public void Reset()
        {
            _state = new float[2, 1, 128];
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _session?.Dispose();
                _disposed = true;
                LogUtils.Verbose("Silero VAD resources disposed");
            }
        }
    }
}