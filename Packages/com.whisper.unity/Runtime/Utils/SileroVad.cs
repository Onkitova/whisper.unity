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
                LogUtils.Verbose($"Silero VAD probability: {probability}");
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

        /// <summary>
        /// Evaluate probability using an external recurrent state (shadow or fresh) without modifying internal state.
        /// Updates the provided state in-place with the new state produced by the model.
        /// </summary>
        public float EvaluateWithState(float[] samples, ref float[,,] externalState)
        {
            if (_disposed || _session == null || samples.Length != _windowSize)
                return 0f;

            try
            {
                var inputTensor = new DenseTensor<float>(samples, new[] { 1, _windowSize });
                // flatten external state
                var flatExternal = new float[2 * 1 * 128];
                Buffer.BlockCopy(externalState, 0, flatExternal, 0, flatExternal.Length * sizeof(float));
                var stateTensor = new DenseTensor<float>(flatExternal, new[] { 2, 1, 128 });
                var srTensor = new DenseTensor<long>(new long[] { _sampleRate }, new[] { 1 });

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input", inputTensor),
                    NamedOnnxValue.CreateFromTensor("sr", srTensor),
                    NamedOnnxValue.CreateFromTensor("state", stateTensor)
                };

                using var results = _session.Run(inputs);
                var output = results.First(x => x.Name == "output").AsEnumerable<float>().ToArray();
                var newState = results.First(x => x.Name == "stateN").AsTensor<float>();

                // copy back to externalState
                var flatNew = newState.ToArray();
                if (flatNew.Length == flatExternal.Length)
                {
                    Buffer.BlockCopy(flatNew, 0, externalState, 0, flatNew.Length * sizeof(float));
                }
                return output[0];
            }
            catch (Exception e)
            {
                LogUtils.Error($"Silero VAD shadow evaluation error: {e.Message}");
                return 0f;
            }
        }

        /// <summary>
        /// Applies exponential decay to the internal recurrent state (soft reset) by multiplying by factor.
        /// factor in [0,1]; lower values clear state faster.
        /// </summary>
        public void DecayState(float factor)
        {
            if (_state == null) return;
            factor = Mathf.Clamp01(factor);
            for (int a = 0; a < 2; a++)
            for (int b = 0; b < 1; b++)
            for (int c = 0; c < 128; c++)
            {
                _state[a, b, c] *= factor;
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