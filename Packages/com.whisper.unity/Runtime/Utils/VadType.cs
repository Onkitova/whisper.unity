using System;

namespace Whisper.Utils
{
    /// <summary>
    /// Voice Activity Detection algorithm types
    /// </summary>
    public enum VadType
    {
        /// <summary>
        /// Simple energy-based VAD (existing implementation)
        /// </summary>
        Simple,
        
        /// <summary>
        /// Silero VAD using ONNX model
        /// </summary>
        Silero
    }
}