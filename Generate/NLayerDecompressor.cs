using NAudio.Wave;
using NLayer;

namespace FHoutGen.Generate;

// NAudio's IMp3FrameDecompressor on NLayer's managed decoder (NAudio's own ACM one marshals structs, which the native AOT
// build cannot). 16-bit PCM out, as Mp3FileReader's ACM path gives.
sealed class NLayerDecompressor : IMp3FrameDecompressor
{
    readonly MpegFrameDecoder decoder = new();
    readonly FrameAdapter frame = new();
    readonly float[] samples = new float[1152 * 2];

    public NLayerDecompressor(WaveFormat source)
    {
        OutputFormat = new WaveFormat(source.SampleRate, 16, source.Channels);
    }

    public WaveFormat OutputFormat { get; }

    public int DecompressFrame(Mp3Frame source, byte[] dest, int destOffset)
    {
        frame.Load(source);
        int count = decoder.DecodeFrame(frame, samples, 0);
        for (int i = 0; i < count; i++)
        {
            short s = (short)Math.Clamp(Math.Round(samples[i] * 32767.0), short.MinValue, short.MaxValue);
            dest[destOffset + i * 2] = (byte)s;
            dest[destOffset + i * 2 + 1] = (byte)(s >> 8);
        }
        return count * 2;
    }

    public void Reset() => decoder.Reset();

    public void Dispose()
    {
    }

    // An NAudio frame as NLayer reads one.
    sealed class FrameAdapter : IMpegFrame
    {
        Mp3Frame? frame;
        int bitPosition;

        public void Load(Mp3Frame source)
        {
            frame = source;
            bitPosition = 0;
        }

        Mp3Frame F => frame ?? throw new InvalidOperationException();
        public int SampleRate => F.SampleRate;
        public int SampleRateIndex => (F.RawData[2] >> 2) & 3;
        public int FrameLength => F.FrameLength;
        public int BitRate => F.BitRate;
        public NLayer.MpegVersion Version => F.MpegVersion switch
        {
            NAudio.Wave.MpegVersion.Version1 => NLayer.MpegVersion.Version1,
            NAudio.Wave.MpegVersion.Version2 => NLayer.MpegVersion.Version2,
            _ => NLayer.MpegVersion.Version25,
        };
        public NLayer.MpegLayer Layer => F.MpegLayer switch
        {
            NAudio.Wave.MpegLayer.Layer1 => NLayer.MpegLayer.LayerI,
            NAudio.Wave.MpegLayer.Layer2 => NLayer.MpegLayer.LayerII,
            _ => NLayer.MpegLayer.LayerIII,
        };
        public MpegChannelMode ChannelMode => (MpegChannelMode)F.ChannelMode;
        public int ChannelModeExtension => F.ChannelExtension;
        public int SampleCount => F.SampleCount;
        public int BitRateIndex => F.BitRateIndex;
        public bool IsCopyrighted => F.Copyright;
        public bool HasCrc => F.CrcPresent;
        public bool IsCorrupted => false;

        public void Reset() => bitPosition = 0;

        public int ReadBits(int bitCount)
        {
            if (bitCount < 1 || bitCount > 32) throw new ArgumentOutOfRangeException(nameof(bitCount));
            int start = 4 + (HasCrc ? 2 : 0);
            var raw = F.RawData;
            int value = 0;
            for (int i = 0; i < bitCount; i++)
            {
                int byteIndex = start + (bitPosition >> 3);
                if (byteIndex >= raw.Length) return -1;
                int bit = (raw[byteIndex] >> (7 - (bitPosition & 7))) & 1;
                value = (value << 1) | bit;
                bitPosition++;
            }
            return value;
        }
    }
}
