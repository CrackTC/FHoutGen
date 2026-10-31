using System.Text.Json;
using AssetsTools.NET;
using AssetsTools.NET.Extra;
using DereTore.Exchange.Archive.ACB;
using DereTore.Exchange.Audio.HCA;

namespace FHoutGen.Extract;

// fhout-extract: pairs of an official voice and its fhout (the 60 Hz mouth-open series the game ships per voice cue).
//
//   stats  --streaming <dir> [--limit N]
//       counts fhout bundles, their records and frames, and how many have their voice sheet <sheet>.acb.unity3d locally.
//   export --streaming <dir> --out <dir> [--limit N] [--every N] [--rate 16000]
//       per sheet <out>/<sheet>/: <cue>.wav (mono 16-bit at --rate) and fhout.json (the bundle's text as it is).
//   --match S keeps the sheets whose name contains S, --every N [--offset K] every Nth bundle from the Kth (in name order), --limit N the first N kept.
//
// A fhout bundle fhout_<sheet>.json.unity3d holds {Header, Body: [{TextId, List: [{t, v}]}]}; each TextId is a cue of the
// voice sheet <sheet>.acb.unity3d (a raw @UTF ACB, with an optional .awb, in TextAssets).
static class Program
{
    // The game's HCA key (the demo's MltdAcbAudio.HcaKey1/2).
    const uint HcaKey1 = 0xbc731a85;
    const uint HcaKey2 = 0x0002b875;

    static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.Error.WriteLine("usage: fhout-extract stats|export --streaming <dir> [--out <dir>] [--limit N] [--rate 16000]");
            return 2;
        }
        string? streaming = null, output = null;
        int limit = 0, every = 1, offset = 0, rate = 16000;
        string? match = null;
        for (int i = 1; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--streaming" when i + 1 < args.Length: streaming = args[++i]; break;
                case "--out" when i + 1 < args.Length: output = args[++i]; break;
                case "--limit" when i + 1 < args.Length: limit = int.Parse(args[++i]); break;
                case "--match" when i + 1 < args.Length: match = args[++i]; break;
                case "--offset" when i + 1 < args.Length: offset = int.Parse(args[++i]); break;
                case "--every" when i + 1 < args.Length: every = int.Parse(args[++i]); break;
                case "--rate" when i + 1 < args.Length: rate = int.Parse(args[++i]); break;
                default: Console.Error.WriteLine("unknown argument: " + args[i]); return 2;
            }
        }
        if (streaming == null || !Directory.Exists(streaming))
        {
            Console.Error.WriteLine("--streaming <dir> is required");
            return 2;
        }
        var bundles = Directory.EnumerateFiles(streaming, "fhout_*.json.unity3d").OrderBy(p => p, StringComparer.Ordinal).ToList();
        if (match != null) bundles = bundles.Where(p => SheetOf(p).Contains(match, StringComparison.Ordinal)).ToList();
        if (every > 1) bundles = bundles.Where((_, i) => i % every == offset).ToList();
        if (limit > 0) bundles = bundles.Take(limit).ToList();
        return args[0] switch
        {
            "stats" => Stats(streaming, bundles),
            "export" when output != null => Export(streaming, bundles, output, rate),
            _ => Usage(),
        };
    }

    static int Usage()
    {
        Console.Error.WriteLine("export needs --out <dir>");
        return 2;
    }

    static string SheetOf(string bundlePath)
    {
        var name = Path.GetFileName(bundlePath);
        return name["fhout_".Length..^".json.unity3d".Length];
    }

    static int Stats(string streaming, List<string> bundles)
    {
        long records = 0, frames = 0, withVoice = 0, recordsWithVoice = 0, framesWithVoice = 0, failed = 0;
        var byKind = new SortedDictionary<string, (int bundles, long frames)>(StringComparer.Ordinal);
        foreach (var path in bundles)
        {
            FHoutTexts? texts;
            try
            {
                texts = ReadFHout(path);
            }
            catch (Exception e)
            {
                failed++;
                Console.Error.WriteLine($"{Path.GetFileName(path)}: {e.Message}");
                continue;
            }
            var sheet = SheetOf(path);
            long n = texts?.Body?.Sum(r => (long)(r.List?.Length ?? 0)) ?? 0;
            int count = texts?.Body?.Length ?? 0;
            records += count;
            frames += n;
            bool voice = File.Exists(Path.Combine(streaming, sheet + ".acb.unity3d"));
            if (voice)
            {
                withVoice++;
                recordsWithVoice += count;
                framesWithVoice += n;
            }
            var kind = string.Join('_', sheet.Split('_').Take(2));
            byKind.TryGetValue(kind, out var k);
            byKind[kind] = (k.bundles + 1, k.frames + n);
        }
        Console.WriteLine($"bundles {bundles.Count} (unreadable {failed}), records {records}, {frames / 60.0 / 3600:F1} h");
        Console.WriteLine($"with a local voice sheet: bundles {withVoice}, records {recordsWithVoice}, {framesWithVoice / 60.0 / 3600:F1} h");
        foreach (var (kind, (n, f)) in byKind.OrderByDescending(x => x.Value.frames).Take(25))
            Console.WriteLine($"  {kind,-28} bundles {n,6}  {f / 60.0 / 3600,7:F2} h");
        return 0;
    }

    static int Export(string streaming, List<string> bundles, string output, int rate)
    {
        Directory.CreateDirectory(output);
        int sheets = 0, cues = 0, missingSheet = 0, missingCue = 0, failed = 0, lengthMismatch = 0;
        var log = new object();
        // Sheets in parallel (decode and resampling are CPU-bound); counters by Interlocked.
        Parallel.ForEach(bundles, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, path =>
        {
            var sheet = SheetOf(path);
            var acbPath = Path.Combine(streaming, sheet + ".acb.unity3d");
            if (!File.Exists(acbPath))
            {
                Interlocked.Increment(ref missingSheet);
                return;
            }
            try
            {
                var (json, texts) = ReadFHoutText(path);
                if (texts?.Body == null || texts.Body.Length == 0) return;
                var dir = Path.Combine(output, sheet);
                Directory.CreateDirectory(dir);
                File.WriteAllText(Path.Combine(dir, "fhout.json"), json);
                using var acb = OpenAcb(acbPath);
                var byName = new Dictionary<string, AcbCueRecord>(StringComparer.Ordinal);
                // DereTore appends the encode type's extension to a cue name (".hca"), as the demo strips it.
                foreach (var cue in acb.Cues)
                    if (cue.CueName != null && cue.IsWaveformIdentified)
                        byName.TryAdd(Path.GetFileNameWithoutExtension(cue.CueName), cue);
                foreach (var record in texts.Body)
                {
                    if (record.TextId == null || record.List == null) continue;
                    if (!byName.TryGetValue(record.TextId, out var cue))
                    {
                        Interlocked.Increment(ref missingCue);
                        continue;
                    }
                    var (pcm, sourceRate) = DecodeCue(acb, cue);
                    // The fhout runs a frame per 1/60 s of the voice (rounded up).
                    int expected = (int)Math.Ceiling(pcm.Length * 60.0 / sourceRate);
                    if (Math.Abs(expected - record.List.Length) > 1) Interlocked.Increment(ref lengthMismatch);
                    WriteWav(Path.Combine(dir, record.TextId + ".wav"), Resample(pcm, sourceRate, rate), rate);
                    Interlocked.Increment(ref cues);
                }
                int done = Interlocked.Increment(ref sheets);
                if (done % 200 == 0)
                    lock (log) Console.WriteLine($"{done} sheets, {Volatile.Read(ref cues)} cues");
            }
            catch (Exception e)
            {
                Interlocked.Increment(ref failed);
                lock (log) Console.Error.WriteLine($"{sheet}: {e.Message}");
            }
        });
        Console.WriteLine($"done: sheets {sheets}, cues {cues}; no local sheet {missingSheet}, cue not in its sheet {missingCue}, "
            + $"failed sheets {failed}, frame count off by more than 1: {lengthMismatch}");
        return failed == 0 ? 0 : 1;
    }

    // ---- bundles ----

    static FHoutTexts? ReadFHout(string path) => ReadFHoutText(path).texts;

    static (string json, FHoutTexts? texts) ReadFHoutText(string path)
    {
        var bytes = ReadTextAssets(path).Values.First();
        var json = System.Text.Encoding.UTF8.GetString(bytes).TrimStart('﻿');
        return (json, JsonSerializer.Deserialize<FHoutTexts>(json));
    }

    // The TextAssets of a bundle (name -> m_Script bytes).
    static Dictionary<string, byte[]> ReadTextAssets(string path)
    {
        var manager = new AssetsManager();
        try
        {
            using var stream = File.OpenRead(path);
            var bundle = manager.LoadBundleFile(stream, true);
            var result = new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase);
            int count = bundle.file.BlockAndDirInfo.DirectoryInfos.Count;
            for (int i = 0; i < count; i++)
            {
                if (!bundle.file.BlockAndDirInfo.DirectoryInfos[i].IsSerialized) continue;
                var file = manager.LoadAssetsFileFromBundle(bundle, i, false);
                foreach (var info in file.file.GetAssetsOfType(AssetClassID.TextAsset))
                {
                    var field = manager.GetBaseField(file, info);
                    result[field["m_Name"].AsString] = field["m_Script"].AsByteArray;
                }
            }
            if (result.Count == 0) throw new InvalidDataException("no TextAsset");
            return result;
        }
        finally
        {
            manager.UnloadAll(true);
        }
    }

    sealed class OpenedAcb : IDisposable
    {
        public required AcbFile File;
        public required Stream Acb;
        public Stream? Awb;
        public AcbCueRecord[] Cues => File.Cues;
        public void Dispose()
        {
            File.Dispose();
            Acb.Dispose();
            Awb?.Dispose();
        }
        public static implicit operator AcbFile(OpenedAcb opened) => opened.File;
    }

    // As the demo's MltdAcbAudio: the TextAsset named *.acb (else the first), and <same name>.awb when present.
    static OpenedAcb OpenAcb(string path)
    {
        var assets = ReadTextAssets(path);
        var acbName = assets.Keys.FirstOrDefault(n => n.EndsWith(".acb", StringComparison.OrdinalIgnoreCase)) ?? assets.Keys.First();
        var acbStream = new MemoryStream(assets[acbName], false);
        MemoryStream? awbStream = assets.TryGetValue(Path.ChangeExtension(acbName, ".awb"), out var awb) ? new MemoryStream(awb, false) : null;
        var file = AcbFile.FromStream(acbStream, 0, acbStream.Length, acbName, awbStream, false);
        return new OpenedAcb { File = file, Acb = acbStream, Awb = awbStream };
    }

    // ---- audio ----

    // The cue's waveform (a voice cue names one), mixed to mono, without the HCA encoder delay and padding.
    static (float[] pcm, int rate) DecodeCue(OpenedAcb acb, AcbCueRecord cue)
    {
        using var data = acb.File.OpenDataStream(cue.CueId);
        using var hca = new MemoryStream();
        data.CopyTo(hca);
        hca.Position = 0;
        using var decoder = new OneWayHcaDecoder(hca, DecodeParams.CreateDefault(HcaKey1, HcaKey2,
            acb.File.FormatVersion >= 0x01300000 ? (cue.IsStreaming ? acb.File.ExternalAwb : acb.File.InternalAwb).HcaKeyModifier : (ushort)0));
        var info = decoder.HcaInfo;
        int channels = checked((int)info.ChannelCount), rate = checked((int)info.SamplingRate);
        long first = info.FmtR01, frames = (long)info.BlockCount * 1024 - info.FmtR01 - info.FmtR02;
        if (frames <= 0) throw new InvalidDataException(cue.CueName + ": no PCM");
        var pcm = new float[frames];
        var buffer = new byte[decoder.GetMinWaveDataBufferSize() * 32];
        long decoded = 0;
        while (true)
        {
            uint blocks = decoder.DecodeBlocks(buffer);
            if (blocks == 0) break;
            long start = decoded;
            decoded += blocks * 1024L;
            for (long f = Math.Max(start, first); f < Math.Min(decoded, first + frames); f++)
            {
                float sum = 0;
                for (int c = 0; c < channels; c++)
                {
                    int offset = (int)((f - start) * channels + c) * 2;
                    sum += unchecked((short)(buffer[offset] | buffer[offset + 1] << 8)) / 32768f;
                }
                pcm[f - first] = sum / channels;
            }
        }
        return (pcm, rate);
    }

    // Windowed-sinc resampling (Kaiser, 32 taps per side, the kernel tabulated); the training features stay below 8 kHz.
    const int KernelHalf = 32, KernelSteps = 512;
    static readonly System.Collections.Concurrent.ConcurrentDictionary<double, float[]> s_kernels = new();

    static float[] Kernel(double cutoff) => s_kernels.GetOrAdd(cutoff, c =>
    {
        const double beta = 8.6;
        double i0Beta = BesselI0(beta);
        var table = new float[KernelHalf * KernelSteps + 2];
        for (int k = 0; k < table.Length; k++)
        {
            double x = (double)k / KernelSteps;
            double r = x / KernelHalf;
            if (r >= 1) continue;
            double sinc = x == 0 ? 1 : Math.Sin(Math.PI * x * c) / (Math.PI * x * c);
            table[k] = (float)(c * sinc * BesselI0(beta * Math.Sqrt(1 - r * r)) / i0Beta);
        }
        return table;
    });

    static float[] Resample(float[] input, int from, int to)
    {
        if (from == to) return input;
        double ratio = (double)to / from;
        int n = (int)Math.Ceiling(input.Length * ratio);
        var output = new float[n];
        var kernel = Kernel(Math.Min(1.0, ratio) * 0.95);
        for (int i = 0; i < n; i++)
        {
            double center = i / ratio;
            int lo = (int)Math.Floor(center) - KernelHalf + 1, hi = (int)Math.Floor(center) + KernelHalf;
            double acc = 0;
            for (int j = Math.Max(lo, 0); j <= Math.Min(hi, input.Length - 1); j++)
            {
                double pos = Math.Abs(j - center) * KernelSteps;
                int k = (int)pos;
                if (k >= kernel.Length - 1) continue;
                double frac = pos - k;
                acc += input[j] * (kernel[k] + (kernel[k + 1] - kernel[k]) * frac);
            }
            output[i] = (float)acc;
        }
        return output;
    }

    static double BesselI0(double x)
    {
        double sum = 1, term = 1, half = x / 2;
        for (int k = 1; k < 32; k++)
        {
            term *= half / k;
            sum += term * term;
        }
        return sum;
    }

    static void WriteWav(string path, float[] samples, int rate)
    {
        using var writer = new BinaryWriter(File.Create(path));
        int bytes = samples.Length * 2;
        writer.Write("RIFF"u8);
        writer.Write(36 + bytes);
        writer.Write("WAVEfmt "u8);
        writer.Write(16);
        writer.Write((short)1);
        writer.Write((short)1);
        writer.Write(rate);
        writer.Write(rate * 2);
        writer.Write((short)2);
        writer.Write((short)16);
        writer.Write("data"u8);
        writer.Write(bytes);
        foreach (var s in samples)
            writer.Write((short)Math.Clamp(Math.Round(s * 32767.0), short.MinValue, short.MaxValue));
    }
}

sealed class FHoutTexts
{
    public string? Header { get; set; }
    public FHoutText[]? Body { get; set; }
}

sealed class FHoutText
{
    public string? TextId { get; set; }
    public FHoutPoint[]? List { get; set; }
}

sealed class FHoutPoint
{
    public int t { get; set; }
    public float v { get; set; }
}
