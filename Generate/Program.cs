using System.Globalization;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Wave;

namespace FHoutGen.Generate;

// fhout-gen: the game's fhout JSON for voices (python/generate.py in .NET).
//
//   fhout-gen [-o out.json] [--model model.onnx] <voice or folder> ...
//
// Each voice (wav, mp3 or anything Windows Media Foundation reads; any rate, mono or stereo) becomes one record of
// {"Header", "Body": [{"TextId": <file name without extension>, "List": [{"t", "v"}]}]}, a value per 1/60 s rounded to 3
// decimals. A folder takes its .wav and .mp3 files in name order. The model is the embedded one unless --model is given.
// Without -o the JSON goes to standard output.
static class Program
{
    static readonly string[] s_extensions = [".wav", ".mp3"];

    static int Main(string[] args)
    {
        string? output = null, modelPath = null;
        var inputs = new List<string>();
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "-o" or "--out" when i + 1 < args.Length: output = args[++i]; break;
                case "--model" when i + 1 < args.Length: modelPath = args[++i]; break;
                case "-h" or "--help": return Usage();
                default: inputs.Add(args[i]); break;
            }
        }
        var files = new List<string>();
        foreach (var input in inputs)
        {
            if (Directory.Exists(input))
                files.AddRange(Directory.EnumerateFiles(input)
                    .Where(f => s_extensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
                    .OrderBy(f => f, StringComparer.Ordinal));
            else if (File.Exists(input))
                files.Add(input);
            else
            {
                Console.Error.WriteLine("not found: " + input);
                return 1;
            }
        }
        if (files.Count == 0) return Usage();

        using var session = new InferenceSession(modelPath != null ? File.ReadAllBytes(modelPath) : EmbeddedModel());
        var json = new StringBuilder();
        json.Append("{\"Header\":\"").Append(DateTime.Now.ToString("yyyy/MM/dd HH:mm:ss", CultureInfo.InvariantCulture))
            .Append("\",\"Body\":[");
        for (int f = 0; f < files.Count; f++)
        {
            var (pcm, rate) = ReadMono(files[f]);
            var values = Predict(session, pcm, rate);
            if (f > 0) json.Append(',');
            json.Append("{\"TextId\":").Append(JsonString(Path.GetFileNameWithoutExtension(files[f]))).Append(",\"List\":[");
            for (int t = 0; t < values.Length; t++)
            {
                if (t > 0) json.Append(',');
                json.Append("{\"t\":").Append(t).Append(",\"v\":")
                    .Append(Math.Round(values[t], 3).ToString("0.0##", CultureInfo.InvariantCulture)).Append('}');
            }
            json.Append("]}");
            Console.Error.WriteLine($"{Path.GetFileName(files[f])}: {values.Length} frames");
        }
        json.Append("]}");
        if (output != null) File.WriteAllText(output, json.ToString(), new UTF8Encoding(false));
        else Console.Out.Write(json.ToString());
        return 0;
    }

    static int Usage()
    {
        Console.Error.WriteLine("usage: fhout-gen [-o out.json] [--model model.onnx] <voice or folder> ...");
        return 2;
    }

    static byte[] EmbeddedModel()
    {
        using var stream = typeof(Program).Assembly.GetManifestResourceStream("model.onnx")
            ?? throw new InvalidOperationException("the embedded model is missing");
        using var copy = new MemoryStream();
        stream.CopyTo(copy);
        return copy.ToArray();
    }

    // The voice as mono float samples at its own rate (channels averaged). An mp3 goes through ACM (Mp3FileReader), which
    // decodes it sample for sample as ffmpeg does; Media Foundation (AudioFileReader's mp3 path) drops the first 528
    // samples, a frame of lip data early.
    static (float[] pcm, int rate) ReadMono(string path)
    {
        using WaveStream stream = Path.GetExtension(path).Equals(".mp3", StringComparison.OrdinalIgnoreCase)
            ? new Mp3FileReader(path)
            : new AudioFileReader(path);
        var reader = stream.ToSampleProvider();
        int channels = reader.WaveFormat.Channels;
        var samples = new List<float>();
        var buffer = new float[reader.WaveFormat.SampleRate * channels];
        int read;
        while ((read = reader.Read(buffer, 0, buffer.Length)) > 0)
            for (int i = 0; i + channels <= read; i += channels)
            {
                float sum = 0;
                for (int c = 0; c < channels; c++) sum += buffer[i + c];
                samples.Add(sum / channels);
            }
        return (samples.ToArray(), reader.WaveFormat.SampleRate);
    }

    // train.predict: the value where the open logit is above 0, else 0.
    static float[] Predict(InferenceSession session, float[] pcm, int rate)
    {
        int frames = Features.FrameCount(pcm.Length, rate);
        var values = new float[frames];
        if (frames == 0) return values;
        var input = new DenseTensor<float>(Features.Compute(pcm, rate, frames), [1, frames, Features.NIn]);
        using var results = session.Run([NamedOnnxValue.CreateFromTensor("features", input)]);
        var logit = results.First(r => r.Name == "open_logit").AsTensor<float>();
        var value = results.First(r => r.Name == "value").AsTensor<float>();
        for (int t = 0; t < frames; t++) values[t] = logit[0, t] > 0 ? value[0, t] : 0f;
        return values;
    }

    static string JsonString(string s)
    {
        var sb = new StringBuilder("\"");
        foreach (char c in s)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                default:
                    if (c < 0x20) sb.Append($"\\u{(int)c:x4}");
                    else sb.Append(c);
                    break;
            }
        }
        return sb.Append('"').ToString();
    }
}
