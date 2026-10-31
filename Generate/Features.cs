namespace FHoutGen.Generate;

// python/features.py and train.normalise, step for step: the voice resampled to 16 kHz (scipy.signal.resample_poly), a
// 512-sample Hann window (numpy.hanning) centred on each 1/60 s frame, the power spectrum's 64 mel bands (natural log)
// and the frame's log energy, then per cue relative to the 95th percentile energy (numpy.percentile) over 5.
static class Features
{
    public const int Rate = 16000, Fps = 60, Window = 512, NMels = 64, NIn = NMels + 1;
    const double FMin = 60.0, FMax = 7600.0;

    static readonly float[,] s_filters = MelFilters();
    static readonly float[] s_hann = Hann(Window);

    // The fhout length of a voice: a frame per 1/60 s, rounded up.
    public static int FrameCount(int samples, int rate) => (int)Math.Ceiling(samples * (double)Fps / rate);

    // (frames, NIn) row-major, normalised as the model takes it.
    public static float[] Compute(float[] pcm, int rate, int frames)
    {
        var x = rate == Rate ? pcm : ResamplePoly(pcm, rate, Rate);
        double hop = (double)Rate / Fps;
        var padded = new float[x.Length + 2 * Window];
        Array.Copy(x, 0, padded, Window, x.Length);
        var result = new float[frames * NIn];
        var re = new double[Window];
        var im = new double[Window];
        var seg = new float[Window];
        var power = new double[Window / 2 + 1];
        for (int i = 0; i < frames; i++)
        {
            long centre = (long)Math.Round((i + 0.5) * hop, MidpointRounding.ToEven);
            double energy = 0;
            for (int k = 0; k < Window; k++)
            {
                long index = Math.Clamp(centre - Window / 2 + Window + k, 0, padded.Length - 1);
                seg[k] = padded[index] * s_hann[k];
                energy += seg[k] * seg[k];
                re[k] = seg[k];
                im[k] = 0;
            }
            Fft(re, im);
            for (int k = 0; k < power.Length; k++) power[k] = re[k] * re[k] + im[k] * im[k];
            for (int m = 0; m < NMels; m++)
            {
                double sum = 0;
                for (int k = 0; k < power.Length; k++) sum += power[k] * s_filters[m, k];
                result[i * NIn + m] = (float)Math.Log(sum + 1e-10);
            }
            result[i * NIn + NMels] = (float)Math.Log(energy / Window + 1e-10);
        }
        // train.normalise
        var energies = new double[frames];
        for (int i = 0; i < frames; i++) energies[i] = result[i * NIn + NMels];
        double reference = Percentile(energies, 95);
        for (int i = 0; i < result.Length; i++) result[i] = (float)((result[i] - reference) / 5.0);
        return result;
    }

    // numpy.percentile, linear interpolation.
    static double Percentile(double[] values, double q)
    {
        var sorted = (double[])values.Clone();
        Array.Sort(sorted);
        if (sorted.Length == 0) return 0;
        double position = (sorted.Length - 1) * q / 100.0;
        int lower = (int)Math.Floor(position);
        int upper = Math.Min(lower + 1, sorted.Length - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    static float[] Hann(int m)
    {
        var w = new float[m];
        for (int n = 0; n < m; n++) w[n] = (float)(0.5 - 0.5 * Math.Cos(2 * Math.PI * n / (m - 1)));
        return w;
    }

    static float[,] MelFilters()
    {
        static double HzToMel(double f) => 2595.0 * Math.Log10(1.0 + f / 700.0);
        static double MelToHz(double m) => 700.0 * (Math.Pow(10, m / 2595.0) - 1.0);
        int bins = Window / 2 + 1;
        var freqs = new double[bins];
        for (int k = 0; k < bins; k++) freqs[k] = (double)Rate / 2 * k / (bins - 1);
        double lo0 = HzToMel(FMin), hi0 = HzToMel(FMax);
        var hz = new double[NMels + 2];
        for (int i = 0; i < hz.Length; i++) hz[i] = MelToHz(lo0 + (hi0 - lo0) * i / (NMels + 1));
        var fb = new float[NMels, bins];
        for (int m = 0; m < NMels; m++)
        {
            double lo = hz[m], mid = hz[m + 1], hi = hz[m + 2];
            for (int k = 0; k < bins; k++)
            {
                double up = (freqs[k] - lo) / (mid - lo), down = (hi - freqs[k]) / (hi - mid);
                fb[m, k] = (float)Math.Max(0, Math.Min(up, down));
            }
        }
        return fb;
    }

    // In-place radix-2 FFT.
    static void Fft(double[] re, double[] im)
    {
        int n = re.Length;
        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }
        for (int len = 2; len <= n; len <<= 1)
        {
            double angle = -2 * Math.PI / len;
            for (int i = 0; i < n; i += len)
            {
                for (int k = 0; k < len / 2; k++)
                {
                    double wr = Math.Cos(angle * k), wi = Math.Sin(angle * k);
                    int a = i + k, b = i + k + len / 2;
                    double xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                }
            }
        }
    }

    // scipy.signal.resample_poly(x, up, down) with its default window ('kaiser', 5.0).
    static float[] ResamplePoly(float[] x, int from, int to)
    {
        int g = Gcd(from, to);
        int up = to / g, down = from / g;
        int maxRate = Math.Max(up, down);
        double fc = 1.0 / maxRate;
        int halfLen = 10 * maxRate;
        var h = Firwin(2 * halfLen + 1, fc, 5.0);
        for (int i = 0; i < h.Length; i++) h[i] *= up;
        int nIn = x.Length;
        int nOut = nIn * up / down + (nIn * up % down != 0 ? 1 : 0);
        int nPrePad = down - halfLen % down;
        int nPostPad = 0;
        int nPreRemove = (halfLen + nPrePad) / down;
        while (OutputLength(h.Length + nPrePad + nPostPad, nIn, up, down) < nOut + nPreRemove) nPostPad++;
        var hp = new double[nPrePad + h.Length + nPostPad];
        Array.Copy(h, 0, hp, nPrePad, h.Length);
        // upfirdn: y[i] = sum_n x[n] * hp[i * down - n * up]
        var y = new float[nOut];
        for (int o = 0; o < nOut; o++)
        {
            long i = o + nPreRemove;
            long t = i * down;
            long nMax = Math.Min(nIn - 1, t / up);
            long nMin = Math.Max(0, (t - (hp.Length - 1) + up - 1) / up);
            double acc = 0;
            for (long n = nMin; n <= nMax; n++) acc += x[n] * hp[t - n * up];
            y[o] = (float)acc;
        }
        return y;
    }

    static int OutputLength(int lenH, int lenX, int up, int down) => ((lenX - 1) * up + lenH - 1) / down + 1;

    // scipy.signal.firwin(numtaps, cutoff, window=('kaiser', beta)), scaled to unit gain at DC.
    static double[] Firwin(int numtaps, double cutoff, double beta)
    {
        var h = new double[numtaps];
        double alpha = 0.5 * (numtaps - 1);
        double i0Beta = BesselI0(beta);
        double sum = 0;
        for (int n = 0; n < numtaps; n++)
        {
            double m = n - alpha;
            double x = cutoff * m;
            double sinc = x == 0 ? 1 : Math.Sin(Math.PI * x) / (Math.PI * x);
            double r = 2.0 * n / (numtaps - 1) - 1.0;
            double window = BesselI0(beta * Math.Sqrt(Math.Max(0, 1 - r * r))) / i0Beta;
            h[n] = cutoff * sinc * window;
            sum += h[n];
        }
        for (int n = 0; n < numtaps; n++) h[n] /= sum;
        return h;
    }

    static double BesselI0(double x)
    {
        double sum = 1, term = 1, half = x / 2;
        for (int k = 1; k < 50; k++)
        {
            term *= half / k;
            sum += term * term;
        }
        return sum;
    }

    static int Gcd(int a, int b)
    {
        while (b != 0) (a, b) = (b, a % b);
        return a;
    }
}
