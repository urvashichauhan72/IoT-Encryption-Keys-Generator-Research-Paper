"""QRNG for IoT key generation -- full experiment.
Sources: Ideal QRNG (Aer), Noisy QRNG raw (FakeBrisbane calibration), Noisy+VonNeumann,
Noisy+SHA-256, MT19937 (Python random), OS CSPRNG (os.urandom), Weak IoT LCG (low byte).
"""
import os, time, json, hashlib, random
import numpy as np
from scipy.special import erfc, gammaincc
from scipy.stats import norm, chisquare
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

NQ = 8
SEQ_LEN = 100_000
N_SEQ = 100
TARGET = SEQ_LEN * N_SEQ          # 10 M bits per source
OUT = {}
rng_np = np.random.default_rng()

# ---------------- generators ----------------
def qrng_shots(sim, shots):
    qc = QuantumCircuit(NQ); qc.h(range(NQ)); qc.measure_all()
    tq = transpile(qc, sim, initial_layout=list(range(NQ)), optimization_level=1)
    mem = []
    left = shots
    while left > 0:
        s = min(left, 500_000)
        mem += sim.run(tq, shots=s, memory=True).result().get_memory()
        left -= s
    # memory string: qubit 0 is rightmost char
    arr = np.frombuffer("".join(mem).encode(), dtype=np.uint8).reshape(-1, NQ) - 48
    return arr[:, ::-1].copy()          # columns = qubit 0..7

def von_neumann(b):
    b = b[: len(b) // 2 * 2].reshape(-1, 2)
    keep = b[:, 0] != b[:, 1]
    return b[keep, 0]

def per_qubit_vn(mat):
    streams = [von_neumann(mat[:, q]) for q in range(mat.shape[1])]
    L = min(len(s) for s in streams)
    return np.stack([s[:L] for s in streams], axis=1).reshape(-1)

def sha256_extract(bits, in_block=512):
    n = len(bits) // in_block
    out = []
    by = np.packbits(bits[: n * in_block]).reshape(n, in_block // 8)
    for row in by:
        out.append(np.frombuffer(hashlib.sha256(row.tobytes()).digest(), dtype=np.uint8))
    return np.unpackbits(np.concatenate(out))

def mt_bits(n):
    r = random.Random()
    return np.unpackbits(np.frombuffer(r.randbytes(n // 8), dtype=np.uint8))

def os_bits(n):
    return np.unpackbits(np.frombuffer(os.urandom(n // 8), dtype=np.uint8))

def lcg_bytes(seed, nbytes):
    s = seed; out = np.empty(nbytes, dtype=np.uint8)
    for i in range(nbytes):
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        out[i] = s & 0xFF
    return out

def lcg_bits(n, seed):
    # vectorised: LCG low 8 bits only depend on low 8 bits of state (m=2^31 power of two)
    # but we compute the full state honestly in chunks
    nbytes = n // 8
    out = np.empty(nbytes, dtype=np.uint8); s = seed
    a, c, m = 1103515245, 12345, 0x7FFFFFFF
    for i in range(nbytes):
        s = (a * s + c) & m
        out[i] = s & 0xFF
    return np.unpackbits(out)

# ---------------- NIST SP 800-22 tests ----------------
def windows(x, m):
    if m == 0: return None
    xe = np.concatenate([x, x[: m - 1]])
    v = np.zeros(len(x), dtype=np.int64)
    for j in range(m):
        v = (v << 1) | xe[j : j + len(x)]
    return v

def t_monobit(x):
    n = len(x); S = np.sum(2 * x.astype(int) - 1)
    return erfc(abs(S) / np.sqrt(n) / np.sqrt(2))

def t_blockfreq(x, M=128):
    N = len(x) // M; pi = x[: N * M].reshape(N, M).mean(1)
    chi = 4 * M * np.sum((pi - 0.5) ** 2)
    return gammaincc(N / 2, chi / 2)

def t_runs(x):
    n = len(x); pi = x.mean()
    if abs(pi - 0.5) >= 2 / np.sqrt(n): return 0.0
    V = 1 + np.count_nonzero(x[1:] != x[:-1])
    return erfc(abs(V - 2 * n * pi * (1 - pi)) / (2 * np.sqrt(2 * n) * pi * (1 - pi)))

def t_longest(x, M=128):
    N = len(x) // M; B = x[: N * M].reshape(N, M)
    longest = np.zeros(N, dtype=int)
    for i, row in enumerate(B):
        d = np.diff(np.concatenate([[0], row, [0]]))
        st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
        longest[i] = (en - st).max() if len(st) else 0
    v = np.array([np.sum(longest <= 4), np.sum(longest == 5), np.sum(longest == 6),
                  np.sum(longest == 7), np.sum(longest == 8), np.sum(longest >= 9)])
    pi = np.array([0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124])
    chi = np.sum((v - N * pi) ** 2 / (N * pi))
    return gammaincc(5 / 2, chi / 2)

def t_cusum(x):
    n = len(x); S = np.cumsum(2 * x.astype(int) - 1); z = np.max(np.abs(S))
    if z == 0: return 0.0
    sq = np.sqrt(n)
    k1 = np.arange(int(np.floor((-n / z + 1) / 4)), int(np.floor((n / z - 1) / 4)) + 1)
    k2 = np.arange(int(np.floor((-n / z - 3) / 4)), int(np.floor((n / z - 1) / 4)) + 1)
    s1 = np.sum(norm.cdf((4 * k1 + 1) * z / sq) - norm.cdf((4 * k1 - 1) * z / sq))
    s2 = np.sum(norm.cdf((4 * k2 + 3) * z / sq) - norm.cdf((4 * k2 + 1) * z / sq))
    return float(min(1.0, max(0.0, 1 - s1 + s2)))

def _phi(x, m):
    n = len(x); c = np.bincount(windows(x, m), minlength=2 ** m) / n
    c = c[c > 0]; return np.sum(c * np.log(c))

def t_apen(x, m=8):
    n = len(x); ap = _phi(x, m) - _phi(x, m + 1)
    chi = 2 * n * (np.log(2) - ap)
    return gammaincc(2 ** (m - 1), chi / 2)

def _psi(x, m):
    if m <= 0: return 0.0
    n = len(x); c = np.bincount(windows(x, m), minlength=2 ** m)
    return (2 ** m / n) * np.sum(c.astype(float) ** 2) - n

def t_serial(x, m=8):
    p0, p1, p2 = _psi(x, m), _psi(x, m - 1), _psi(x, m - 2)
    d1 = p0 - p1; d2 = p0 - 2 * p1 + p2
    return min(gammaincc(2 ** (m - 2), d1 / 2), gammaincc(2 ** (m - 3), d2 / 2))

def t_dft(x):
    n = len(x); X = 2 * x.astype(float) - 1
    M = np.abs(np.fft.fft(X))[: n // 2]
    T = np.sqrt(np.log(1 / 0.05) * n); N0 = 0.95 * n / 2; N1 = np.sum(M < T)
    d = (N1 - N0) / np.sqrt(n * 0.95 * 0.05 / 4)
    return erfc(abs(d) / np.sqrt(2))

TESTS = [("Frequency (Monobit)", t_monobit), ("Block Frequency", t_blockfreq),
         ("Runs", t_runs), ("Longest Run of Ones", t_longest), ("Cumulative Sums", t_cusum),
         ("Approximate Entropy", t_apen), ("Serial", t_serial), ("DFT (Spectral)", t_dft)]

def nist_suite(bits, seq_len=SEQ_LEN):
    nseq = len(bits) // seq_len
    res = {}
    for name, f in TESTS:
        ps = np.array([f(bits[i * seq_len : (i + 1) * seq_len]) for i in range(nseq)])
        passed = int(np.sum(ps >= 0.01))
        hist = np.histogram(ps, bins=10, range=(0, 1))[0]
        unif = float(gammaincc(9 / 2, chisquare(hist).statistic / 2)) if nseq >= 55 else None
        res[name] = dict(passed=passed, total=nseq, prop=passed / nseq, uniformity_p=unif,
                         mean_p=float(ps.mean()))
    phat = 0.99
    res["_min_prop"] = phat - 3 * np.sqrt(phat * (1 - phat) / nseq)
    return res

# ---------------- statistics ----------------
def stats(bits):
    p1 = float(bits.mean()); p0 = 1 - p1
    H = -(p1 * np.log2(p1) + p0 * np.log2(p0))
    Hmin = -np.log2(max(p0, p1))
    by = np.packbits(bits[: len(bits) // 8 * 8])
    cnt = np.bincount(by, minlength=256)
    Hmin_byte = -np.log2(cnt.max() / cnt.sum()) / 8
    chi = chisquare(cnt)
    x = 2 * bits[:1_000_000].astype(float) - 1
    ac1 = float(np.mean(x[1:] * x[:-1]))
    return dict(n_bits=int(len(bits)), p1=p1, bias=abs(p1 - 0.5), shannon=float(H),
                min_entropy_bit=float(Hmin), min_entropy_byte=float(Hmin_byte),
                chi2_byte=float(chi.statistic), chi2_byte_p=float(chi.pvalue), autocorr_lag1=ac1)

# ---------------- next-bit predictability (ML) ----------------
from sklearn.ensemble import HistGradientBoostingClassifier
def nextbit_accuracy(bits, k=16, n=200_000):
    b = bits[: n + k]
    X = np.lib.stride_tricks.sliding_window_view(b[:-1], k)[:n]
    y = b[k : k + n]
    split = int(0.7 * n)
    clf = HistGradientBoostingClassifier(max_iter=100).fit(X[:split], y[:split])
    acc = float((clf.predict(X[split:]) == y[split:]).mean())
    ntest = n - split
    ci = 1.96 * np.sqrt(0.25 / ntest)
    return dict(acc=acc, ci95=float(ci), ntest=ntest)

# ---------------- IoT key experiment ----------------
def key_metrics(bits, nkeys=10_000):
    K = bits[: nkeys * 128].reshape(nkeys, 128)
    uniq = len({row.tobytes() for row in np.packbits(K, axis=1)})
    idx = rng_np.integers(0, nkeys, size=(20_000, 2)); idx = idx[idx[:, 0] != idx[:, 1]]
    hd = np.sum(K[idx[:, 0]] != K[idx[:, 1]], axis=1)
    pos_bias = np.abs(K.mean(0) - 0.5).max()
    return dict(unique=uniq, nkeys=nkeys, hd_mean=float(hd.mean()), hd_std=float(hd.std()),
                max_pos_bias=float(pos_bias))

def seed_attack(n_devices=200, seed_bits=16):
    """Each device seeds the LCG with a 16-bit value (e.g. boot-time ms) and makes
    a 128-bit key followed by a second key. Attacker sees key1, brute-forces seed, predicts key2."""
    a, c, m = 1103515245, 12345, 0x7FFFFFFF
    # precompute: first 16 bytes from every seed (vectorised over all seeds)
    seeds = np.arange(2 ** seed_bits, dtype=np.int64)
    s = seeds.copy(); first = np.empty((len(seeds), 16), dtype=np.uint8)
    for i in range(16):
        s = (a * s + c) & m; first[:, i] = s & 0xFF
    succ = 0; times = []
    for _ in range(n_devices):
        sd = int(rng_np.integers(0, 2 ** seed_bits))
        out = lcg_bytes(sd, 32); k1, k2 = out[:16], out[16:]
        t = time.perf_counter()
        cand = np.where((first == k1).all(1))[0]
        hit = any(np.array_equal(lcg_bytes(int(cs), 32)[16:], k2) for cs in cand)
        times.append(time.perf_counter() - t); succ += hit
    return dict(devices=n_devices, recovered=succ, rate=succ / n_devices,
                mean_ms=1000 * float(np.mean(times)), seed_space=2 ** seed_bits)

# ---------------- run ----------------
if __name__ == "__main__":
    from qiskit_aer.noise import NoiseModel, ReadoutError
    fb = FakeBrisbane(); props = fb.properties()
    nm = NoiseModel.from_backend(fb, readout_error=False)   # calibrated gate + thermal-relaxation errors
    for q in range(NQ):   # calibrated ASYMMETRIC assignment errors (Aer's default is symmetric)
        p10 = props.qubit_property(q)["prob_meas1_prep0"][0]; p01 = props.qubit_property(q)["prob_meas0_prep1"][0]
        nm.add_readout_error(ReadoutError([[1 - p10, p10], [p01, 1 - p01]]), [q])
    ideal = AerSimulator(); noisy = AerSimulator.from_backend(fb, noise_model=nm)
    OUT["backend_readout"] = [dict(q=q, p10=props.qubit_property(q)["prob_meas1_prep0"][0],
                                   p01=props.qubit_property(q)["prob_meas0_prep1"][0])
                              for q in range(NQ)]
    src = {}; thr = {}
    t = time.perf_counter(); m = qrng_shots(ideal, TARGET // NQ); thr["Ideal QRNG"] = TARGET / (time.perf_counter() - t)
    src["Ideal QRNG"] = m.reshape(-1)
    t = time.perf_counter(); mn = qrng_shots(noisy, 5 * TARGET // NQ + 100_000); dt = time.perf_counter() - t
    raw = mn.reshape(-1)
    thr["Noisy QRNG (raw)"] = len(raw) / dt
    OUT["per_qubit_p1_noisy"] = mn.mean(0).tolist()
    OUT["per_qubit_p1_ideal"] = m.mean(0).tolist()
    OUT["per_qubit_monobit_raw"] = []
    for q in range(NQ):
        col = mn[:, q]; ns = len(col) // SEQ_LEN
        ps = [t_monobit(col[i*SEQ_LEN:(i+1)*SEQ_LEN]) for i in range(ns)]
        vq = von_neumann(col); nv = len(vq) // SEQ_LEN
        pv = [t_monobit(vq[i*SEQ_LEN:(i+1)*SEQ_LEN]) for i in range(nv)]
        OUT["per_qubit_monobit_raw"].append(dict(q=q, p1=float(col.mean()), passed=int(sum(p>=0.01 for p in ps)), total=ns,
            hmin=float(-np.log2(max(col.mean(), 1-col.mean()))), vn_p1=float(vq.mean()), vn_passed=int(sum(p>=0.01 for p in pv)), vn_total=nv))
    t = time.perf_counter(); vn = per_qubit_vn(mn); thr["Noisy QRNG + Von Neumann"] = len(vn) / (dt + time.perf_counter() - t)
    OUT["vn_yield"] = len(vn) / len(raw)
    t = time.perf_counter(); sh = sha256_extract(raw); thr["Noisy QRNG + SHA-256"] = len(sh) / (dt + time.perf_counter() - t)
    src["Noisy QRNG (raw)"] = raw[:TARGET]
    src["Noisy QRNG + Von Neumann"] = vn[:TARGET]
    src["Noisy QRNG + SHA-256"] = sh[:TARGET]
    t = time.perf_counter(); src["MT19937 PRNG"] = mt_bits(TARGET); thr["MT19937 PRNG"] = TARGET / (time.perf_counter() - t)
    t = time.perf_counter(); src["OS CSPRNG (urandom)"] = os_bits(TARGET); thr["OS CSPRNG (urandom)"] = TARGET / (time.perf_counter() - t)
    t = time.perf_counter(); src["Weak IoT LCG"] = lcg_bits(TARGET, 12345); thr["Weak IoT LCG"] = TARGET / (time.perf_counter() - t)

    OUT["sources"] = {}
    for name, b in src.items():
        print("==", name, len(b), flush=True)
        d = dict(stats=stats(b), nist=nist_suite(b), nextbit=nextbit_accuracy(b),
                 keys=key_metrics(b), throughput_bps=thr[name])
        OUT["sources"][name] = d
        print(json.dumps({k: d["nist"][k]["passed"] for k in d["nist"] if not k.startswith("_")}),
              d["stats"]["bias"], d["nextbit"]["acc"], flush=True)
    OUT["seed_attack"] = seed_attack()
    print(OUT["seed_attack"])
    json.dump(OUT, open("results.json", "w"), indent=1, default=float)
    np.save("ideal_sample.npy", src["Ideal QRNG"][:200_000])
    np.save("noisy_perq.npy", mn[:200_000])
