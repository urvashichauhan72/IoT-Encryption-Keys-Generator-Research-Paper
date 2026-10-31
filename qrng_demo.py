"""
QRNG for IoT - quick live demo (runs in under a minute)
Install once:  pip install qiskit qiskit-aer qiskit-ibm-runtime
Run:           python qrng_demo.py
"""
import hashlib, time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

SHOTS = 60000
line = "-" * 60

def get_bits(sim, shots):
    qc = QuantumCircuit(8)
    qc.h(range(8))            # step 1: put all 8 qubits in a 50/50 spin (H gate)
    qc.measure_all()          # step 2: measure them -> 8 random bits
    tq = transpile(qc, sim, initial_layout=list(range(8)))
    mem = sim.run(tq, shots=shots, memory=True).result().get_memory()
    return np.array([[int(c) for c in m[::-1]] for m in mem])  # columns = q0..q7

def von_neumann(bits):
    pairs = bits[: len(bits) // 2 * 2].reshape(-1, 2)
    keep = pairs[:, 0] != pairs[:, 1]      # 01 -> 0, 10 -> 1, drop 00 and 11
    return pairs[keep, 0]

print(line)
print("1) The quantum circuit: 8 qubits, H gate on each, then measure")
print(line)
qc = QuantumCircuit(8); qc.h(range(8)); qc.measure_all()
print(qc.draw(output="text"))

print(line)
print("2) Perfect quantum computer (no mistakes)")
print(line)
ideal = get_bits(AerSimulator(), SHOTS)
for q in range(8):
    print(f"   qubit {q}: gave 1 in {ideal[:, q].mean()*100:5.1f}% of runs")

print(line)
print("3) With the real mistakes of IBM's ibm_brisbane computer")
print(line)
fb = FakeBrisbane(); props = fb.properties()
nm = NoiseModel.from_backend(fb, readout_error=False)
for q in range(8):
    p10 = props.qubit_property(q)["prob_meas1_prep0"][0]
    p01 = props.qubit_property(q)["prob_meas0_prep1"][0]
    nm.add_readout_error(ReadoutError([[1 - p10, p10], [p01, 1 - p01]]), [q])
noisy = get_bits(AerSimulator.from_backend(fb, noise_model=nm), SHOTS)
for q in range(8):
    flag = "   <-- unfair coin!" if abs(noisy[:, q].mean() - 0.5) > 0.05 else ""
    print(f"   qubit {q}: gave 1 in {noisy[:, q].mean()*100:5.1f}% of runs{flag}")

print(line)
print("4) After cleaning with the Von Neumann pair trick")
print(line)
for q in range(8):
    clean = von_neumann(noisy[:, q])
    print(f"   qubit {q}: gave 1 in {clean.mean()*100:5.1f}% of runs")

print(line)
print("5) A 128-bit quantum key for an IoT device (SHA-256 cleaned)")
print(line)
raw = noisy.reshape(-1)[:512]
key = hashlib.sha256(np.packbits(raw).tobytes()).hexdigest()[:32]
print("   key =", key)

print(line)
print("6) Hacking a weak IoT generator (16-bit seed)")
print(line)
def lcg(seed, n):
    out = []
    for _ in range(n):
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
        out.append(seed & 0xFF)
    return bytes(out)
secret_seed = int(np.random.default_rng().integers(0, 65536))
device_key1 = lcg(secret_seed, 32)[:16]
device_key2 = lcg(secret_seed, 32)[16:]
print("   device key 1 (hacker can see):", device_key1.hex())
t = time.perf_counter()
for s in range(65536):                       # try every possible seed
    if lcg(s, 16) == device_key1:
        guess = lcg(s, 32)[16:]
        break
ms = (time.perf_counter() - t) * 1000
print("   hacker's guess of key 2:      ", guess.hex())
print("   real key 2:                   ", device_key2.hex())
print(f"   hacked: {guess == device_key2}  (in {ms:.1f} ms)")
print("   A quantum key has no seed, so this attack is impossible.")
print(line)
