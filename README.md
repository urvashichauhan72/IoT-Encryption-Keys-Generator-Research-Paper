# Quantum Random Number Generation for Securing IoT Devices

**Authors:** Urvashi, Mehul, Pritam — Noida Institute of Engineering and Technology, Greater Noida

IoT devices need random numbers to make encryption keys, but many use weak generators with guessable seeds. In this project we make random bits with a quantum circuit (Qiskit), add the real readout errors of IBM's 127-qubit **ibm_brisbane** computer, clean the biased bits, and compare them with normal generators.

## What we did

1. **Quantum circuit:** 8 qubits, a Hadamard (H) gate on each, then measure. Each run gives 8 random bits.
2. **Real noise:** Qiskit Aer simulator loaded with ibm_brisbane calibration data, using each qubit's real asymmetric readout errors.
3. **Cleaning:** per-qubit Von Neumann corrector and SHA-256 extraction (512 → 256 bits).
4. **Comparison:** 7 generators, 10 million bits each: ideal QRNG, noisy QRNG (raw), noisy + Von Neumann, noisy + SHA-256, MT19937, OS CSPRNG (`os.urandom`) and a weak IoT-style LCG with a 16-bit seed.
5. **Tests:** 8 NIST SP 800-22 tests, a machine-learning next-bit predictor, a 10,000-key test and a seed-recovery attack.

## Main results

| Generator | NIST tests passed | Next-bit prediction (50% = unpredictable) | Unique keys / 10,000 |
|---|---|---|---|
| Noisy QRNG (raw) | 0 / 8 | 52.02% | 10,000 |
| Noisy + Von Neumann | 8 / 8 | 50.06% | 10,000 |
| Noisy + SHA-256 | 8 / 8 | 49.81% | 10,000 |
| OS CSPRNG | 8 / 8 | 50.36% | 10,000 |
| Weak IoT LCG | 0 / 8 | 95.03% | 16 |

The weak IoT generator's seed was recovered for 200 of 200 simulated devices.

## Files

- `qrng_experiment.py`: full experiment (about 1 minute; writes `results.json`)
- `qrng_demo.py`: quick demo of the main ideas (about 10 seconds)
- `QRNG_IoT_IEEE_Paper.pdf`: research paper (IEEE format)

## How to run

```
pip install qiskit qiskit-aer qiskit-ibm-runtime scikit-learn scipy numpy
python qrng_demo.py
python qrng_experiment.py
```

## Note

This is a simulation study using real IBM calibration data, not a run on real quantum hardware. Running on real IBM Quantum hardware is planned future work.
