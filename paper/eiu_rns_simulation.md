# EIU Residue Number System: Simulation and CRT Solver

> Recorded verbatim as received (2026-08-12). The production implementation
> (with the corrections noted in [REVIEW_haloalg.md](REVIEW_haloalg.md)) lives
> in `src/socrates/eiu/rns.py`.

In the HaloAlg Framework the Exact Inference Unit (EIU) implements a Residue
Number System (RNS). Unlike positional number systems (like binary or decimal)
where carry-propagation limits calculation speed, RNS performs addition,
subtraction, and multiplication in independent, parallel channels with no
carries. This achieves constant clock-cycle (Iso-Latency) multiplication,
which is a crucial hardware requirement for supporting unscaled dot products
in Halo's rational arithmetic.

## 1. Python Simulation of EIU's RNS Parallel Multiplication

```python
def extended_gcd(a, b):
    """Extended Euclidean Algorithm to find GCD and Bezout coefficients."""
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

def mod_inverse(a, m):
    """Computes the modular multiplicative inverse of 'a' modulo 'm'."""
    gcd, x, _ = extended_gcd(a, m)
    if gcd != 1:
        raise ValueError(f"Modular inverse does not exist for {a} mod {m}")
    return x % m

class ExactInferenceUnitRNS:
    def __init__(self, moduli):
        """
        Initializes the RNS with a set of pairwise coprime moduli.
        In hardware, these are chosen as large prime numbers to maximize dynamic range.
        """
        # Verification of coprime moduli
        for i in range(len(moduli)):
            for j in range(i + 1, len(moduli)):
                gcd, _, _ = extended_gcd(moduli[i], moduli[j])
                if gcd != 1:
                    raise ValueError(f"Moduli {moduli[i]} and {moduli[j]} are not coprime!")

        self.moduli = moduli
        self.M = 1
        for m in moduli:
            self.M *= m

        # Precompute CRT constants for hardware optimization
        self.M_i = [self.M // m for m in moduli]
        self.inv_M_i = [mod_inverse(self.M_i[k], self.moduli[k]) for k in range(len(moduli))]

    def encode(self, x):
        """Encodes a positional integer 'x' into parallel RNS channels (residues)."""
        if not (0 <= x < self.M):
            raise ValueError(f"Value {x} is out of the dynamic range bounds [0, {self.M - 1}]")
        return [x % m for m in self.moduli]

    def decode(self, residues):
        """Reconstructs the original integer from residues using the Chinese Remainder Theorem."""
        x = 0
        for k in range(len(self.moduli)):
            # Equation: x = Sum(residues_k * M_k * (M_k^-1 mod m_k)) mod M
            term = (residues[k] * self.M_i[k] * self.inv_M_i[k]) % self.M
            x = (x + term) % self.M
        return x

    def multiply(self, x_residues, y_residues):
        """
        Performs parallel, carry-free modular multiplication.
        In the EIU hardware, each of these channels computes in parallel over an independent ALU.
        """
        return [(x_r * y_r) % m for x_r, y_r, m in zip(x_residues, y_residues, self.moduli)]

# Execute and Verify
if __name__ == "__main__":
    # Define a sample 4-channel coprime moduli set
    moduli_set = [97, 101, 103, 107]
    eiu_rns = ExactInferenceUnitRNS(moduli_set)

    # Inputs to multiply
    X = 123456
    Y = 789

    # RNS Encoding
    x_res = eiu_rns.encode(X)
    y_res = eiu_rns.encode(Y)

    print(f"Moduli Set: {moduli_set}")
    print(f"EIU Dynamic Range (M): {eiu_rns.M}")
    print(f"X = {X} -> RNS Encoded: {x_res}")
    print(f"Y = {Y} -> RNS Encoded: {y_res}\n")

    # EIU Parallel Channel Multiplication
    z_res = eiu_rns.multiply(x_res, y_res)
    print(f"Z Residues (Parallel Multiplication Output): {z_res}")

    # CRT Decoding & Verification
    Z = eiu_rns.decode(z_res)
    expected_Z = (X * Y) % eiu_rns.M

    print(f"Decoded Z: {Z}")
    print(f"Expected Classical Z: {expected_Z}")
    print(f"Verification Result: {'SUCCESS' if Z == expected_Z else 'FAILED'}")
```

### Simulation Verification Output

```
Moduli Set: [97, 101, 103, 107]
EIU Dynamic Range (M): 107972737
X = 123456 -> RNS Encoded: [72, 34, 62, 85]
Y = 789 -> RNS Encoded: [13, 82, 68, 40]

Z Residues (Parallel Multiplication Output): [63, 61, 96, 83]
Decoded Z: 97406784
Expected Classical Z: 97406784
Verification Result: SUCCESS
```

## 2. The EIU RNS Solver Algorithm (CRT Reconstruction)

```
Algorithm: EIU_RNS_CRT_Decoder (Modular Reconstruction)
========================================================================
Inputs:
  - residues   : Array of size N, [r_1, r_2, ..., r_N] representing residues
  - moduli     : Array of size N, [m_1, m_2, ..., m_N] of pairwise coprime moduli
  - M_i        : Precomputed array of size N, where M_i = M / m_i (M is the total product)
  - inv_M_i    : Precomputed array of size N, where inv_M_i = mod_inverse(M_i, m_i)
  - M          : Precomputed total dynamic range of the moduli set

Output:
  - X          : The unique reconstructed integer in the range [0, M - 1]

Procedure:
  1. Initialize accumulated_sum = 0

  2. For i = 1 to N (Run concurrently on parallel execution channels):
        a. Compute the product: product_i = residues[i] * inv_M_i[i]
        b. Reduce modulo m_i: scaled_residue_i = product_i (mod m_i)
        c. Scale by division constant: term_i = scaled_residue_i * M_i

  3. Perform parallel log-step reduction to sum terms (using hardware adder tree):
        accumulated_sum = Sum(term_i for i from 1 to N)

  4. Perform the final modular reduction to prevent overflow:
        X = accumulated_sum (mod M)

  5. Return X
========================================================================
```

### Mechanics of the Hardware Optimization

In traditional computers, steps 3 and 4 are bottlenecked by the modulo of a
very large integer M. In the EIU architecture, this is optimized by either:

* **Fractional CRT approximation** — storing `inv_M_i[i] / m_i` as
  high-precision fractional binary values to compute the quotient ⌊X/M⌋ via
  shift-adds.
* **Mixed-Radix Conversion (MRC)** — bypassing the need for any division or
  global modulo operations entirely, relying instead on a triangular hardware
  array of modular subtractors.
