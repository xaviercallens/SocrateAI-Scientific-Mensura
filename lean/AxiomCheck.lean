import CallensDualScale
import QuantumFluidsShell

#print axioms CallensDualScale.genesis_no_singularity
#print axioms CallensDualScale.Reff_ge_sqrt
#print axioms CallensDualScale.sym2_recurrence

-- Cross-stream import from SocrateAI-Scientific-QuantumFluids (2026).
-- Eleven theorems; all must show [propext, Classical.choice, Quot.sound] only.
#print axioms QuantumFluids.ShellComplex.shellBc_energy_conservation
#print axioms QuantumFluids.ShellComplex.shellBc_real
#print axioms QuantumFluids.ShellComplex.shell_divergence_zero
#print axioms QuantumFluids.ShellComplex.seam_conserves_iff
#print axioms QuantumFluids.ShellComplex.seam_zero_conserves
#print axioms QuantumFluids.ShellComplex.seam_gpe_conserves
