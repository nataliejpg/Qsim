# Qsim

The structure is:

qsim/
	*method*/ - exact, pms, rk4
		state_vectors.py - creating state vectors and changing between representations
		hamiltonians.py - create hamiltonian in appropriate format where relevant
		unitaries.py - create unitary in appropriate format where revant
		methods.py - evolving a state under a hamiltonian and doing measurements
		*other_helpers* - generally hcak and need refactoring
	helpers.py
	plotting.py
	monte_carlo.py - needs refactoring into methods
