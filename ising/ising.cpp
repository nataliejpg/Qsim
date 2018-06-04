#include <vector>
#include <complex>
#include <cassert>
#include <cmath>
#include <iostream>

/* defines wavefuntion to be a vector made of complex doubles */
using Wavefunction = std::vector<std::complex<double>>;

/* function returning the energy of a configuration of spins */
double ising_energy(unsigned s, double J, unsigned L)
{
	double energy = 0;
	for(unsigned i = 0; i < L; ++i)
	{
		unsigned si = (s >> i) & 1;
		unsigned j = (i + 1) % L;
		unsigned sj = (s >> j) & 1;
		energy += (si == sj? -J: J);
	}
	return energy;
}

/*  function which does the Z part of the ising hamiltonian on an input wavefunction
and writes it to an output one */
void Hz(double J, unsigned L, Wavefunction const&in, Wavefunction &out)
{
	assert(in.size() == out.size());
	assert(in.size() >= (1<<L));
	for(unsigned s = 0; s < (1<<L); ++s)
		out[s] = ising_energy(s, J, L) * in[s];	
}

/*  function which does the X part of the ising hamiltonian on an input wavefunction
and writes it to an output one */
void Hx(double g, unsigned L, Wavefunction const&in, Wavefunction &out)
{
	assert(in.size() == out.size());
	assert(in.size() >= (1<<L));
	for(unsigned i = 0; i < L; ++i)
	{
		unsigned mask = (1 << i);
		for(unsigned s = 0; s < (1<<L); ++s)
			out[s] += g * in[s ^ mask];
	}
}

/*  function which does the Z and then X part of the ising hamiltonian on an input wavefunction
and writes it to an output one */
void H(double g, double J, unsigned L, Wavefunction const&in, Wavefunction &out)
{
	Hz(J, L, in, out);
	Hx(g, L, in, out);
}


/* function which evolves the wavefunction under the Z part of the hamiltonian */
void Uz(double J, unsigned L, double delta_t, Wavefunction &in)
{
	assert(in.size() >= (1<<L));
	for(unsigned s = 0; s < (1<<L); ++s)
		in[s] = std::exp(std::complex<double>(0., -1.) * delta_t * ising_energy(s, J, L)) * in[s];	
}

/* function which evolves the wavefunction under the X part of the hamiltonian for a single qubit i */
void Ux(double g, unsigned L, double delta_t, unsigned i, Wavefunction const&in, Wavefunction &out)
{
	assert(in.size() == out.size());
	assert(in.size() >= (1<<L));
	for(unsigned s = 0; s < (1<<L); ++s)
	{
		unsigned mask = (1 << i);
		for(unsigned s = 0; s < (1<<L); ++s)
			out[s] = std::cos(g * delta_t) * in[s] + std::complex<double>(0., -1.) * std::sin(g * delta_t) * in[s ^ mask];
	}
}

/* function which evolves the state under the X hamiltonian for each qubit */
void Ux(double g, unsigned L, double delta_t, Wavefunction &in, Wavefunction &out)
{
	for(unsigned i = 0; i < L; ++i)
	{
		Ux(g, L, delta_t, i, in, out);
		std::swap(in, out);
	}
	std::swap(in, out);
}

/* function which does evolution under Z and then X parts of the ising hamiltonian*/
void U(double g, double J, unsigned L, double delta_t, Wavefunction &in, Wavefunction &out)
{
	Uz(J, L, delta_t, in);
	Ux(g, L, delta_t, in, out);
}

/* function which measures along Z axis the magnetisation of qubit i */
double measure(unsigned L, unsigned i, Wavefunction const&in)
{
	double result = 0;
	for(unsigned s = 0; s < (1 << L); ++s)
	{
		result += std::abs(in[s]) * std::abs(in[s]) * (((s >> i) & 1) == 1? -1: 1);
	}
	return result;
}

int main()
{
	unsigned L;
	double J;
	double g;
	double delta_t;
	double t;
	std::cin >> L >> J >> g >> delta_t >> t;
	Wavefunction psi(1<<L);
	Wavefunction phi(1<<L);
	unsigned s;
	std::cin >> s;
	psi[s] = 1.;
	for(double i = 0; i < t; i += delta_t)
	{
		U(g, J, L, delta_t, psi, phi);
		std::swap(psi, phi);
	}
	std::swap(psi, phi);
	// change this to second order

	for(unsigned i = 0; i < L; i++)
		std::cout << measure(L, i, phi) << std::endl;
}

