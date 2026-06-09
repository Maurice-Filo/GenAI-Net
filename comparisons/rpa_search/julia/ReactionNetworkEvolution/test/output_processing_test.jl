astr = """
// Compartments and Species:
species S0, S1, S2;

// Reactions:
_J0: S0 -> S0 + S0; k1*S0;
_J1: S0 + S1 -> S1; k2*S0*S1;
_J2: S0 + S1 -> S1 + S1; k3*S0*S1;
_J3: S0 + S2 -> S0 + S0; k4*S0*S2;
_J4: S1 -> S0 + S1; k5*S1;
_J5: S0 -> S2 + S2; k6*S0;
_J6: S2 -> S2; k7*S2;
_J7: S1 + S2 -> S2; k8*S1*S2;
_J8: S2 -> S1; k9*S2;

// Species initializations:
S0 = 1;
S1 = 5;
S2 = 9;

// Variable initializations:
k1 = 82.0537175603692;
k2 = 1.46247635550851;
k3 = 11.0307336282238;
k4 = 71.8330483162337;
k5 = 0.500520887516744;
k6 = 82.8386889676032;
k7 = 31.2822563337994;
k8 = 25.2435734826658;
k9 = 9.24024972949223;

// Other declarations:
const k1, k2, k3, k4, k5, k6, k7, k8, k9;
"""
@test ReactionNetworkEvolution.is_oscillator(astr)

broken_astr = """// Created by libAntimony v2.12.0
// Compartments and Species:
species S0, S2, S1;

// Reactions:
_J0: S0 -> S0 + S0; k1*S0;
_J1: S0 + S0 -> S0 + S2; k2*S0*S0;
_J2: S0 + S1 -> S1; k3*S0*S1;
_J3: S2 -> S0 + S1; k4*S2;
_J4: S2 -> S0 + S2; k5*S2;
_J5: S1 -> S0 + S1; k6*S1;
_J6: S1 -> S2; k7*S1;
_J7: S0 + S1 -> S2; k8*S0*S1;
_J8: S2 -> S0; k9*S2;

// Species initializations:
S0 = 1;
S2 = 9;
S1 = 5;

// Variable initializations:
k1 = 32.4337083820538;
k2 = 23.1523829004425;
k3 = 27.4435591840713;
k4 = 26.7551029533406;
k5 = 159.762534773627;
k6 = 17.6602317953216;
k7 = 3.98039116617274;
k8 = 2.66364360792616;
k9 = 50.3741152852779;

// Other declarations:
const k1, k2, k3, k4, k5, k6, k7, k8, k9;

#fitness: 0.015460235793677475"""
@test !ReactionNetworkEvolution.is_oscillator(broken_astr)
@test ReactionNetworkEvolution.is_broken_oscillator(broken_astr)

fixed_astr = """// Created by libAntimony v2.12.0
// Compartments and Species:
species S0, S2, S1;

// Reactions:
#_J0: S0 -> S0 + S0; k1*S0;
_J1: S0 + S0 -> S0 + S2; k2*S0*S0;
_J2: S0 + S1 -> S1; k3*S0*S1;
_J3: S2 -> S0 + S1; k4*S2;
_J4: S2 -> S0 + S2; k5*S2;
_J5: S1 -> S0 + S1; k6*S1;
_J6: S1 -> S2; k7*S1;
_J7: S0 + S1 -> S2; k8*S0*S1;
_J8: S2 -> S0; k9*S2;

// Species initializations:
S0 = 1;
S2 = 9;
S1 = 5;

// Variable initializations:
k1 = 32.4337083820538;
k2 = 23.1523829004425;
k3 = 27.4435591840713;
k4 = 26.7551029533406;
k5 = 159.762534773627;
k6 = 17.6602317953216;
k7 = 3.98039116617274;
k8 = 2.66364360792616;
k9 = 50.3741152852779;

// Other declarations:
const k1, k2, k3, k4, k5, k6, k7, k8, k9;

#fitness: 0.015460235793677475"""

test_fixed_astr = ReactionNetworkEvolution.fix_broken_oscillator(broken_astr)
@test startswith(test_fixed_astr, fixed_astr)