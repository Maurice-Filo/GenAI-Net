settings_dict = Dict{String, Any}(
    "chemical_species_names" => ["A", "B"],
    "initial_concentrations" => [10, 0],
    )
settings = ReactionNetworkEvolution.read_usersettings(settings_dict)
smallnetwork_str = """A -> B; k1*A 
    k1 = 2
    A = 10
    B = 0
    """
smallnetwork = ReactionNetworkEvolution.convert_from_antimony(smallnetwork_str)
tspan = (0.0, 5.0)
sol = ReactionNetworkEvolution.solve_ode(smallnetwork, tspan)
# Get rate of change for A at t = 0

rateofchange = (sol.u[2][1] - sol.u[1][1])/sol.t[2]
@test abs(rateofchange + 20) < 0.00001

settings, objfunct = ReactionNetworkEvolution.read_usersettings("DEFAULT")
astr="""
    S0 + S1 -> S2; k1*S0*S1
    S2 + S2 -> S2; k2*S2*S2
    S1 -> S0; k3*S1
    S2 -> S1 + S0; k4*S2

    k1 = 1
    k2 = 3
    k3 = 0.5
    k4 = 1

    S0 = 1
    S1 = 5
    S2 = 9
    """
network = ReactionNetworkEvolution.convert_from_antimony(astr)
# Check initial rates of change
du = ReactionNetworkEvolution.test_ode_funct([0.,0.,0.], [1., 5., 9.], network, 0.1)
@test du == [6.5, 1.5, -247]

# Make sure inactive reactions aren't included in rate of change calculations
network.reactionlist[[["S1"], ["S0"]]].isactive = false
du = ReactionNetworkEvolution.test_ode_funct([0.,0.,0.], [1., 5., 9.], network, 0.1)
@test du == [4, 4, -247]