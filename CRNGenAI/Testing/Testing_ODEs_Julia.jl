using Catalyst
using OrdinaryDiffEq
using Plots

# Define the reaction network
sAIF_Network = @reaction_network begin
    @parameters γ μ θ η k γ₁ γ₂              # Declare parameters explicitly
    γ, X₁ --> ∅
    μ, ∅ --> Z₁
    θ, X₁ --> X₁ + Z₂
    η, Z₁ + Z₂ --> ∅
    k, Z₁ --> Z₁ + X₁
    γ₁, Z₁ --> ∅
    γ₂, Z₂ --> ∅
end


# Set initial conditions for species
initial_conditions = [:X₁ => 0.0, :Z₁ => 0.0, :Z₂ => 0.0] 

# Define parameter values
p = [
    :γ => 1,  
    :μ => 100,   
    :θ => 1,   
    :η => 10, 
    :k => 10,  
    :γ₁ => 0, 
    :γ₂ => 0,  
]

# Define time span for simulation
tspan = (0.0, 20.0)

# Run the simulation
tic = time()
prob = ODEProblem(sAIF_Network, initial_conditions, tspan, p)
sol = solve(prob, Tsit5())
toc = time()
println("Simulation Time: ", toc-tic, " seconds")

# Plot the results
plot(sol, xlabel="Time", ylabel="Concentrations", legend=:topright)
