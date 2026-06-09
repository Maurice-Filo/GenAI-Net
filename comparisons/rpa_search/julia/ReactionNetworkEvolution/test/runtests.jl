using ReactionNetworkEvolution
using Test

@testset "basic functions" begin
	include("basic_functions_test.jl")
end

@testset "solver" begin
	include("solver_test.jl")
end

@testset "network mutations" begin
	include("network_mutations_test.jl")	
end

@testset "output processing" begin
	include("output_processing_test.jl")
end