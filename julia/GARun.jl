module GARun

include("./GASub.jl")
include("./GACost.jl")
include("./GAAlg.jl")

using .GASub
using .GAAlg
using .GACost: calc_distances, cost

using Random
using StatsBase: sample, Weights

function run()
    # Make runs reproducible
    Random.seed!(1234)

    # Settings
    n_teams = 16
    max_team_size = 8
    n_platforms = 10
    n_helicopters = 4
    helicopter_capacities = [25, 25, 15, 15]

    # Generate environment
    helicopter_ids = collect(1:n_helicopters)

    platform_ids = collect(1:n_platforms)
    a::Float32 = 0.0
    b::Float32 = 100.0
    platform_locations = rand(a:b, (n_platforms, 2))
    distances = calc_distances(platform_locations)

    team_ids = collect(1:n_teams)
    team_sizes = rand(1:max_team_size, n_teams, 1)
    team_locations = sample(platform_ids, n_teams; replace = true)
    team_assignments = sample(platform_ids, n_teams; replace = true)
    team_transports = hcat(team_locations, team_assignments)

    ###########################################################################
    # Hyper-parameter settings
    ###########################################################################
    # Iterations and population size
    n_iterations = 1000
    n = 100

    # Parent selection
    select_p::Float32 = 0.1
    weights = Weights(GASub.calc_weights(n, select_p))

    # Mutation
    mutate_p::Float32 = 0.1

    # Select relevant subroutines for the GA algorithm
    cost_func(individual) = cost(individual, team_sizes, team_transports, helicopter_capacities, distances)
    select_parents_func(population::Array{Individual}) = select_parents(population, weights)
    crossover_func(parent1::Individual, parent2::Individual) = crossover(parent1, parent2)
    mutate_func(child::Individual) = mutate(mutate_p, child)
    survival_func(population::Array{GASub.Individual}, children::Array{GASub.Individual}) = survival(population, children)

    # Execute the algorithm
    best_sol, population = initialize_population(n, helicopter_ids, team_ids, cost_func)
    best_sol = execute(n_iterations,
        best_sol,
        population,
        select_parents_func,
        crossover_func,
        mutate_func,
        survival_func,
        cost_func)

    println()
    println(best_sol)
end

@time run()
@time run()

end
