module GAAlg

export execute

#include("./GASub.jl")
using ..GARun

function execute(n_iterations::Int64,
    best_sol::GARun.GASub.Individual,
    population::Array{GARun.GASub.Individual},
    select_parents_func::Function,
    crossover_func::Function,
    mutate_func::Function,
    survival_func::Function,
    cost_func::Function)::GARun.GASub.Individual

    # Iterate generations
    for iteration in 1:n_iterations
        best_sol, population = step(best_sol,
            population,
            select_parents_func,
            crossover_func,
            mutate_func,
            survival_func,
            cost_func)

        #print("\e[2K")
        #print("\e[1G")
        println("Iteration $iteration, cost $(best_sol.cost[1])")
    end

    best_sol
end

function step(best_sol::GARun.GASub.Individual,
    population::Array{GARun.GASub.Individual},
    select_parents_func::Function,
    crossover_func::Function,
    mutate_func::Function,
    survival_func::Function,
    cost_func::Function)::Tuple{GARun.GASub.Individual, Array{GARun.GASub.Individual}}

    n_children = size(population, 1)
    children = Array{GARun.GASub.Individual}(undef, n_children)

    for i = 1:n_children ÷ 2
        # Select parents
        parent1, parent2 = select_parents_func(population)

        # Crossover
        child1, child2 = crossover_func(parent1, parent2)

        # Mutate
        mutate_func(child1)
        mutate_func(child2)

        # Apply boundaries
        # TODO

        # Set costs for the children
        cost_func(child1)
        if best_sol.cost[0] > child1.cost[0]
            best_sol = child1
        end

        cost_func(child2)
        if best_sol.cost[0] > child2.cost[0]
            best_sol = child2
        end

        # Add to children
        children[(2*i) - 1] = child1
        children[2*i] = child2
    end

    # Select next generation
    population = survival_func(population, children)

    # Return
    best_sol, population
end

end
