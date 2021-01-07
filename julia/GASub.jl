module GASub

using Random
using StatsBase

export Individual, initialize_population, select_parents,
        crossover, mutate, survival

struct Individual
    pod_order::Vector{Int32}
    helicopter_ids::Vector{Int32}
    helicopter_n_teams::Vector{Int32}
    cost::Vector{Float32}
    #Individual(x,y,z,a) = x > y ? error("out of order") : new(x,y)
end

function create_individual(n_teams::Int64,
                           pod_order::Array{Int64},
                           n_helicopters::Int64,
                           helicopter_ids::Array{Int64},
                           cost_func::Function)::Individual

    s_pod_order = shuffle(pod_order)
    s_helicopter_ids = shuffle(helicopter_ids)

    # Create the number of teams assigned to each helicopter
    mat = hcat(ones(Int64, n_teams), zeros(Int64, n_teams, size(helicopter_ids, 1) - 1))
    for i = 1:n_teams
        shuffle!(@view mat[i,:])
    end
    helicopter_n_teams = vec(sum(mat, dims=1))

    # Create individual
    individual = Individual(s_pod_order, s_helicopter_ids, helicopter_n_teams, [Inf32])
    cost_func(individual)

    # Return
    individual
end

function initialize_population(n::Int64,
                               helicopter_ids::Array{Int64},
                               team_ids::Array{Int64},
                               cost_func::Function)::Tuple{Individual, Array{Individual}}
    # Retrieve settings
    n_teams = size(team_ids, 1)
    n_helicopters = size(helicopter_ids, 1)

    # Teams need to be picked up and dropped off
    # Every team therefore gets 2 entries
    pod_order = vcat(team_ids, (n_teams + 1) .+ team_ids)

    # Generate population
    population = [create_individual(n_teams,
                                    pod_order,
                                    n_helicopters,
                                    helicopter_ids,
                                    cost_func) for i in 1:n]

    # Sort by cost
    sort!(population, by=x->x.cost[1])
    best_sol = population[1]

    # Return
    best_sol, population
end

function calc_weights(n::Int64, p::Float32)::Array{Float32}
    weights = p * ((1-p) .^ collect(0:n-1))
    weights[n] /= p
    weights
end

function select_parents(population::Array{Individual},
                        weights::Weights{Float32,Float32,Array{Float32,1}})::Tuple{Individual, Individual}
    parents = sample(population, weights, (2); replace=true)
    (parents[1], parents[2])
end

function crossover(parent1::Individual,
                   parent2::Individual)::Tuple{Individual, Individual}

    function double_point(p1, p2)

        function not_in(a, b)
            setb = Set(b)
            [el for el in a if !(el in setb)]
        end

        # Determine cutpoints
        n = size(p1,1)
        rand_idxs = rand(1:n,2)
        cut_start = minimum(rand_idxs)
        cut_end = maximum(rand_idxs)

        # Make cuts
        cut1 = p1[cut_start:cut_end]
        cut2 = p2[cut_start:cut_end]

        # Make pre and post segments
        rest1 = not_in(p2, cut1)
        pre1 = cut_start > 1 ? rest1[1:cut_start-1] : zeros(Int64, 0)
        post1 = cut_end < n ? rest1[cut_start:end] : zeros(Int64, 0)

        rest2 = not_in(p1, cut2)
        pre2 = cut_start > 1 ? rest2[1:cut_start-1] : zeros(Int64, 0)
        post2 = cut_end < n ? rest2[cut_start:end] : zeros(Int64, 0)

        # Assemble results
        c1 = vcat(pre1, cut1, post1)
        c2 = vcat(pre2, cut2, post2)

        c1, c2
    end

    # Crossover team seqs
    ts1, ts2 = double_point(parent1.pod_order,
                            parent2.pod_order)

    # Assemble children
    child1 = Individual(ts1, parent1.helicopter_ids, parent1.helicopter_n_teams, [Inf32])
    child2 = Individual(ts2, parent2.helicopter_ids, parent2.helicopter_n_teams, [Inf32])

    child1, child1

end

function mutate(mutate_p::Float32,
                child::Individual)::Nothing
    if rand(1,1)[1] < mutate_p
        # Change route lengths
        n_helicopters = size(child.helicopter_ids, 1)
        if n_helicopters > 1
            non_zero_idxs = findall(>(0), child.helicopter_n_teams)
            shuffle!(non_zero_idxs)
            child.helicopter_n_teams[non_zero_idxs[1]] -= 1

            random_idx = rand(1:n_helicopters,1)[1]
            child.helicopter_n_teams[random_idx] += 1
        end

        # Switch 2 elements in the team_seqs
        n_teams_seq = size(child.pod_order,1)
        rand_idxs = rand(1:n_teams_seq,2)
        tmp = child.pod_order[rand_idxs[1]]
        child.pod_order[rand_idxs[1]] = child.pod_order[rand_idxs[2]]
        child.pod_order[rand_idxs[2]] = tmp
    end
    nothing
end

function survival(population::Array{Individual},
                  children::Array{Individual})::Array{Individual}
    # Merge and sort the population and the children
    n = size(population,1)
    all = vcat(population, children)
    sort!(all, by=x->x.cost[1])

    # Return n best
    all[1:n]
end

end
