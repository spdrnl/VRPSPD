module GACost

using OrderedCollections

function calc_distances(locations)::Array{Float32}
    n_locations = size(locations, 1)
    distances = zeros(Float32, n_locations, n_locations)
    for i in 1:n_locations
        for j in 1:n_locations
            distances[i, j] = sqrt(sum((locations[i, :] - locations[j, :]).^2))
        end
    end
    distances
end

function calc_helicopter(n_teams,
                         team_sizes::Array{Int64},
                         team_transports::Array{Int64},
                         pick_or_drop_seq::Array{Int64},
                         distances::Array{Float32})::Tuple{Float32, Float32}
    # Team tally
    pick_or_drop = zeros(Int64, n_teams)

    # Implicit start base
    helicopter_location = 1

    # Helpers to calculate the max_load
    helicopter_max_load = 0
    helicopter_distance = 0
    helicopter_load_delta = 0

    # Resolve the planning for this helicopter
    for team_id in pick_or_drop_seq
        if pick_or_drop[team_id] == 0 # Pickup of team
            pick_or_drop[team_id] += 1
            to_location = team_transports[team_id, 1]
            team_load_delta = team_sizes[team_id]
        else # Drop off of teams
            to_location = team_transports[team_id, 2]
            team_load_delta = -team_sizes[team_id]
        end

        # Add the distance to the helicopters distance
        helicopter_distance += distances[helicopter_location, to_location]

        # Resolve max loads
        if to_location == helicopter_location
            helicopter_load_delta += team_load_delta
        else
            helicopter_max_load = max(helicopter_max_load, helicopter_max_load + helicopter_load_delta)
            helicopter_load_delta = 0
        end

        helicopter_location = to_location
    end

    # Return
    helicopter_distance, helicopter_max_load

end

function calc(individual, team_sizes, team_transports, helicopter_capacities, distances)
    # Retrieve settings
    n_teams = size(team_sizes,1)
    n_helicopters = size(helicopter_capacities,1)
    pod_order = individual.pod_order .% (n_teams + 1)
    helicopter_ids = individual.helicopter_ids
    helicopter_n_teams = individual.helicopter_n_teams

    # In which order do the teams appear in the team_seq?
    team_order = collect(OrderedSet(pod_order))

    # Initialize the planning
    helicoper_distances = zeros(n_helicopters)
    helicopter_max_loads = zeros(n_helicopters)

    # Start the planning
    start = 1
    for (helicopter_id, number_of_teams) in zip(helicopter_ids, helicopter_n_teams)
        # Anything to do?
        if number_of_teams == 0
            continue
        end

        # Get the unique teams assigned to the helicopter
        team_assignments = team_order[start:start+number_of_teams-1]
        start += number_of_teams

        # Get the pickup and dropof order of the assigned teams
        pick_or_drop_seq = [team for team in pod_order if team in team_assignments]

        helicopter_distance, helicopter_max_load = calc_helicopter(n_teams,
                                                                    team_sizes,
                                                                    team_transports,
                                                                    pick_or_drop_seq,
                                                                    distances)
        helicoper_distances[helicopter_id] = helicopter_distance
        helicopter_max_loads[helicopter_id] = helicopter_max_load
    end

    # Return
    helicoper_distances, helicopter_max_loads
end

function cost(individual, team_sizes, team_transports, helicopter_capacities, distances)
    # Calculate the planning
    helicoper_distances, helicopter_max_loads = calc(individual, team_sizes, team_transports, helicopter_capacities, distances)

    # Distance cost
    distance_cost = sum(helicoper_distances)

    # Max load costs
    max_load_cost = 1000 * sum(helicopter_capacities - helicopter_capacities)

    # Calculate the cost for this planning
    cost = distance_cost + max_load_cost

    # Update
    individual.cost[1] = cost

    # Return
    cost
end

end
