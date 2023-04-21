""" This module defines data structures and functions defining the
linear programming energy system models used in the project"""
module ModelSetup
using JuMP, HiGHS
export SystemData, solve_system_model, AbstractConfig, AbstractGreenfield 
export GreenfieldConfig, LoadshareBioConfig, LoadshareNoBioConfig, FixedBatteryBioConfig, FixedBatteryNoBioConfig
export NIMBYBioConfig, NIMBYNoBioConfig

# Define abstract type for passing configuration settings to the model generation (remember to export these for convenience)
abstract type AbstractConfig end
abstract type AbstractGreenfield <: AbstractConfig end
struct GreenfieldConfig     <: AbstractGreenfield end
struct LoadshareBioConfig   <: AbstractGreenfield
    share::Float64
    LoadshareBioConfig(share) = (share <= 1) && (share >= 0) ? new(share) : error("Share must be between 0 and 1")
end
struct LoadshareNoBioConfig <: AbstractGreenfield
    share::Float64
    LoadshareNoBioConfig(share) = (share <= 1) && (share >= 0) ? new(share) : error("Share must be between 0 and 1")
end
struct FixedBatteryBioConfig   <: AbstractGreenfield
    epratio::Float64
end
struct FixedBatteryNoBioConfig <: AbstractGreenfield
    epratio::Float64
end
struct NIMBYBioConfig <: AbstractGreenfield
    scale::Real
end
struct NIMBYNoBioConfig <: AbstractGreenfield
    scale::Real
end

# Define a data structure for system data
struct SystemData   
    plants::AbstractArray{Symbol}
    capacities::AbstractArray
    capacityfactors::AbstractArray
    marginalcosts::AbstractArray
    fomcosts::AbstractArray
    load::AbstractArray
    storage::AbstractDict
    investmentcaps::AbstractDict
    investmentcosts::AbstractArray
end 

# Define how the system data structure is displayed after construction
function Base.show(io::IO, data::SystemData)
    # Show error messages if data is not valid
    if length(data.load) !== size(data.capacityfactors, 2)
        println(io, "WARNING: There are $(length(data.load)) hours with load data,")
        println(io, "but only $(size(data.capacityfactors, 2)) hours with capacity data.")
        println(io, "(Perhaps the hourly capacity matrix should be flipped?)")
    elseif length(data.plants) !== length(data.capacities)
        println(io, "WARNING: There are $(length(data.plants)) plants registered,")
        println(io, "but only $(length(data.capacities)) plants have capacity data")
    end
    println(io, "Electricity system data for $(length(data.plants)) plants and $(length(data.load)) hours registered:")
    for i in eachindex(data.plants)
        print(io, "$(rpad(string(data.plants[i]), maximum(length.(string.(data.plants))))): ")
        print(io, "Capacity = $(rpad(string(round(Int,data.capacities[i])), maximum(length.(string.(round.(Int,data.capacities)))))), ")
        print(io, "Marginal Cost = $(rpad(string(round(data.marginalcosts[i], digits=1)) * ",", maximum(length.(string.(round.(data.marginalcosts, digits=1)))) + 1)) ")
        print(io, "Fixed operating and maintenance costs = $(rpad(string(round(Int,data.fomcosts[i])), maximum(length.(string.(round.(Int, data.fomcosts[i]))))))")
        if data.investmentcosts != Vector()
            print(io, ", Investment costs = $(round(Int, data.investmentcosts[i]))")
        end
        println(io, "")
    end
    if data.storage != Dict()
        println(io,)
        println(io, "Additionally, a storage technology has been registered with the following characteristics:")
        print(io, "Charge efficiency = $(data.storage["Charge efficiency"]*100)%, ")
        print(io, "Discharge efficiency = $(data.storage["Discharge efficiency"]*100)%, ")
        println(io, "Daily energy loss = $(data.storage["Energy loss"]*100)%")
        print(io, "Marginal cost = $(round(data.storage["mc"], digits=1)), ")
        print(io, "Fixed operating and maintenance costs = $(round(Int, data.storage["fom"])), ")
        print(io, "Investment costs = $(round(Int, data.storage["ic"]))")
    end
    return
end

# Define a function for generating and solving a system model based on system data input
function solve_system_model(data::SystemData, config::AbstractConfig; silent::Bool=false)
    model = Model(HiGHS.Optimizer)
    # Define model variables
    add_system_variables(model, data, config)
    # Define model objective
    add_system_objective(model, data, config)
    # Define model constraints
    add_system_constraints(model, data, config)
    # Optimize the model and return it
    if silent
        set_silent(model)
    end
    optimize!(model)
    return model
end

# Define a dispatch for the Greenfield-type models
function add_system_variables(model::Model, data::SystemData, config::AbstractGreenfield)
    # Add generation variables
    @variable(model, 0 <= E[i = 1:length(data.plants), h = 1:length(data.load)])
    @variable(model, 0 <= Bi[h = 1:length(data.load)])
    @variable(model, 0 <= Bo[h = 1:length(data.load)])
    @variable(model, 0 <= St[h = 1:length(data.load)])
    # Add bound variables
    gencaps = [data.investmentcaps[string(sym)] for sym in data.plants]
    @variable(model, 0 <= E_cap[i = 1:length(data.plants)] <= gencaps[i])
    @variable(model, 0 <= E_up[i = 1:length(data.plants), h = 1:length(data.load)])
    @variable(model, 0 <= Bi_up <= data.investmentcaps["Battery - Flow"])
    @variable(model, 0 <= Bo_up <= data.investmentcaps["Battery - Flow"])
    @variable(model, 0 <= St_up <= data.investmentcaps["Battery - Storage"])
    return
end

function add_system_objective(model::Model, data::SystemData, config::AbstractGreenfield)
    # Define expressions for each type of cost
    marginal_costs   = @expression(model, sum([sum(data.marginalcosts .* model[:E][:,h]) for h in 1:length(data.load)]) + 
                                          sum((model[:Bi] + model[:Bo] ./ data.storage["Discharge efficiency"]) .* data.storage["mc"]))
    fom_costs        = @expression(model, sum(data.fomcosts .* model[:E_cap]) + 
                                          (model[:Bo_up] / data.storage["Discharge efficiency"]) * data.storage["fom"])
    investment_costs = @expression(model, sum(data.investmentcosts .* model[:E_cap]) + model[:St_up] * data.storage["ic"])
    # Combine expressions to objective
    @objective(model, Min, marginal_costs + fom_costs + investment_costs)
    return
end

function add_system_constraints(model::Model, data::SystemData, config::AbstractGreenfield)
    # Define bound constraints
    @constraint(model, capcons,  model[:E_up] .== model[:E_cap] .* data.capacityfactors)
    @constraint(model, E_cons,   model[:E]    .<= model[:E_up])
    @constraint(model, Bi_cons,  model[:Bi]   .<= model[:Bi_up])
    @constraint(model, Bo_cons,  model[:Bo]   .<= model[:Bo_up])
    @constraint(model, St_cons,  model[:St]   .<= model[:St_up])
    @constraint(model, flowcons, model[:Bi_up] == model[:Bo_up])
    # Define other constraints 
    @constraint(model, equilibrium, sum(model[:E], dims=1)[:] - model[:Bi] + model[:Bo] .== data.load)
    @constraint(model, storageflow, model[:Bi] + model[:Bo] .<= model[:Bo_up])
    @constraint(model, init_storage, model[:St][1] == 0)
    @constraint(model, storage[h = 2:length(data.load)], model[:St][h] == model[:St][h - 1]*(1 - data.storage["Energy loss"]/24) + 
                                                                          model[:Bi][h - 1]*data.storage["Charge efficiency"] - 
                                                                          model[:Bo][h - 1]/data.storage["Discharge efficiency"] )
    # if fieldnames(typeof(config)) == (:share,)
    #     @constraint(model, share, sum(model[:E][[1:3; 6], :]) >= sum(model[:E][:, :]) * config.share)
    # end
    return
end

# Add alternative contraint dispatches
function add_system_constraints(model::Model, data::SystemData, config::LoadshareBioConfig)
    add_system_constraints(model::Model, data::SystemData, GreenfieldConfig())
    @constraint(model, share, sum(model[:E][[1:3; 6], :]) >= sum(model[:E][:, :]) * config.share)
    return
end

function add_system_constraints(model::Model, data::SystemData, config::LoadshareNoBioConfig)
    add_system_constraints(model::Model, data::SystemData, GreenfieldConfig())
    @constraint(model, share, sum(model[:E][1:3, :]) >= sum(model[:E][:, :]) * config.share)
    return
end

function add_system_constraints(model::Model, data::SystemData, config::FixedBatteryBioConfig)
    add_system_constraints(model::Model, data::SystemData, LoadshareBioConfig(1.))
    @constraint(model, batterytype, model[:St_up] == model[:Bi_up] * config.epratio)
    return
end

function add_system_constraints(model::Model, data::SystemData, config::FixedBatteryNoBioConfig)
    add_system_constraints(model::Model, data::SystemData, LoadshareNoBioConfig(1.))
    @constraint(model, batterytype, model[:St_up] == model[:Bi_up] * config.epratio)
    return
end

function add_system_constraints(model::Model, data::SystemData, config::NIMBYBioConfig)
    add_system_constraints(model::Model, data::SystemData, LoadshareBioConfig(1.))
    plantindex(x) = findfirst(==(x), data.plants)
    @constraint(model, NIMBY, sum(model[:E_cap][[plantindex(:WindOnshore), plantindex(:Solar)]]) <= 
                config.scale * sum(data.capacities[[plantindex(:WindOnshore), plantindex(:Solar)]]))
    return
end

function add_system_constraints(model::Model, data::SystemData, config::NIMBYNoBioConfig)
    add_system_constraints(model::Model, data::SystemData, LoadshareNoBioConfig(1.))
    plantindex(x) = findfirst(==(x), data.plants)
    @constraint(model, NIMBY, sum(model[:E_cap][[plantindex(:WindOnshore), plantindex(:Solar)]]) <=
                config.scale * sum(data.capacities[[plantindex(:WindOnshore), plantindex(:Solar)]]))
    return
end

end