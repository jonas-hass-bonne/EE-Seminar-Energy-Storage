# Load packages used in the project
using Revise                # QoL packages
using JuMP, HiGHS           # Modelling packages
using Plots, StatsPlots     # Plotting packages
using DataFrames, CSV, XLSX # Data wrangling packages

# Load custom modules for data loading and modelling
includet("modelsetup.jl")
includet("dataloading.jl")
includet("modelplots.jl")

using .ModelSetup, .DataLoading, .ModelPlots

########################
# FUNCTION DEFINITIONS #
########################

# Define convenience functions for styled printing
styleprintln(x::String; color::Union{Symbol, Integer}=:white) = printstyled(x * "\n", bold=true, reverse=true, color=color)
styleprint(x::String; color::Union{Symbol, Integer}=:white) = printstyled(x, bold=true, reverse=true, color=color)
function threadprint(iter::Integer, thread::Integer)  # Convenience function for producing styled printing during multi-threaded loop
    print("Solving model ") 
    styleprint(lpad(iter, 2), color=iter+70)
    print(" on thread ")
    styleprint(string(thread), color=thread + 51)
    println("... ")
end

# Define convenience function to compute curtailment in greenfield models
curtailment(model::Model) = 1 - sum(value.(model[:E][1:3,:]))/sum(value.(model[:E_up][1:3,:]))

# Define function to gather relevant results from model simulations
function scenarioreport(model::Model, reference::Real)
    systemcosts = objective_value(model)
    totalgen    = sum(value.(model[:E]))
    totalstored = sum(value.(model[:Bi]))
    curtailed   = curtailment(model)
    powercaps   = value.(model[:E_cap])
    totalcap    = sum(powercaps)
    flowcap     = value.(model[:Bi_up])
    storagecap  = value.(model[:St_up])
    epratio     = storagecap/flowcap
    relsyscosts = (systemcosts / reference)*100
    return [systemcosts; totalgen; totalstored; curtailed; powercaps; totalcap; 
            flowcap; storagecap; epratio; relsyscosts]
end

# Define convenience function for solving a model, saving output and producing figures
function standardsolve(data::SystemData, config::AbstractConfig, name::String, 
                        df::DataFrame, reference::Real)
    # Solve the model
    styleprintln("Solving the " * name * " model")
    out = solve_system_model(data, config)
    println("")

    # Save model output in dataframe
    df[:, name] = scenarioreport(out, reference)

    # Produce graphs for the output
    plot_hourly_gen(data, out, name)
    plot_hourly_storage(data, out, name)
    plot_capacities(data, out, name)
    plot_generation(data, out, name)
    return out
end

# Define convenience dispatch with a baseline model
standardsolve(data::SystemData, config::AbstractConfig, name::String, 
                df::DataFrame) = standardsolve(data, config, name, df, baseref)

################
# DATA LOADING #
################

# Load data
df_h    = loadhourlyOPSdata(2018)
df_c    = loadcapacityOPSdata(2018)
df_cf   = loadcapfactorOPSdata(2018)
sheet_c = loadcoststructure(2)
sheet_b = loadbatterydata(2)
IEA_vec = loadIEAdata()

# Initialise vectors with capacity constraints and marginal costs
plants  = [:WindOffshore, :WindOnshore, :Solar, :Coal, :NaturalGas, :Bio]   # Vector of symbols for each plant type 
q_vec   = [df_c[occursin.(x, df_c[:,1]),2][1] for x in ["Offshore", "Onshore", "Solar", "coal", "Natural gas"]]
q_vec   = [q_vec; sum(df_c[occursin.("bio",df_c[:,1]), 2])]                         # Vector of generation capacities
c_vec   = Vector{Float64}(sheet_c[sheet_c[:,1] .== "Total variable cost (EUR/MWh)", 2:end][:])    
c_vec   = c_vec[[5, 4, 6, 1, 2, 3]]                                                 # Vector of marginal costs
fom_vec = Vector{Float64}(sheet_c[sheet_c[:,1] .== "FOM (euro/MW/year)", 2:end][:]) 
fom_vec = fom_vec[[5, 4, 6, 1, 2, 3]]                                               # Vector of fixed operating and maintenance costs
L_vec   = Vector{Float64}(df_h[:,findfirst(occursin.("load_actual",names(df_h)))])  # Vector of load at different hours                                                   # Marginal WTP to avoid load-shedding

# Initialise matrices with hourly capacity constraints
cf_mat = [reshape(df_cf[:,findfirst(occursin.("offshore", names(df_cf)))], (1,:)) # Initialise matrix of gamma values with intermittent generator values
             reshape(df_cf[:,findfirst(occursin.("onshore", names(df_cf)))], (1,:))   
             reshape(df_cf[:,findfirst(occursin.("pv", names(df_cf)))], (1,:))]
while size(cf_mat, 1) < length(q_vec)    # Update gamma matrix until all generators have gamma values
    global cf_mat = [cf_mat; reshape([1 for i in 1:length(L_vec)], (1,:))]  # Update gamma matrix with 1's for dispatchable generators
end

# Create a vector with required load-shedding given capacity constraints and load
S_vec = (L_vec - sum(q_vec .* cf_mat, dims=1)[:] .>= 0) .* (L_vec - sum(q_vec .* cf_mat, dims=1)[:])
L_vec -= S_vec  # Subtract load shedding from demand to ensure equilibrium is feasible in all periods

# Prepare DataFrame for saving simulation results
scenario_df = DataFrame((Rownames=["System costs (Euro)"; "Total energy generation (MWh)"; 
                        "Total energy stored (MWh)"; "Curtailment (%)"; 
                        ["Installed capacity for " * string(x) for x in plants] .* " (MW)"; "Total installed capacity (MW)";
                        "Installed storage flow capacity (MW)"; "Installed storage capacity (MWh)"; 
                        "Energy/Power ratio (MWh/MW)"; "System costs relative to baseline"]))

# Load data for lithium battery costs and efficiencies
b_dict = Dict(["Charge efficiency", "Discharge efficiency", "mc", "fom"] .=> sheet_b[[3, 4, 7, 8], 2])
b_dict["Energy loss"] = 0.001       # Add daily energy loss
b_dict["Charge efficiency"] /= 100  # Ensure efficiency stored as percentage
b_dict["Discharge efficiency"] /= 100
b_dict["fom"]   += sheet_b[14,2]    # Add amortized investment costs to FOM
b_dict["ic"]    = sheet_b[13, 2]    # Add amortized storage investment costs

# Define maximum storage capacities
icap_dict = Dict(["Battery - Flow", "Battery - Storage"] .=> [0., 0.])
# Add maximum installed capacity constraints for regular plants
for plant in plants
    icap_dict[string(plant)] = q_vec[findfirst(==(plant), plants)]
end

# Define investment costs
ic_vec = Vector{Float64}(sheet_c[sheet_c[:,1] .== "Yearly investment", 2:end][:])
ic_vec = ic_vec[[5, 4, 6, 1, 2, 3]] # Vector of investment costs

# Create a new SystemData object with the updated data for storage
baseline_data = SystemData(plants, q_vec, cf_mat, c_vec, fom_vec, L_vec, b_dict, icap_dict, ic_vec)

# Create a plot of load and capacity factors
plot_hourly_capfactors(baseline_data, "baseline")

########################
# MAIN MODEL SOLUTIONS #
########################

# Solve the initial storage model including investment costs for storage
baseref  = objective_value(solve_system_model(baseline_data, GreenfieldConfig(), silent=true))   # Generate a baseline system cost
base_out = standardsolve(baseline_data, GreenfieldConfig(), "baseline", scenario_df)

# Plot slices of hourly generation sorted by plant type for IEA data
plot_generation(baseline_data, IEA_vec, "IEA")

# Construct data object for greenfield model
for plant in ["WindOffshore", "WindOnshore", "Solar", "Coal", "NaturalGas", "Bio"]
    icap_dict[plant] = 1e9   # Allow unlimited capacities for each plant
end
greenfield_data = SystemData(plants, q_vec, cf_mat, c_vec, fom_vec, L_vec, b_dict, icap_dict, ic_vec)

# Solve the greenfield model
# greenfield_out = standardsolve(greenfield_data, GreenfieldConfig(), "greenfield", scenario_df)

# Solve the greenfield model for varying renewable shares
lk = Threads.SpinLock() # Define a lock for multi-threaded loops
outvecbio   = Vector{Model}(undef, 5)   # Define vector for storing models including bio
outvecnobio = Vector{Model}(undef, 5)   # Define vector for storing models excluding bio

styleprintln("Solving the greenfield model for different renewable shares")
Threads.@threads for i in 1:(length(outvecbio) + length(outvecnobio))
    lock(lk)
        threadprint(i, Threads.threadid())
    unlock(lk)
    if i <= length(outvecbio)
        global outvecbio[i] = solve_system_model(greenfield_data, LoadshareBioConfig(.5 + i/10), silent=true)
    else
        global outvecnobio[i-length(outvecbio)] = solve_system_model(greenfield_data, LoadshareNoBioConfig(.5 + (i - length(outvecbio))/10), silent=true)
    end
end
println("")

styleprintln("Saving output from model simulations")
# Create vectors for storing model outputs
bioresults   = deepcopy(objective_value.(outvecbio))
nobioresults = deepcopy(objective_value.(outvecnobio))
# Save output to DataFrame and generate plots
for outvec in [outvecbio, outvecnobio] 
    # Set name to use
    if outvec == outvecbio
        dfname  = "Greenfield (Bio) ("
        figname = "greenfield_bio_"
    else
        dfname  = "Greenfield (No Bio) ("
        figname = "greenfield_nobio_"
    end
    for (i, model) in enumerate(outvec) 
        # Save results to DataFrame
        global scenario_df[:, dfname * string(50 + 10*i) * "% cap)"] = scenarioreport(model, baseref)
        # Plot generation, capacities and residual demand
        plot_hourly_gen(greenfield_data, model, figname * string(50 + 10*i) * "%")
        plot_hourly_storage(greenfield_data, model, figname * string(50 + 10*i) * "%")
        plot_capacities(greenfield_data, model, figname * string(50 + 10*i) * "%")
        plot_generation(greenfield_data, model, figname * string(50 + 10*i) * "%")
    end
end

# Now enable storage investments
icap_dict["Battery - Flow"] = 1e9
icap_dict["Battery - Storage"] = 1e9

# greenfield2_out = standardsolve(greenfield_data, GreenfieldConfig(), "greenfield2", scenario_df)

# Solve the greenfield model with storage for varying renewable shares
styleprintln("Solving the greenfield model with storage for different renewable shares")
Threads.@threads for i in 1:(length(outvecbio) + length(outvecnobio))
    lock(lk)
        threadprint(i, Threads.threadid())
    unlock(lk)
    if i <= length(outvecbio)
        global outvecbio[i] = solve_system_model(greenfield_data, LoadshareBioConfig(.5 + i/10), silent=true)
    else
        n = i - length(outvecbio)
        global outvecnobio[n] = solve_system_model(greenfield_data, LoadshareNoBioConfig(.5 + n/10), silent=true)
    end
end
println("")

styleprintln("Saving output from model simulations")
for outvec in [outvecbio, outvecnobio] 
    # Set name to use
    if outvec == outvecbio
        dfname  = "Greenfield - Storage (Bio) ("
        figname = "greenfield_storage_bio_"
        plot_systemcosts(bioresults, outvec, "bio")
    else
        dfname  = "Greenfield - Storage (No Bio) ("
        figname = "greenfield_storage_nobio_"
        plot_systemcosts(nobioresults, outvec, "nobio")
    end
    for (i, model) in enumerate(outvec) 
        # Save results to DataFrame
        global scenario_df[:, dfname * string(50 + 10*i) * "% cap)"] = scenarioreport(model, baseref)
        # Plot generation, capacities and residual demand
        plot_hourly_gen(greenfield_data, model, figname * string(50 + 10*i) * "%")
        plot_hourly_storage(greenfield_data, model, figname * string(50 + 10*i) * "%")
        plot_capacities(greenfield_data, model, figname * string(50 + 10*i) * "%")
        plot_generation(greenfield_data, model, figname * string(50 + 10*i) * "%")
    end
end

########################
# Sensitivity analysis #
########################

#########################################################################
# NIMBY constraint for Solar (10x baseline) and OnshoreWind (2x baseline)

# icap_dict["Solar"]             = 3*baseline_data.capacities[baseline_data.plants .== :Solar][1]
# icap_dict["WindOnshore"]       = 3*baseline_data.capacities[baseline_data.plants .== :WindOnshore][1]
icap_dict["Battery - Flow"]    = 0 # Compute NIMBY constrained model without storage
icap_dict["Battery - Storage"] = 0

greenfield_nimby_nostorage_bio_out   = standardsolve(greenfield_data, NIMBYBioConfig(3.), "greenfield_nimby_nostorage_bio", scenario_df)
greenfield_nimby_nostorage_nobio_out = standardsolve(greenfield_data, NIMBYNoBioConfig(3.), "greenfield_nimby_nostorage_nobio", scenario_df)

# Now compute NIMBY constrained model with storage
icap_dict["Battery - Flow"]    = 1e9
icap_dict["Battery - Storage"] = 1e9

greenfield_nimby_storage_bio_out   = standardsolve(greenfield_data, NIMBYBioConfig(3.), "greenfield_nimby_storage_bio", scenario_df)
greenfield_nimby_storage_nobio_out = standardsolve(greenfield_data, NIMBYNoBioConfig(3.), "greenfield_nimby_storage_nobio", scenario_df)

# Reset investment caps for Solar and OnshoreWind plants
# icap_dict["Solar"]       = 1e9
# icap_dict["WindOnshore"] = 1e9

#################################################
# Battery type constrained to be a 4-hour battery

icap_dict["Battery - Flow"]    = 0 # Compute battery-type constrained model without storage
icap_dict["Battery - Storage"] = 0

greenfield_batconstraint_nostorage_bio_out   = standardsolve(greenfield_data, FixedBatteryBioConfig(4.), "greenfield_batconstraint_nostorage_bio", scenario_df)
greenfield_batconstraint_nostorage_nobio_out = standardsolve(greenfield_data, FixedBatteryNoBioConfig(4.), "greenfield_batconstraint_nostorage_nobio", scenario_df)

# Now compute battery-type constrained model with storage
icap_dict["Battery - Flow"]    = 1e9
icap_dict["Battery - Storage"] = 1e9

greenfield_batconstraint_storage_bio_out   = standardsolve(greenfield_data, FixedBatteryBioConfig(4.), "greenfield_batconstraint_storage_bio", scenario_df)
greenfield_batconstraint_storage_nobio_out = standardsolve(greenfield_data, FixedBatteryNoBioConfig(4.), "greenfield_batconstraint_storage_nobio", scenario_df)

plot_systemcosts([bioresults[5], objective_value(greenfield_nimby_nostorage_bio_out), objective_value(greenfield_batconstraint_nostorage_bio_out)],
                 [outvecbio[5], greenfield_nimby_storage_bio_out, greenfield_batconstraint_storage_bio_out], "nimby_battery_bio", 
                 xvals = [x for _ in 1:2, x in ["Baseline", "NIMBY-constrained", "Battery-type-constrained"]][:],
                 xlabel = "Simulation scenario")
plot_systemcosts([nobioresults[5], objective_value(greenfield_nimby_nostorage_nobio_out), objective_value(greenfield_batconstraint_nostorage_nobio_out)],
                 [outvecnobio[5], greenfield_nimby_storage_nobio_out, greenfield_batconstraint_storage_nobio_out], "nimby_battery_nobio",
                 xvals = [x for _ in 1:2, x in ["Baseline", "NIMBY-constrained", "Battery-type-constrained"]][:],
                 xlabel = "Simulation scenario")

############################################################################################
# Higher fixed operating and maintenance costs and investment costs for storage technologies
# Loop through increasing cost scenarios and solve the model,
# increasing the cost by 10 percentage points each iteration

# Generate separate datasets and model outputs to allow for multithreaded solution
datavec = Vector{SystemData}(undef, 21)
for (i, scale) in enumerate(0.5:.1:2.5)
    cur_dict = deepcopy(b_dict)
    cur_dict["fom"] *= scale
    cur_dict["ic"]  *= scale
    global datavec[i] = SystemData(plants, q_vec, cf_mat, c_vec, fom_vec, L_vec, cur_dict, icap_dict, ic_vec)
end
outvecbio   = Vector{Model}(undef, 21)
outvecnobio = Vector{Model}(undef, 21) 

# Generate a vector with tuple of values used for solving the model and saving the output
loopvec = [(i, "Greenfield - Storage " * y, data) 
            for (i,y,data) in zip(1:21, ["(" * string(x) * "% cost increase)" for x in -50:10:150], datavec)]

# Run the multithreaded loop to solve model for each cost
styleprintln("Solve the model for increasing storage costs (takes a while)")
Threads.@threads for (i, name, data) in loopvec
    lock(lk)
        threadprint(i, Threads.threadid())
    unlock(lk)
    global outvecbio[i] = solve_system_model(data, LoadshareBioConfig(1.), silent=true)
    global outvecnobio[i] = solve_system_model(data, LoadshareNoBioConfig(1.), silent=true)
end
println("")

styleprintln("Saving output from model simulations")
for outvec in [outvecbio, outvecnobio] 
    # Set name to use
    if outvec == outvecbio
        dfname  = "Greenfield - Storage (Bio) ("
        figname = "greenfield_storage_bio_"
        plot_storagecosts(outvec, bioresults[5], figname)
    else
        dfname  = "Greenfield - Storage (No Bio) ("
        figname = "greenfield_storage_nobio_"
        plot_storagecosts(outvec, nobioresults[5], figname)
    end
    for (i, model) in enumerate(outvec) 
        # Save results to DataFrame
        global scenario_df[:, dfname * string(10*i) * "% cost increase)"] = scenarioreport(model, baseref)
    end
    
end

######################################################
# Using alternative data for load and capacity factors

# Load data for each year and save the associated system data
datavec = Vector{SystemData}(undef, 5)
Threads.@threads for (i,year) in [(1, 2015), (2, 2016), (3, 2017), (4, 2018), (5, 2019)]
    tmpin   = loadhourlyOPSdata(year)
    if year == 2015
        tmpin[1:2, findfirst(occursin.("load_actual", names(tmpin)))] = [3286., 3211.]
    end
    tmpload = Vector{Float64}(tmpin[:, findfirst(occursin.("load_actual", names(tmpin)))])
    tmpin   = loadcapfactorOPSdata(year)
    tmpcf   = [reshape(tmpin[:,findfirst(occursin.("offshore", names(tmpin)))], (1,:))
               reshape(tmpin[:,findfirst(occursin.("onshore", names(tmpin)))], (1,:))   
               reshape(tmpin[:,findfirst(occursin.("pv", names(tmpin)))], (1,:))
               reshape([1 for _ in 1:length(tmpload)], (1,:))
               reshape([1 for _ in 1:length(tmpload)], (1,:))
               reshape([1 for _ in 1:length(tmpload)], (1,:))]
    tmpshed = (tmpload - sum(q_vec .* tmpcf, dims=1)[:] .>= 0) .* (tmpload - sum(q_vec .* tmpcf, dims=1)[:])
    tmpload -= tmpshed
    global datavec[i] = SystemData(plants, q_vec, tmpcf, c_vec, fom_vec, tmpload, b_dict, icap_dict, ic_vec)
end
for (data, year) in [(datavec[i], string(i+2014)) for i in 1:length(datavec)]
    plot_hourly_capfactors(data, year)
end

# Solve the greenfield model for varying data years without storage
outvecbio   = Vector{Model}(undef, 5)
outvecnobio = Vector{Model}(undef, 5)
icap_dict["Battery - Flow"]    = 0
icap_dict["Battery - Storage"] = 0
styleprintln("Solving the Greenfield model without storage for different data years")
Threads.@threads for i in 1:(length(outvecbio) + length(outvecnobio))
    lock(lk)
        threadprint(i, Threads.threadid())
    unlock(lk)
    if i <= length(outvecbio)
        global outvecbio[i] = solve_system_model(datavec[i], LoadshareBioConfig(1.), silent=true)
    else
        n = i - length(outvecbio)
        global outvecnobio[n] = solve_system_model(datavec[n], LoadshareNoBioConfig(1.), silent=true)
    end
end
println("")

# Create vectors for storing model outputs
bioresults   = deepcopy(objective_value.(outvecbio))
nobioresults = deepcopy(objective_value.(outvecnobio))

# Now solve the greenfield model for varying data years with storage
icap_dict["Battery - Flow"]    = 1e9
icap_dict["Battery - Storage"] = 1e9

# styleprintln("Solving the Greenfield model with storage for different data years")
Threads.@threads for i in 1:(length(outvecbio) + length(outvecnobio))
    lock(lk)
        threadprint(i, Threads.threadid())
    unlock(lk)
    if i <= length(outvecbio)
        global outvecbio[i] = solve_system_model(datavec[i], LoadshareBioConfig(1.), silent=true)
    else
        n = i - length(outvecbio)
        global outvecnobio[n] = solve_system_model(datavec[n], LoadshareNoBioConfig(1.), silent=true)
    end
end
println("")

# Create figure to illustrate relative system costs associated with different data years
plot_systemcosts(bioresults, outvecbio, "bio_datayears", xvals=[string(x) for _ in 1:2, x in 2015:2019][:], xlabel="Data year")
plot_systemcosts(nobioresults, outvecnobio, "nobio_datayears", xvals=[string(x) for _ in 1:2, x in 2015:2019][:], xlabel="Data year")

# Save output from model simulations
# for (i, model) in enumerate(outvec)
#     global scenario_df[:, "greenfield2 (" * string(2014 + i) * ")"] = scenarioreport(model, baseline_out)
# end

# Write experiment results to CSV file and print the associated dataframe
CSV.write("../output/scenarios.csv", scenario_df)
show(scenario_df)