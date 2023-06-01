""" This module defines functions to plot relevant output from
the energy system model simulations"""
module ModelPlots 
using Plots, StatsPlots, JuMP, DataFrames
export plot_hourly_gen, plot_hourly_storage, plot_capacities, plot_generation
export plot_storagecosts, plot_systemcosts, plot_hourly_capfactors

default(fontfamily="computer modern")   # Set default font family
scalefontsizes()    # Make sure font scaling is reset
# scalefontsizes(.9)  # Set font sizes as fraction of defaults

# Create a function to plot slices of hourly generation for different plant types
function plot_hourly_gen(data, output::Model, name::String; dpi::Integer=1000, gencolors=nothing)
    # Define a matrix containing hourly generation for each plant type in columns
    genout = permutedims(value.(output[:E])) 

    # Create a list of labels for the plot
    genlabels = permutedims(string.(data.plants))

    # Generate a dictionary of colors for each generator
    if gencolors == nothing
        gencolors = Dict([("WindOffshore", :blue), ("WindOnshore", :cyan), ("Solar", :yellow),
                    ("Coal", :black), ("NaturalGas", :grey), ("Bio", :green)])
    end

    # Remove plants with zero generation
    i = 1
    while i <= size(genout, 2)
        if genout[:,i] == fill(0., size(genout, 1))
            genout = genout[:, (1:end) .!= i]
            delete!(gencolors, genlabels[i])
            genlabels = genlabels[:, (1:end) .!= i]
        else
            i += 1
        end
    end

    # Compute cumulative generation in each hour
    for (i, col) in enumerate(eachcol(genout[:,2:end]))
        genout[:, i + 1] += genout[:, i]
    end

    # Generate a matrix of fillranges
    genfill = [fill(0., size(genout, 1)) genout[:, 1:end - 1]]

    # Create plots for a number of time periods during the year
    winter  = plot(title="Winter", legendposition=:none, ylabel="MW")
    spring  = plot(title="Spring", legendposition=:none)
    summer  = plot(title="Summer", bottommargin=(16, :pt), legendposition=(.3,-.4), legendcolumn=-1, legendforegroundcolor=nothing, ylabel="MW", xlabel="Hours")
    autumn  = plot(title="Fall",   bottommargin=(16, :pt), legendposition=:none, xlabel="Hours")
    for (p,i) in ((winter, 1), (spring, 2160), (summer, 4344), (autumn, 6552))
        for (series, fill, name) in zip(eachcol(genout), eachcol(genfill), eachcol(genlabels))
            plot!(p, i:i+336, series[i:i+336], fillrange=fill[i:i+336], label=name[1], linewidth=0, seriescolor=gencolors[name[1]])
        end
        plot!(p, i:i+336, data.load[i:i+336], label="Demand", seriescolor=:red)
    end
    outplot = plot(winter, spring, summer, autumn, layout=(2,2), dpi=dpi, link=:y, formatter=:plain)
    outstring = "../output/" * name * "_gen"
    png(outplot, outstring)
    return 
end

# Create a function to plot residual demand and storage flows
function plot_hourly_storage(data, output::Model, name::String; dpi::Integer=1000, highlight::Bool=false)
    storage_in  = -value.(output[:Bi])
    storage_out = value.(output[:Bo])
    storage_flowcap_hi = [value(output[:Bi_up]) for _ in 1:length(data.load)]
    storage_flowcap_lo = [-value(output[:Bo_up]) for _ in 1:length(data.load)]
    if has_upper_bound(output[:E][1,1]) # Check if generation has an upper bound
        resdemand = data.load - sum([upper_bound(output[:E][i,h]) for i in 1:3, h in 1:8760], dims=1)[:]
    else    # Assume greenfield and query the bound variable instead
        resdemand = data.load - sum(value.(output[:E_up][1:3,:]), dims=1)[:]
    end
    plot_matrix = sortslices([resdemand storage_in storage_out], dims=1, rev=true)
    plot(plot_matrix[:,1], labels="Residual demand", dpi=dpi, ylabel="MW", xlabel="Hours", formatter=:plain,
         legendposition=(-.05, -.15), legendforegroundcolor=nothing, legendbackgroundcolor=nothing, bottom_margin=(16,:pt))
    if storage_in != fill(0., length(storage_in))
        plot!(plot_matrix[:,2:end], fillrange=0, labels=["Storage charging" "Storage discharging"], dpi=dpi, 
              seriestype=:steppre, linewidth=0, legendcolumn=-1)
        plot!([storage_flowcap_hi storage_flowcap_lo], labels=["Power capacity" ""], 
              linestyle=:dash, dpi=dpi, seriescolor=[:grey :grey])
    end
    outstring = "../output/" * name * "_resdemand"
    png(outstring)

    if highlight
        for (n, i) in enumerate([1, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
            curplot = plot()
            plot!(curplot, i:i+336, plot_matrix[i:i+336,1], labels="Residual demand", dpi=dpi, ylabel="MW", xlabel="Hours")
            plot!(curplot, i:i+336, plot_matrix[i:i+336,2:end], fillrange=0, labels=[ "Storage inflow" "Storage outflow"], dpi=dpi, seriestype=:steppre, linewidth=0)
            outstring = "../output/" * name * "_resdemand_" * string(n)
            png(curplot, outstring)
        end
    end
    return
end

# Helper function for placing piechart markers sensibly
function ann_pos(x)
    cumx = cumsum(x)
    midpoints = [x[1]/2; x[2:end]./2 .+ cumx[1:end-1]]
    [cos.(midpoints.*2π).*0.6 sin.(midpoints.*2pi).*0.6]
end

# Function to create pie chart for total generation capacity
function plot_capacities(data, output::Model, name::String; dpi::Integer=1000)
    # Check if capacities should be taken from data or model
    if all(keys(output.obj_dict) .!= :E_cap)
        capfracs = data.capacities./sum(data.capacities)
        caps = data.capacities
    else
        capfracs = value.(output[:E_cap])./sum(value.(output[:E_cap]))
        caps = value.(output[:E_cap])
    end

    # Define colors, legend labels, annotation and annotation positions
    colors = [:blue, :cyan, :yellow, :black, :grey, :green]
    plants = string.(data.plants) .* "\n" .* string.(round.(capfracs*100, digits=1)) .* "%"
    ann = string.(round.(Int, caps[caps .> 0])) .* " MW"
    pos = ann_pos(capfracs[capfracs .> 0])

    # Remove any plants with zero installed capacity
    i = 1
    while i <= length(caps)
        if caps[i] == 0
            deleteat!(caps, i)
            deleteat!(colors, i)
            deleteat!(plants, i)
        else
            i += 1
        end
    end

    # Produce the initial piechart with legend
    pie(plants, caps, legend_position=:topleft, legendforegroundcolor=nothing, legendbackgroundcolor=nothing, seriescolor=colors, dpi=dpi)
    
    # Annotate the piechart
    for annot in eachrow([pos ann])
        annotate!(Tuple(annot))
    end
    
    # Add markers for the annotations
    adjustment = 2/17
    for (i,text) in enumerate(ann) 
        num_boxes = round(length(text)*5/8, RoundUp) - 1
        for adj in (-adjustment*(num_boxes/2)):adjustment:(adjustment*(num_boxes/2))
            scatter!([pos[i,1] + adj], [pos[i,2]], marker=(:rect, 10.03, .75, :white, stroke(0)), label="")
        end
    end
    
    # Save the produced piechart
    outstring = "../output/" * name * "_caps" 
    png(outstring)
    return
end

# Define a dispatch of the capacity pie chart for data projections
function plot_capacities(data, input::AbstractVector, name::String; dpi::Integer=1000)
    # Turn capacity data into capacity fractions
    capfracs = input./sum(input)
    caps = input

    # Define colors, legend labels, annotation and annotation positions
    if length(input) == 6
        colors = [:red, :blue, :cyan, :yellow, :grey, :green]
        plants = [["Other\n" * string(round(capfracs[1]*100, digits=1)) * "%"] ; 
                    string.(data.plants[1:end .!= 4]) .* "\n" .* string.(round.(capfracs[2:end].*100, digits=1)) .* "%"]
    elseif length(input) == 7
        colors = [:red, :blue, :cyan, :yellow, :black, :grey, :green]
        plants = [["Other\n" * string(round(capfracs[1]*100, digits=1)) * "%"] ;
                    string.(data.plants) .* "\n" .* string.(round.(capfracs[2:end].*100, digits=1)) .* "%"]
    end
    ann = string.(round.(Int, caps[caps .> 0])) .* " MW"
    pos = ann_pos(capfracs[capfracs .> 0])

    # Remove any plants with zero installed capacity
    i = 1
    while i <= length(caps)
        if caps[i] == 0
            deleteat!(caps, i)
            deleteat!(colors, i)
            deleteat!(plants, i)
        else
            i += 1
        end
    end

    # Produce the initial piechart with legend
    pie(plants, caps, legend_position=(.08, .9), legendforegroundcolor=nothing, legendbackgroundcolor=nothing, seriescolor=colors, dpi=dpi)
    
    # Annotate the piechart
    for annot in eachrow([pos ann])
        annotate!(Tuple(annot))
    end
    
    # Add markers for the annotations
    adjustment = 2/17
    for (i,text) in enumerate(ann) 
        num_boxes = round(length(text)*5/8, RoundUp) - 1
        for adj in (-adjustment*(num_boxes/2)):adjustment:(adjustment*(num_boxes/2))
            scatter!([pos[i,1] + adj], [pos[i,2]], marker=(:rect, 10, .75, :white, stroke(0)), label="")
        end
    end
    
    # Save the produced piechart
    outstring = "../output/" * name * "_caps" 
    png(outstring)
    return
end

function plot_generation(data, output::Model, name::String; dpi::Integer=1000)
    # Define fractional generation, generation, colors, legend labels, annotations and annotation positions
    genfracs = (sum(value.(output[:E]), dims=2)./sum(value.(output[:E])))[:]
    gen = sum(value.(output[:E]), dims=2)[:]
    colors = [:blue, :cyan, :yellow, :black, :grey, :green] 
    plants = string.(data.plants) .* "\n" .* string.(round.(genfracs*100, digits=1)) .* "%"
    ann = string.(round.(Int, gen[gen .> 0]./1000)) .* " GWh"
    pos = ann_pos(genfracs[genfracs .> 0])

    # Remove any plants with zero generation
    i = 1
    while i <= length(gen)
        if gen[i] == 0
            deleteat!(gen, i)
            deleteat!(colors, i)
            deleteat!(plants, i)
        else
            i += 1
        end
    end
    
    # Produce the initial piechart with legend
    pie(plants, gen, legend_position=:topleft, legendforegroundcolor=nothing, legendbackgroundcolor=nothing, seriescolor=colors, dpi=dpi)
    
    # Annotate the piechart
    for ann in eachrow([pos ann])
        annotate!(Tuple(ann))
    end

    # Add markers for the annotations
    adjustment = 2/17
    for (i,text) in enumerate(ann) 
        num_boxes = round(length(text)*5/8, RoundUp) - 1
        for adj in (-adjustment*(num_boxes/2)):adjustment:(adjustment*(num_boxes/2))
            scatter!([pos[i,1] + adj], [pos[i,2]], marker=(:rect, 10.03, .75, :white, stroke(0)), label="")
        end
    end

    # Save the produced piechart
    outstring="../output/" * name * "_totalgen"
    png(outstring)
    return
end

# Define a dispatch of the generation pie chart function for IEA data
function plot_generation(data, input::AbstractVector, name::String; dpi::Integer=1000)
    genfracs = input./sum(input)
    gen = input
    colors = [:blue, :yellow, :black, :grey, :green]

    plants = ["Wind\n" * string(round(genfracs[1]*100, digits=1)) * "%";
                string.(data.plants[3:end]) .* "\n" .* string.(round.(genfracs[2:end]*100, digits=1)) .* "%"]
    ann = string.(round.(Int, gen[gen .> 0])) .* " GWh"
    pos = ann_pos(genfracs[genfracs .> 0])

    # Produce the initial piechart with legend
    pie(plants, gen, legend_position=:topleft, seriescolor=colors, dpi=dpi)
    
    # Annotate the piechart
    for ann in eachrow([pos ann])
        annotate!(Tuple(ann))
    end
    
    # Add markers for the annotations
    adjustment = 2/17
    for (i,text) in enumerate(ann) 
        num_boxes = round(length(text)*5/8, RoundUp) - 1
        for adj in (-adjustment*(num_boxes/2)):adjustment:(adjustment*(num_boxes/2))
            scatter!([pos[i,1] + adj], [pos[i,2]], marker=(:rect, 10.03, .75, :white, stroke(0)), label="")
        end
    end

    # Save the produced piechart
    outstring="../output/" * name * "_totalgen"
    png(outstring)
    return
end

function plot_storagecosts(results::Vector{Model}, reference::Real, name::String; dpi::Integer=1000)
    # Unpack the DataFrames into separate vectors
    syscosts      = objective_value.(results) ./ reference .* 100
    storagepower  = value.(variable_by_name.(results, "Bi_up"))
    storageenergy = value.(variable_by_name.(results, "St_up"))

    # Create series and groupings for the grouped bar chart
    pevec    = [x[i] for x in [storagepower, storageenergy], i in eachindex(storagepower)][:]
    grouping = [x for x in ["Storage power (MW, left)", "Storage energy (MWh, left)"], _ in eachindex(storagepower)][:]
    xvals    = [string(x) * "%" for _ in 1:2, x in -50:10:150][:]

    # Create the grouped bar chart for storage capacities
    bar(xvals, pevec, group=grouping, legend_position=(.25, -.18), legendforegroundcolor=nothing, legendbackgroundcolor=nothing, 
        legendcolumn=-1, ylabel="MW/MWh", xlabel="Relative increase in storage cost", dpi=dpi, 
        formatter=:plain, bottom_margin=(40,:pt), size=(900,400), margin=(3,:mm))
    xticks!(0.5:20.5, [x for x in unique(xvals)])    # Add tickmarks for each x-value
    annotate!([(xvals[i], pevec[i - 1] + ylims()[2]/35, string(round(Int, pevec[i]/pevec[i-1]))) for i in eachindex(pevec)[begin+1:2:end]])
    
    # Add the scatter plot for relative system costs    
    scatter!(twinx(), [string(x) * "%" for x in -50:10:150], syscosts, dpi=dpi,
            label="Costs with storage (right)", leg=(.4, -.25), legendforegroundcolor=nothing, legendbackgroundcolor=nothing,
            color=:green, ylims=(minimum(syscosts) - 10, maximum(syscosts) + 10), ylabel="System costs relative to no storage")
    
    png("../output/" * name * "_storage_sensitivity")
    return
end

function plot_systemcosts(base::Vector{Float64}, storage::Vector{Model}, name::String; 
                          dpi::Integer=1000, xvals=[string(x) * "%" for _ in 1:2, x in 60:10:100][:],
                          xlabel="Share of renewable energy")
    # Unpack model vectors
    storagesyscosts = objective_value.(storage) ./ base .* 100
    storagepower    = value.(variable_by_name.(storage, "Bi_up"))
    storageenergy   = value.(variable_by_name.(storage, "St_up"))
    
    # Create series and groupings for the grouped bar chart
    pevec    = [x[i] for x in [storagepower, storageenergy], i in eachindex(storagepower)][:]
    grouping = [x for x in ["Storage power (MW, left)", "Storage energy (MWh, left)"], _ in eachindex(storagepower)][:]

    # Create the grouped bar chart for storage capacities
    bar(xvals, pevec, group=grouping, legend_position=(.25, -.15), legendforegroundcolor=nothing, legendbackgroundcolor=nothing, legendcolumn=2,
        ylabel="MW/MWh", xlabel=xlabel, dpi=dpi, formatter=:plain, bottom_margin=(32,:pt))
    xticks!(0.5:length(xvals)/2, [xvals[i] for i in 1:2:length(xvals)])    # Add tickmarks for each x-value
    if !all(pevec .== fill(0., length(pevec)))
        annotate!([(xvals[i], pevec[i - 1] + ylims()[2]/35, string(round(Int, pevec[i]/pevec[i-1]))) for i in 2:2:length(pevec)])
    end

    # Add the scatter plot for relative system costs    
    scatter!(twinx(), [xvals[i] for i in 1:2:length(xvals)], storagesyscosts, dpi=dpi,
            label="Costs with storage (right)", legend_position=(.4, -.23), legendforegroundcolor=nothing, 
            color=:green, legendbackgroundcolor=nothing,
            ylims=(minimum(storagesyscosts) - 10, maximum(storagesyscosts) + 10), ylabel="System costs relative to no storage")
            
    png("../output/" * name * "_sys_costs")
    return
end

function plot_hourly_capfactors(data, name::String; dpi::Integer=1000)
    # Unpack relevant data
    plants  = string.(reshape(data.plants[1:3], 1, :)) .* " (Left)"
    capfacs = transpose(data.capacityfactors)[:,1:3]
    load    = data.load
    colors  = [:blue :cyan :yellow]

    # Create plots for a number of time periods during the year
    winter  = plot(title="Winter", legendposition=:none, ylabel="Natural Conditions")
    spring  = plot(title="Spring", legendposition=:none)
    summer  = plot(title="Summer", bottommargin=(16,:pt), ylabel="Natural Conditions", xlabel="Hours",
                   legendposition=(.1,-.4), legendcolumn=-1, legendforegroundcolor=nothing)
    autumn  = plot(title="Fall",   bottommargin=(16,:pt), legendposition=:none, xlabel="Hours")
    for (p,i) in ((winter, 1), (spring, 2160), (summer, 4344), (autumn, 6552))
        plot!(p, i:i+336, capfacs[i:i+336, :], label=plants, seriescolor=colors, ylims=(0,1))
        if (p == spring) | (p == autumn)
            axislabel = "MW"
            if p == autumn
                leglabel = "Demand (Right)"
            else
                leglabel = ""
            end
        else
            axislabel = ""
            leglabel  = ""
        end
        plot!(twinx(p), i:i+336, load[i:i+336], label=leglabel, seriescolor=:red, 
              ylabel=axislabel, ylims=(0, Inf), legendposition=(.4, -.4), legendforegroundcolor=nothing)
    end
    outplot = plot(winter, spring, summer, autumn, layout=(2,2), dpi=dpi, link=:y, formatter=:plain)
    outstring = "../output/" * name * "_capfacs"
    png(outplot, outstring)
    return 
end

end