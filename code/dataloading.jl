""" This module defines data structures and functions defining the
linear programming energy system models used in the project"""
module DataLoading
using DataFrames, CSV, XLSX 
export loadhourlyOPSdata, loadcapacityOPSdata, loadcapfactorOPSdata, loadcoststructure, loadbatterydata
export loadIEAdata, loadcapprojections

# Define fucntion to load the hourly Open Power System data
function loadhourlyOPSdata(year::String)
    input_data = CSV.read("..\\input\\time_series_60min_singleindex.csv", DataFrame)
    df_hourly = select(input_data, 2, r"DK")[:,1:11]
    # Create a mask to select data from the desired year
    mask = Vector{Bool}()
    for row in df_hourly[:,1]
        if occursin(year, row)
            push!(mask, true)
        else
            push!(mask, false)
        end
    end
    return df_hourly[mask,:]
end

# Define convenience method to convert number to string
loadhourlyOPSdata(year::Number) = loadhourlyOPSdata(string(Int(year))::String)

# Define function to load the Open Power System aggregated capacity data
function loadcapacityOPSdata(year::Integer)
    input_data = CSV.read("..\\input\\national_generation_capacity_stacked_filtered.csv", DataFrame)
    df_cap = input_data[input_data[:,:year] .== 2018,:] # Select only data from 2018
    df_cap = df_cap[df_cap[:,:source] .== "ENTSO-E Power Statistics",:] # Select only data from ENTSO-E
    df_cap_lower = df_cap[df_cap[:,:energy_source_level_3] .== 1, [:technology, :capacity]] # Select data on the most disaggregated level
    
    # Further disaggregate wind into onshore and offshore
    df_cap_wind = append!(df_cap[df_cap[:,:technology] .== "Onshore", [:technology, :capacity]], df_cap[df_cap[:,:technology] .== "Offshore", [:technology, :capacity]])

    # Combine the datasets and return capacities for the desired plants
    return df = append!(df_cap_lower[[1, 5, 10, 14, 17], :], df_cap_wind)
end

# Define convenience method to convert general number to integer
loadcapacityOPSdata(year::Number) = loadcapacityOPSdata(Int(year)::Integer)

# Define function to load the Open Power System capacity factor data
function loadcapfactorOPSdata(year::String)
    input_data = CSV.read("../input/ninja_pv_wind_profiles_singleindex_filtered.csv", DataFrame)
    df_capfac = input_data[:,[1,2,4,5]]
    # Create mask to select data form the desired year
    mask = Vector{Bool}()
    for row in df_capfac[:,1]
        if occursin(year, row)
            push!(mask, true)
        else
            push!(mask, false)
        end
    end
    return df_capfac[mask,:]
end

# Define convenience method to convert number to string
loadcapfactorOPSdata(year::Number) = loadcapfactorOPSdata(string(Int(year))::String)

# Define function to load cost structure from the Danish Energy Agency
function loadcoststructure(sheet::String)
    input_sheet = XLSX.readdata("../input/Data collection.xlsx", sheet, "A15:G25")
    input_sheet[1,1] = "Variable/Plant"
    input_sheet = input_sheet[Not(ismissing.(input_sheet[:,1])), :]
    return input_sheet
end

# Define convenience method to convert number to relevant sheet string
loadcoststructure(sheet::Number) = loadcoststructure("Simulering scenarie " * string(sheet))

# Define function to load data on lithium batteries from the Danish Energy Agency
function loadbatterydata(sheet::String)
    input_sheet = XLSX.readdata("../input/Data collection.xlsx", sheet, "A33:B51")
    input_sheet = input_sheet[Not(ismissing.(input_sheet[:,1])),:]
    return input_sheet
end

# Define convenience method to convert number to relevant sheet string
loadbatterydata(sheet::Number) = loadbatterydata("Simulering scenarie " * string(sheet))

# Define function to load data from IEA on danish electricity generation
function loadIEAdata(;sheet::String="Electricity generation by sourc")
    input_sheet = XLSX.readdata("../input/IEA data.xlsx", sheet, "A5:H8")
    outvec = [input_sheet[4, i] for i in [2, 4, 5, 7, 8]]
    return outvec[[4, 5, 1, 2, 3]]
end

# Define function to load capacity projections from the Danish Energy Agency
function loadcapprojections(;sheet::String="Sheet1", year::Integer=2030)
    input_sheet = XLSX.readdata("../input/2030 projections.xlsx", sheet, "B21:I27")
    input_2022  = input_sheet[1:3, 2:end]
    input_2030  = input_sheet[5:7, 2:end-1]
    if year == 2030
        outvec = input_2030[2,[1,4,3,6,5,2]]
    elseif year == 2022
        outvec = input_2022[2,[1,4,3,6,7,5,2]]
    end
    return outvec
end

end