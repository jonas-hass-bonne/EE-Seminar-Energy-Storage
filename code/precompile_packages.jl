# Load packages used in the project
using JuMP, HiGHS           # Modelling packages
using Plots, StatsPlots     # Plotting packages
using DataFrames, CSV, XLSX # Data wrangling packages
using Revise

# Include custom functions and track changes
includet("modelsetup.jl")
includet("dataloading.jl")
using .ModelSetup, .DataLoading  

# Reduce startup of plotting functions
p = plot(rand(2,2))
display(p)

# Reduce startup of data-loading function
df_h = loadhourlyOPSdata(2018)
df_c = loadcapacityOPSdata(2018)
df_cf = loadcapfactorOPSdata(2018)
sheet_c = loadcoststructure()
sheet_b = loadbatterydata()

# Reduce startup of JuMP model solving
q_vec   = [df_c[6,2], df_c[7,2], df_c[5,2], df_c[2,2], df_c[3,2], df_c[1,2] + df_c[4,2]]    # Vector of generation capacities
c_vec   = Vector{Float64}(sheet_c[3:end, 9])  # Vector of marginal costs
fom_vec = Vector{Float64}(sheet_c[3:end, 3])  # Vector of fixed operating and maintenance costs
L_vec   = Vector{Float64}(df_h[:,2])         # Vector of load at different hours
u = maximum(c_vec) + 10  # Marginal WTP to avoid load-shedding

# Initialise matrices with hourly capacity constraints
gamma_mat = reshape(df_cf[:,4], (1,:)) # Initialise matrix of gamma values with intermittent generator values
gamma_mat = [gamma_mat; reshape(df_cf[:,3], (1,:))]   
gamma_mat = [gamma_mat; reshape(df_cf[:,2], (1,:))]
while size(gamma_mat, 1) < length(q_vec)    # Update gamma matrix until all generators have gamma values
    global gamma_mat = [gamma_mat; reshape([1 for i in 1:length(L_vec)], (1,:))]  # Update gamma matrix with 1's for dispatchable generators
end
qh_mat = Array{Float64}(undef, length(q_vec), length(L_vec))            # Initialise matrix of hourly generation capacities
for i in 1:size(qh_mat,1)
    global qh_mat[i,:] = q_vec[i] * gamma_mat[i,:]     # Update matrix by multiplying capacity with gamma value in each hour
end

# Create a SystemData object with the previously defined data
init_data = SystemData([:WindOffshore, :WindOnshore, :Solar, :Coal, :NaturalGas, :Bio],
                    q_vec, qh_mat, c_vec, fom_vec, L_vec, u)
m_out = solve_system_model(init_data)