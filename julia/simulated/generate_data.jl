# load packages
using DataFrames
using CSV
using JLD

# include functions
include("../functions/simulate_models.jl")

######################
# model without kout #
######################
# initialize parameters
cells = 200
kin = 0.098
kmet = 0.01823
Kd = 0.866
params = [kin,kmet,Kd]
c1_init = rand(Normal(2.5,0.5),cells) # initial C1 concentrations
sd = repeat([0.1],cells) # measurement noise
dag = rand(Normal(1.2,0.1),cells) # uncaged DAGs
dosetimes = [15]

# generate data
t_data = repeat(collect(0:2:400), outer = [1,cells])
c1_data = simulate_data(t_data, dag, c1_init, params, dosetimes, sd)
c1_recruit = transpose(mean(c1_data[1:5,:],dims=1)-minimum(c1_data,dims=1)) # difference of the mean of 1st five datapoints before uncaging and minimum C1 after uncaging

# save data
save("./data/simulated/cells_t_data.jld","t_data",t_data)
save("./data/simulated/cells_c1_data.jld","c1_data",c1_data)
save("./data/simulated/cells_dag.jld","dag",dag)

###################
# model with kout #
###################
# initialize parameters
cells = 200
kin = 0.065
kout = 0.12
kmet = 0.056
Kd = 0.28
params = [kin,kout,kmet,Kd]
c1_init = rand(Normal(2.5,0.5),cells) # initial C1 concentrations
sd = repeat([0.1],cells) # measurement noise
dag = rand(Normal(1.2,0.1),cells) # uncaged DAGs
dosetimes = [15]

# generate data
t_data = repeat(collect(0:2:400), outer = [1,cells])
c1_data = simulate_data(t_data, dag, c1_init, params, dosetimes, sd)
c1_recruit = transpose(mean(c1_data[1:5,:],dims=1)-minimum(c1_data,dims=1)) # difference of the mean of 1st five datapoints before uncaging and minimum C1 after uncaging

# save data
save("./data/simulated/cells_t_data_kout.jld","t_data",t_data)
save("./data/simulated/cells_c1_data_kout.jld","c1_data",c1_data)
save("./data/simulated/cells_dag_kout.jld","dag",dag)
