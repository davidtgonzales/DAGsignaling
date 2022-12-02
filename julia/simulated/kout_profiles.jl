# load packages
using DataFrames
using CSV
using JLD

# include functions
include("../functions/simulate_models.jl")
include("../functions/mle_functions.jl")
include("../functions/profile_likelihoods.jl")
include("../functions/plotting_functions.jl")

# load data
t_data = load("./data/simulated/cells_t_data_kout.jld")["t_data"]
c1_data = load("./data/simulated/cells_c1_data_kout.jld")["c1_data"]
dag = load("./data/simulated/cells_dag_kout.jld")["dag"]
c1_recruit = transpose(mean(c1_data[1:5,:],dims=1)-minimum(c1_data,dims=1))

# set number of cells
cells = parse(Int, ARGS[1]) # change for 10, 50, 100, 200 cells
set = string(string(cells),"cells")

# true values
kin = 0.065
kout = 0.12
kmet = 0.056
Kd = 0.28
sd = 0.1

# set parameter index
fdx = parse(Int, ARGS[2])

# get profiles
r_max = 1
p_init = [kin,kout,kmet,Kd,sd]
dosetimes = [15]
pl_arr, pj_arr, pr_arr, pf_arr = get_fixedpl_kout(fdx,log.(p_init),cells,t_data[:,1:cells],c1_data[:,1:cells],0,r_max,c1_recruit[1:cells],dosetimes,p_init)
save(string("./out/simulated/pl_",set,"_",string(fdx),"_kout.jld"), "pl_arr", pl_arr)
save(string("./out/simulated/pj_",set,"_",string(fdx),"_kout.jld"), "pj_arr", pj_arr)
save(string("./out/simulated/pr_",set,"_",string(fdx),"_kout.jld"), "pr_arr", pr_arr)
save(string("./out/simulated/pf_",set,"_",string(fdx),"_kout.jld"), "pf_arr", pf_arr)