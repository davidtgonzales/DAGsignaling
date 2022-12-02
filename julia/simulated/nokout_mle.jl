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
t_data = load("./data/simulated/cells_t_data.jld")["t_data"]
c1_data = load("./data/simulated/cells_c1_data.jld")["c1_data"]
dag = load("./data/simulated/cells_dag.jld")["dag"]
c1_recruit = transpose(mean(c1_data[1:5,:],dims=1)-minimum(c1_data,dims=1))

# set number of cells
cells = parse(Int, ARGS[1]) # change for 10, 50, 100, 200 cells
set = string(string(cells),"cells")

# true values
kin = 0.098
kmet = 0.01823
Kd = 0.866
sd = 0.1

# initialize parameters
r_max = 1
dosetimes = [15]
p_init = [kin,kmet,Kd,sd]

# get MLE
res_r = optimize(r->objfcn_r(r,t_data[:,1:cells],c1_data[:,1:cells],cells,c1_recruit[1:cells],dosetimes,p_init),0.0,r_max,abs_tol=0.01,rel_tol=0.001)
r_opt = Optim.minimizer(res_r)
r_params, f_params, obj = optimized_r(r_opt, t_data[:,1:cells], c1_data[:,1:cells], cells, c1_recruit[1:cells], dosetimes, p_init)

# save results
save(string("./out/simulated/r_params_",set,".jld"), "r_params", r_params)
save(string("./out/simulated/f_params_",set,".jld"), "f_params", f_params)
save(string("./out/simulated/obj_",set,".jld"), "obj", obj)
save(string("./out/simulated/r_",set,".jld"), "r_opt", r_opt)