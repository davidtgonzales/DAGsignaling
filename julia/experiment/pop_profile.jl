# load packages
using DataFrames
using CSV
using JLD

# include functions
include("../functions/simulate_models.jl")
include("../functions/mle_functions.jl")
include("../functions/profile_likelihoods.jl")
include("../functions/extract_data.jl")

############################################
# Profile likelihoods on experimental data #
############################################
# load data
idx = parse(Int, ARGS[1])
t_data, c1_data, c1, cells, c1_recruit, dosetimes, set_id = load_dataset(idx)

# initialize parameters and load MLE results
r_max = 1
p_init = load(string("./out/experiment/p_init_",set_id,".jld"))["p_init"]
r_params = load(string("./out/experiment/r_params_",set_id,".jld"))["r_params"]
f_params = load(string("./out/experiment/f_params_",set_id,".jld"))["f_params"]
obj = load(string("./out/experiment/obj_",set_id,".jld"))["obj"]

# get profiles
fdx = parse(Int, ARGS[2])
pl_arr, pj_arr, pr_arr, pf_arr = get_fixedpl(fdx,f_params[end,:],cells,t_data[:,1:cells],c1_data[:,1:cells],0,r_max,c1_recruit[1:cells],dosetimes,p_init)

# save results
save(string("./out/experiment/pl_",set_id,"_",string(fdx),".jld"), "pl_arr", pl_arr)
save(string("./out/experiment/pj_",set_id,"_",string(fdx),".jld"), "pj_arr", pj_arr)
save(string("./out/experiment/pr_",set_id,"_",string(fdx),".jld"), "pr_arr", pr_arr)
save(string("./out/experiment/pf_",set_id,"_",string(fdx),".jld"), "pf_arr", pf_arr)