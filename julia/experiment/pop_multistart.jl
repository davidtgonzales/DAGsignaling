# load packages
using DataFrames
using CSV
using JLD

# include functions
include("../functions/simulate_models.jl")
include("../functions/mle_functions.jl")
include("../functions/extract_data.jl")

# load data
idx = parse(Int, ARGS[1])
run = parse(Int, ARGS[2])
t_data, c1_data, c1, cells, c1_recruit, dosetimes, set_id = load_dataset(idx)

# initialize parameters
n = 20 # number of trials
p_mat = zeros(n,4)
d1 = Uniform(0,1)
d2 = Uniform(0,1)
d3 = Uniform(0,20)
d4 = Uniform(0,0.5)
p_mat[:,1] = rand(d1,n)
p_mat[:,2] = rand(d2,n)
p_mat[:,3] = rand(d3,n)
p_mat[:,4] = rand(d4,n)
# for 1st initial parameters
if run == 1
    p_mat[1,:] = [0.1,0.01,1,0.1] # kin,kmet,Kd,sd
end

r_max = 1
mle_multi = zeros(n)
logp_multi = zeros(n,4)
for i in 1:n
    display(i)
    p_init = p_mat[i,:]
    res_r = optimize(r->objfcn_r(r,t_data[:,1:cells],c1_data[:,1:cells],cells,c1_recruit[1:cells],dosetimes,p_init),0.0,r_max,abs_tol=0.01,rel_tol=0.001)
    r_opt = Optim.minimizer(res_r)
    r_params, f_params, obj = optimized_r(r_opt, t_data[:,1:cells], c1_data[:,1:cells], cells, c1_recruit[1:cells], dosetimes, p_init)
    mle_multi[i] = obj[end]
    logp_multi[i,:] = f_params[end,:]
end

# save results
set = string(set_id,"_",string(run))
save(string("./out/experiment/p_mat_",set,".jld"), "p_mat", p_mat)
save(string("./out/experiment/mle_multi_",set,".jld"), "mle_multi", mle_multi)
save(string("./out/experiment/logp_multi_",set,".jld"), "logp_multi", logp_multi)