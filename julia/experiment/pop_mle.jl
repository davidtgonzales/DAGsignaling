# load packages
using DataFrames
using CSV
using JLD

# include functions
include("../functions/simulate_models.jl")
include("../functions/mle_functions.jl")
include("../functions/extract_data.jl")

############################
# MLE on experimental data #
############################
# index of each DAG in dataset
# 2 (MS46) cgDMG
# 5 (MS12) cgSAG
# 6 (MS47) cgDArG
# 7 (MS45) cgSLG
# 9 (MS44) cgOSG
# 11 (MS6) cgSOG

# load data
idx = parse(Int, ARGS[1])
t_data, c1_data, c1, cells, c1_recruit, dosetimes, set_id = load_dataset(idx)
display(cells)
display(set_id)

# load best initial conditions from multistart results
p_mat1 = load(string("./out/experiment/p_mat_",set_id,"_1.jld"))["p_mat"]
mle_multi1 = load(string("./out/experiment/mle_multi_",set_id,"_1.jld"))["mle_multi"]
logp_multi1 = load(string("./out/experiment/logp_multi_",set_id,"_1.jld"))["logp_multi"]
p_mat2 = load(string("./out/experiment/p_mat_",set_id,"_2.jld"))["p_mat"]
mle_multi2 = load(string("./out/experiment/mle_multi_",set_id,"_2.jld"))["mle_multi"]
logp_multi2 = load(string("./out/experiment/logp_multi_",set_id,"_2.jld"))["logp_multi"]
p_mat3 = load(string("./out/experiment/p_mat_",set_id,"_3.jld"))["p_mat"]
mle_multi3 = load(string("./out/experiment/mle_multi_",set_id,"_3.jld"))["mle_multi"]
logp_multi3 = load(string("./out/experiment/logp_multi_",set_id,"_3.jld"))["logp_multi"]
p_mat4 = load(string("./out/experiment/p_mat_",set_id,"_4.jld"))["p_mat"]
mle_multi4 = load(string("./out/experiment/mle_multi_",set_id,"_4.jld"))["mle_multi"]
logp_multi4 = load(string("./out/experiment/logp_multi_",set_id,"_4.jld"))["logp_multi"]
p_mat5 = load(string("./out/experiment/p_mat_",set_id,"_5.jld"))["p_mat"]
mle_multi5 = load(string("./out/experiment/mle_multi_",set_id,"_5.jld"))["mle_multi"]
logp_multi5 = load(string("./out/experiment/logp_multi_",set_id,"_5.jld"))["logp_multi"]
p_mat = vcat(p_mat1,p_mat2,p_mat3,p_mat4,p_mat5)
mle_multi = vcat(mle_multi1,mle_multi2,mle_multi3, mle_multi4,mle_multi5)
logp_multi = vcat(logp_multi1,logp_multi2,logp_multi3,logp_multi4,logp_multi5)
# sort results
mle_sort = mle_multi[sortperm(mle_multi)]
logp_sort = logp_multi[sortperm(mle_multi),:]
p_mat_sort = p_mat[sortperm(mle_multi),:]

# initialize parameters from best result of multistart
p_init = exp.(logp_sort[1,:]) # kin,kmet,Kd,sd
r_max = 1

# get MLE
res_r = optimize(r->objfcn_r(r,t_data[:,1:cells],c1_data[:,1:cells],cells,c1_recruit[1:cells],dosetimes,p_init),0.0,r_max,abs_tol=0.01,rel_tol=0.001)
r_opt = Optim.minimizer(res_r)
r_params, f_params, obj = optimized_r(r_opt, t_data[:,1:cells], c1_data[:,1:cells], cells, c1_recruit[1:cells], dosetimes, p_init)

# save results
save(string("./out/experiment/p_init_",set_id,".jld"), "p_init", p_init)
save(string("./out/experiment/r_params_",set_id,".jld"), "r_params", r_params)
save(string("./out/experiment/f_params_",set_id,".jld"), "f_params", f_params)
save(string("./out/experiment/obj_",set_id,".jld"), "obj", obj)
save(string("./out/experiment/r_",set_id,".jld"), "r_opt", r_opt)