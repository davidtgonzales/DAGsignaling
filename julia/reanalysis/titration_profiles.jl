# load packages
using DataFrames
using CSV
using JLD
using Plots

# include functions
include("../functions/simulate_models.jl")
include("../functions/mle_functions.jl")
include("../functions/extract_data.jl")
include("../functions/plotting_functions.jl")
include("../functions/profile_likelihoods.jl")

# short traces laser power titration
LP_list = [0,0.1,0.2,0.5,1,5,10,20,40]
LP_label = ["00","01","02","05","1","5","10","20","40"]
labels = ["0% LP" "0.1" "0.2" "0.5" "1" "5" "10" "20" "40"]

# long traces laser power titration
LP_list_long = [10,40]
LP_label_long = ["10","40"]

# load normalized data
t_mean = load(string("./data/reanalysis/t_mean.jld"))["t_mean"]
c1_norm = load(string("./data/reanalysis/c1_norm.jld"))["c1_norm"]
t_mean_long = load(string("./data/reanalysis/t_mean_long.jld"))["t_mean_long"]
c1_norm_long = load(string("./data/reanalysis/c1_norm_long.jld"))["c1_norm_long"]

# convert to uM units
SAG_init = (3123407.79958640)/6.022E23*1E15*1000000/3053.819
SOG_init = (3181009.45102653)/6.022E23*1E15*1000000/3053.819
DOG_init = (3143182.74585179)/6.022E23*1E15*1000000/3053.819
DAG_init = [SAG_init,SOG_init,DOG_init]
SAG_plat = (780680.993186633)/6.022E23*1E15*1000000/3053.819
SOG_plat = (2437584.95429104)/6.022E23*1E15*1000000/3053.819
DOG_plat = (2308565.59003329)/6.022E23*1E15*1000000/3053.819
DAG_plat = [SAG_plat,SOG_plat,DOG_plat]
# DAG_plat = [0,0,0] # uncomment to rerun analysis without correction

c1_uM = zeros(3,30,9)
c1_uM[1,:,:] = c1_norm[1,:,:].*(SAG_init)
c1_uM[2,:,:] = c1_norm[2,:,:].*(SOG_init)
c1_uM[3,:,:] = c1_norm[4,:,:].*(DOG_init)

c1_uM_long = zeros(3,120,2)
c1_uM_long[1,:,:] = c1_norm_long[1,:,:].*(SAG_init)
c1_uM_long[2,:,:] = c1_norm_long[2,:,:].*(SOG_init)
c1_uM_long[3,:,:] = c1_norm_long[4,:,:].*(DOG_init)

# measured DAG uncaging
SAG = [0,3.94E5,6.93E5,9.79E5,1.15E6,1.37E6,1.56E6,1.93E6,2.3E6]./6.022E23*1E15*1000000/3053.819
SOG = [0,1.75E5,3.94E5,5E5,6.39E5,1.16E6,1.38E6,2.12E6,2.96E6]./6.022E23*1E15*1000000/3053.819
DOG = [0,3.34E5,6.27E5,7.68E5,1.12E6,1.44E6,1.78E6,2.34E6,2.96E6]./6.022E23*1E15*1000000/3053.819
DAG = [SAG, SOG, DOG]
SAG_long = [1.56E6,2.3E6]./6.022E23*1E15*1000000/3053.819
SOG_long = [1.38E6,2.96E6]./6.022E23*1E15*1000000/3053.819
DOG_long = [1.78E6,2.96E6]./6.022E23*1E15*1000000/3053.819
DAG_long = [SAG_long,SOG_long,DOG_long]

mle_collect = load("./out/reanalysis/mle_collect.jld")["mle_collect"]
logp_collect = load("./out/reanalysis/logp_collect.jld")["logp_collect"]
p_mat = load("./out/reanalysis/p_mat.jld")["p_mat"]

logp_mle =zeros(3,4)
for i in 1:3
    logp_multi = logp_collect[i,:,:]
    mle_multi = mle_collect[i,:]
    min_idx = collect(getindex.(argmin(mle_multi,dims=1),1))
    logp_mle[i,:] = logp_multi[min_idx,:]
end

dosetimes = [14]

# get profiles for SAG, SOG, DOG
n_params = 4
pj_collect = zeros(3,n_params,11)
pf_collect = zeros(3,n_params,11,n_params-1)
pl_collect = zeros(3,n_params,11)
for idx in 1:3
    fep = logp_mle[idx,:]
    for fdx in 1:n_params
        pj_arr, pf_arr, pl_arr = get_pl_titration(fdx,fep,t_mean[idx,:,:],c1_uM[idx,:,:].-DAG_plat[idx],t_mean_long[idx,:,:],c1_uM_long[idx,:,:].-DAG_plat[idx],DAG[idx],DAG_long[idx],dosetimes)
        pj_collect[idx,fdx,:] = pj_arr
        pf_collect[idx,fdx,:,:] = pf_arr
        pl_collect[idx,fdx,:] = pl_arr
    end
end


# get confidence intervals
ci_collect = zeros(3,n_params,2)
for i in 1:3
    ci_collect[i,:,:] = get_ci(1, n_params-1, pl_collect[i,:,:], pj_collect[i,:,:])
end
display(ci_collect)

for i in 1:3
    DAG_labels = ["SAG","SOG","DOG"]
    filename = string("./plot/reanalysis/profiles_titration_",DAG_labels[i],".pdf")
    plot_profiles_titration(pj_collect[i,:,:], pl_collect[i,:,:], minimum(mle_collect[i,:]), exp.(logp_mle[i,:]), filename)
end