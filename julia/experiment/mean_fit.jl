# load packages
using DataFrames
using CSV
using JLD

# include functions
include("../functions/simulate_models.jl")
include("../functions/mle_functions.jl")
include("../functions/extract_data.jl")
include("../functions/profile_likelihoods.jl")
include("../functions/plotting_functions.jl")

###############################################
# MLE on population mean of experimental data #
###############################################
# load data
# index of each DAG in dataset
# 2 (MS46) cgDMG
# 5 (MS12) cgSAG
# 6 (MS47) cgDArG
# 7 (MS45) cgSLG
# 9 (MS44) cgOSG
# 11 (MS6) cgSOG

t_data = zeros(120,6)
c1_data = zeros(120,6)
c1_init = zeros(6)
c1_recruit = zeros(6)
for (i,idx) in enumerate([2,9,11,7,5,6])
    t_datai, c1_datai, c1i, cells, c1_recruiti, dosetimes, set_id = load_dataset(idx)
    # get mean values
    t_data[:,i] = mean(t_datai,dims=2)
    c1_data[:,i] = mean(c1_datai,dims=2)
    c1_init[i] = mean(c1i)
    c1_recruit[i] = mean(c1_recruiti)
end

# load best initial conditions from multistart results
p_init = load(string("./out/experiment/p_init_multi_mean.jld"))["p_init"]
mle_multi = load(string("./out/experiment/mle_multi_mean.jld"))["mle_multi"]
logp_multi = load(string("./out/experiment/logp_multi_mean.jld"))["logp_multi"]

# initial values and average DAG concentrations
min_idx = collect(getindex.(argmin(mle_multi,dims=1),1))
# dag = [1.2106,0.6996,1.465,1.227,1.6086,1.4617]
dag = [1.824383581,1.054389975,2.207953554,1.849868644,2.424218508,2.202893915]

# get mle
dosetimes = [14]
cells = 6
mle = zeros(cells)
logp_mle = zeros((cells,4))
c1_sim = zeros(size(t_data)[1],cells)
for idx in 1:cells
    display(idx)
    logp0 = logp_multi[min_idx[idx],idx,:]
    mle[idx], logp_mle[idx,:] = opt_mle_mean(logp0,t_data[:,idx],c1_data[:,idx],dag[idx],dosetimes)
end
display(mle)
display(exp.(logp_mle))

# plot results
c1_fit = zeros(size(c1_data))
for i in 1:cells
    c1_fit[:,i] = simulate_data_single(t_data[:,i], dag[i], mean(c1_data[1:5,i]), exp.(logp_mle[i,1:3]), dosetimes, 0)
end

labels = reshape(["DMG", "OSG","SOG","SLG","SAG","DArG"],1,6)
filename = "./plot/experiment/mean_fit.pdf"
plot_fit_single_mean(cells, t_data, c1_data, c1_fit, dosetimes, labels, filename)

# get profiles
n_params = 4
pj_collect = zeros(cells,n_params,11)
pf_collect = zeros(cells,n_params,11,n_params-1)
pl_collect = zeros(cells,n_params,11)
for idx in 1:cells
    display(idx)
    fep = logp_mle[idx,:]
    for fdx in 1:n_params
        display(fdx)
        pj_arr, pf_arr, pl_arr = get_pl_mean(fdx,fep,t_data[:,idx],c1_data[:,idx],dag[idx],dosetimes)
        pj_collect[idx,fdx,:] = pj_arr
        pf_collect[idx,fdx,:,:] = pf_arr
        pl_collect[idx,fdx,:] = pl_arr
    end
end

# get confidence intervals
ci_collect = zeros(cells,n_params,2)
for i in 1:cells
    ci_collect[i,:,:] = get_ci(1, n_params-1, pl_collect[i,:,:], pj_collect[i,:,:])
end

# profiles
n_profiles = 4
filename = "./plot/experiment/mean_fit_profiles.pdf"
labels = ["DMG", "OSG","SOG","SLG","SAG","DArG"]
plot_profiles_single_mean(cells, n_profiles, n_params, pj_collect, pl_collect, mle, labels, filename)