# load packages
using DataFrames
using CSV
using JLD

# include functions
include("../functions/simulate_models.jl")
include("../functions/mle_functions.jl")
include("../functions/profile_likelihoods.jl")
include("../functions/plotting_functions.jl")

###########################################################
# Different amounts of uncaged DAG for model without kout #
###########################################################
# generate simulated data
cells = 5
kin = 0.098
kmet = 0.01823
Kd = 0.866
t_data = repeat(collect(0:3.5:400), outer = [1,cells])
params = [kin,kmet,Kd]
c1_init = [2.5,2.5,2.5,2.5,2.5]
dag = [1.5,1,0.5,0.2,0.1] # titrate different amounts of uncaged DAG
sd = [1E-8,1E-8,1E-8,1E-8,1E-8] # measurement noise
dosetimes = [15]
c1_data = simulate_data(t_data, dag, c1_init, params, dosetimes, sd)
c1_recruit = transpose(mean(c1_data[1:5,:],dims=1)-minimum(c1_data,dims=1))

# get mle
mle = zeros(cells)
logp_mle = zeros((cells,5))
c1_sim = zeros(size(t_data)[1],cells)
for idx in 1:cells
    display(idx)
    logp0 = log.(vcat(kin,kmet,Kd,dag[idx],sd[idx])) # initialize at true values
    # logp0 = log.(vcat(0.1,0.01,1,dag[idx],1E-8)) # initializing not at true values also works
    mle[idx], logp_mle[idx,:] = opt_mle(logp0,t_data[:,idx],c1_data[:,idx],dosetimes)
end
display(exp.(logp_mle))

# get profiles
n_params = 4
pj_collect = zeros(cells,n_params+1,11)
pf_collect = zeros(cells,n_params+1,11,n_params)
pl_collect = zeros(cells,n_params+1,11)
for idx in 1:cells
    display(idx)
    # fep = logp_mle[idx,:] # take profile around MLE
    fep = log.(vcat(params,dag[idx],sd[idx])) # take profile around true values
    for fdx in 1:n_params+1
        display(fdx)
        pj_arr, pf_arr, pl_arr = get_pl(fdx,fep,t_data[:,idx],c1_data[:,idx],dosetimes)
        pj_collect[idx,fdx,:] = pj_arr
        pf_collect[idx,fdx,:,:] = pf_arr
        pl_collect[idx,fdx,:] = pl_arr
    end
end

# get confidence intervals
ci_collect = zeros(cells,n_params+1,2)
for i in 1:cells
    ci_collect[i,:,:] = get_ci(1, n_params, pl_collect[i,:,:], pj_collect[i,:,:])
end

# plot results
c1_fit = zeros(size(c1_data))
for i in 1:cells
    c1_fit[:,i] = simulate_data_single(t_data[:,i], exp.(logp_mle[i,4]), mean(c1_data[1:5,i]), exp.(logp_mle[i,1:3]), dosetimes, 0)
end

# plot fitting
labels = reshape(["1.5uM DAG","1","0.5","0.2","0.1"],1,5)
filename = "./plot/simulated/dag_fit.pdf"
plot_fit_single(t_data, c1_data, c1_fit, dosetimes, labels, filename)

# plot profiles
n_profiles = 5
filename = "./plot/simulated/dag_profiles.pdf"
plot_profiles_single(n_params, pj_collect, pl_collect, mle, filename)

#################################################################
# Different amounts of measurement noise for model without kout #
#################################################################
# generate simulated data
cells = 5
kin = 0.098
kmet = 0.01823
Kd = 0.866
t_data = repeat(collect(0:3.5:400), outer = [1,cells])
params = [kin,kmet,Kd]
c1_init = [2.5,2.5,2.5,2.5,2.5]
dag = [1,1,1,1,1] # uncaged DAG
sd = [0.001,0.01,0.05,0.1,0.15] # titrate different amounts of measurement noise
dosetimes = [15]
c1_data = simulate_data(t_data, dag, c1_init, params, dosetimes, sd)
c1_recruit = transpose(mean(c1_data[1:5,:],dims=1)-minimum(c1_data,dims=1))

# get mle
mle = zeros(cells)
logp_mle = zeros((cells,5))
c1_sim = zeros(size(t_data)[1],cells)
for idx in 1:cells
    display(idx)
    logp0 = log.(vcat(kin,kmet,Kd,dag[idx],sd[idx])) # initialize at true values
    mle[idx], logp_mle[idx,:] = opt_mle(logp0,t_data[:,idx],c1_data[:,idx],dosetimes)
end
display(exp.(logp_mle))

# get profiles
n_params = 4
pj_collect = zeros(cells,n_params+1,11)
pf_collect = zeros(cells,n_params+1,11,n_params)
pl_collect = zeros(cells,n_params+1,11)
for idx in 1:cells
    display(idx)
    fep = log.(vcat(params,dag[idx],sd[idx])) # take profile around true values
    for fdx in 1:n_params+1
        display(fdx)
        pj_arr, pf_arr, pl_arr = get_pl(fdx,fep,t_data[:,idx],c1_data[:,idx],dosetimes)
        pj_collect[idx,fdx,:] = pj_arr
        pf_collect[idx,fdx,:,:] = pf_arr
        pl_collect[idx,fdx,:] = pl_arr
    end
end

# get confidence intervals
ci_collect = zeros(cells,n_params+1,2)
for i in 1:cells
    ci_collect[i,:,:] = get_ci(1, n_params, pl_collect[i,:,:], pj_collect[i,:,:])
end

# plot results
c1_fit = zeros(size(c1_data))
for i in 1:cells
    c1_fit[:,i] = simulate_data_single(t_data[:,i], exp.(logp_mle[i,4]), mean(c1_data[1:5,i]), exp.(logp_mle[i,1:3]), dosetimes, 0)
end

# plot fitting
labels = reshape(["0.15uM stdev","0.1","0.05","0.01","0.001"],1,5) # reversed in order so that the high noise don't cover the low noise plots
filename = "./plot/simulated/noise_fit.pdf"
plot_fit_single(t_data[:,[5,4,3,2,1]], c1_data[:,[5,4,3,2,1]], c1_fit[:,[5,4,3,2,1]], dosetimes, labels, filename)

# plot profiles
n_profiles = 5
filename = "./plot/simulated/noise_profiles.pdf"
plot_profiles_single(n_params, pj_collect[[5,4,3,2,1],:,:], pl_collect[[5,4,3,2,1],:,:], reverse(mle), filename)

########################################################
# Different amounts of uncaged DAG for model with kout #
########################################################
# generate simulated data
cells = 5
kin = 0.065
kout = 0.12
kmet = 0.056
Kd = 0.28
t_data = repeat(collect(0:3.5:400), outer = [1,cells])
params = [kin,kout,kmet,Kd]
c1_init = [2.5,2.5,2.5,2.5,2.5]
dag = [1.5,1,0.5,0.2,0.1] # titrate different amounts of uncaged DAG
sd = [1E-8,1E-8,1E-8,1E-8,1E-8] # measurement noise
dosetimes = [15]
c1_data = simulate_data(t_data, dag, c1_init, params, dosetimes, sd)
c1_recruit = transpose(mean(c1_data[1:5,:],dims=1)-minimum(c1_data,dims=1))

# get mle
mle = zeros(cells)
logp_mle = zeros((cells,6))
c1_sim = zeros(size(t_data)[1],cells)
for idx in 1:cells
    display(idx)
    logp0 = log.(vcat(kin,kout,kmet,Kd,dag[idx],sd[idx])) # initialize at true values
    # logp0 = log.(vcat(0.1,0.1,0.01,0.1,dag[idx],1E-8)) # initializing not at true values also works but not as well as simpler model without kout
    mle[idx], logp_mle[idx,:] = opt_mle(logp0,t_data[:,idx],c1_data[:,idx],dosetimes)
end
display(exp.(logp_mle))

# get profiles
n_params = 5
pj_collect = zeros(cells,n_params+1,11)
pf_collect = zeros(cells,n_params+1,11,n_params)
pl_collect = zeros(cells,n_params+1,11)
for idx in 1:cells
    display(idx)
    fep = log.(vcat(params,dag[idx],sd[idx])) # take profile around true values
    for fdx in 1:n_params+1
        display(fdx)
        pj_arr, pf_arr, pl_arr = get_pl(fdx,fep,t_data[:,idx],c1_data[:,idx],dosetimes)
        pj_collect[idx,fdx,:] = pj_arr
        pf_collect[idx,fdx,:,:] = pf_arr
        pl_collect[idx,fdx,:] = pl_arr
    end
end

# get confidence intervals
ci_collect = zeros(cells,n_params+1,2)
for i in 1:cells
    ci_collect[i,:,:] = get_ci(1, n_params, pl_collect[i,:,:], pj_collect[i,:,:])
end

# plot results
c1_fit = zeros(size(c1_data))
for i in 1:cells
    c1_fit[:,i] = simulate_data_single(t_data[:,i], exp.(logp_mle[i,5]), mean(c1_data[1:5,i]), exp.(logp_mle[i,1:4]), dosetimes, 0)
end

# fitting
labels = reshape(["1.5uM DAG","1","0.5","0.2","0.1"],1,5)
filename = "./plot/simulated/dag_fit_kout.pdf"
plot_fit_single(t_data, c1_data, c1_fit, dosetimes, labels, filename)

# profiles
n_profiles = 6
filename = "./plot/simulated/dag_profiles_kout.pdf"
plot_profiles_single_kout(n_params, pj_collect, pl_collect, mle, filename)

##############################################################
# Different amounts of measurement noise for model with kout #
##############################################################
# generate simulated data
cells = 5
kin = 0.065
kout = 0.12
kmet = 0.056
Kd = 0.28
t_data = repeat(collect(0:3.5:400), outer = [1,cells])
params = [kin,kout,kmet,Kd]
c1_init = [2.5,2.5,2.5,2.5,2.5]
dag = [1,1,1,1,1] # uncaged DAG
sd = [0.001,0.01,0.05,0.1,0.15] # titrate different amounts of measurement noise
dosetimes = [15]
c1_data = simulate_data(t_data, dag, c1_init, params, dosetimes, sd)
c1_recruit = transpose(mean(c1_data[1:5,:],dims=1)-minimum(c1_data,dims=1))

# get mle
mle = zeros(cells)
logp_mle = zeros((cells,6))
c1_sim = zeros(size(t_data)[1],cells)
for idx in 1:cells
    display(idx)
    logp0 = log.(vcat(kin,kout,kmet,Kd,dag[idx],sd[idx])) # initialize at true values
    mle[idx], logp_mle[idx,:] = opt_mle(logp0,t_data[:,idx],c1_data[:,idx],dosetimes)
end
display(mle)
display(exp.(logp_mle))

# get profiles
n_params = 5
pj_collect = zeros(cells,n_params+1,11)
pf_collect = zeros(cells,n_params+1,11,n_params)
pl_collect = zeros(cells,n_params+1,11)
for idx in 1:cells
    display(idx)
    fep = log.(vcat(params,dag[idx],sd[idx]))
    for fdx in 1:n_params+1
        display(fdx)
        pj_arr, pf_arr, pl_arr = get_pl(fdx,fep,t_data[:,idx],c1_data[:,idx],dosetimes)
        pj_collect[idx,fdx,:] = pj_arr
        pf_collect[idx,fdx,:,:] = pf_arr
        pl_collect[idx,fdx,:] = pl_arr
    end
end

# get confidence intervals
ci_collect = zeros(cells,n_params+1,2)
for i in 1:cells
    ci_collect[i,:,:] = get_ci(1, n_params, pl_collect[i,:,:], pj_collect[i,:,:])
end

# plot results
c1_fit = zeros(size(c1_data))
for i in 1:cells
    c1_fit[:,i] = simulate_data_single(t_data[:,i], exp.(logp_mle[i,5]), mean(c1_data[1:5,i]), exp.(logp_mle[i,1:4]), dosetimes, 0)
end

# plot fitting
labels = reshape(["0.15uM stdev","0.1","0.05","0.01","0.001"],1,5)  # reversed in order so that the high noise don't cover the low noise plots
filename = "./plot/simulated/noise_fit_kout.pdf"
plot_fit_single(t_data[:,[5,4,3,2,1]], c1_data[:,[5,4,3,2,1]], c1_fit[:,[5,4,3,2,1]], dosetimes, labels, filename)

# plot profiles
n_profiles = 6
filename = "./plot/simulated/noise_profiles_kout.pdf"
plot_profiles_single_kout(n_params, pj_collect[[5,4,3,2,1],:,:], pl_collect[[5,4,3,2,1],:,:], reverse(mle), filename)