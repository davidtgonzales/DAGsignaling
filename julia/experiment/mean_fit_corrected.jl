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

##############################################################################
# MLE on population mean of experimental data with correction for saturation #
##############################################################################
# load data
# index of each DAG in dataset
# 5 (MS12) cgSAG
# 11 (MS6) cgSOG
t_data = zeros(120,2)
c1_data = zeros(120,2)
c1_init = zeros(2)
c1_recruit = zeros(2)
for (i,idx) in enumerate([11,5]) # SOG and SAG
    t_datai, c1_datai, c1i, cells, c1_recruiti, dosetimes, set_id = load_dataset(idx)
    # get mean values
    t_data[:,i] = mean(t_datai,dims=2)
    c1_data[:,i] = mean(c1_datai,dims=2)
    c1_init[i] = mean(c1i)
    c1_recruit[i] = mean(c1_recruiti)
end

# multistart
n = 100 # 100 trials of random initial conditions
mle_multi = zeros(n,2)
logp_multi = zeros(n,2,4)

d1 = Uniform(0,1)
d2 = Uniform(0,1)
d3 = Uniform(0,20)
d4 = Uniform(0,0.5)
kin_draw = rand(d1,n)
kmet_draw = rand(d2,n)
Kd_draw = rand(d3,n)
sd_draw = rand(d4,n)

# for 1st initial parameters
kin_draw[1] = 0.1
kmet_draw[1] = 0.01
Kd_draw[1] = 1
sd_draw[1] = 0.1

# correction of available C1
correction = mean(c1_data[1:5,:],dims=1).*[0.75 0.25] # SOG and SAG correction factor
c1_data_corr = c1_data .- correction # deduct correction from c1 data

# average measured concentrations of SOG and SAG
dag = [2.207953554,2.424218508]
for i in 1:n
    display(i)
    kin = kin_draw[i]
    kmet = kmet_draw[i]
    Kd = Kd_draw[i]
    sd = sd_draw[i]
    dosetimes = [14]
    cells = 2
    mle = zeros(cells)
    logp_mle = zeros((cells,4))
    c1_sim = zeros(size(t_data)[1],cells)
    for idx in 1:cells
        logp0 = log.(vcat(kin,kmet,Kd,sd))
        mle[idx], logp_mle[idx,:] = opt_mle_mean(logp0,t_data[:,idx],c1_data_corr[:,idx],dag[idx],dosetimes)
    end
    mle_multi[i,:] = mle
    logp_multi[i,:,:] = logp_mle
end

# save results
p_init = hcat(kin_draw,kmet_draw,Kd_draw,sd_draw)
save(string("./out/experiment/p_init_multi_mean_corrected.jld"), "p_init", p_init)
save(string("./out/experiment/mle_multi_mean_corrected.jld"), "mle_multi", mle_multi)
save(string("./out/experiment/logp_multi_mean_corrected.jld"), "logp_multi", logp_multi)

labels = reshape(["SOG","SAG"],1,2)
filename = "./plot/experiment/multistart_mean_corrected.pdf"
fig = plot(layout=(1,2),size=(800,300),grid=false,bottom_margin=10mm,left_margin=10mm,right_margin=5mm,top_margin=5mm,framestyle=:box)
mle_sort = mle_multi[sortperm(mle_multi[:,1]),1]
scatter!(mle_sort,color="black",subplot=1,xlabel="Sorted fit",ylabel="-log(Likelihood)",legend=false,labelfont = "Helvetica",axis=(font(17)),title = labels[1],titlefontsize=20,ms=3,msw=0)
mle_sort = mle_multi[sortperm(mle_multi[:,2]),2]
scatter!(mle_sort,color="black",subplot=2,xlabel="Sorted fit",ylabel="-log(Likelihood)",legend=false,labelfont = "Helvetica",axis=(font(17)),title = labels[2],titlefontsize=20,ms=3,msw=0)
display(fig)
savefig(fig,filename)

# load best initial conditions from multistart results
p_init = load(string("./out/experiment/p_init_multi_mean_corrected.jld"))["p_init"]
mle_multi = load(string("./out/experiment/mle_multi_mean_corrected.jld"))["mle_multi"]
logp_multi = load(string("./out/experiment/logp_multi_mean_corrected.jld"))["logp_multi"]

# get mle
dosetimes = [14]
cells = 2
mle = zeros(cells)
logp_mle = zeros((cells,4))
c1_sim = zeros(size(t_data)[1],cells)
for idx in 1:cells
    display(idx)
    logp0 = logp_multi[min_idx[idx],idx,:]
    mle[idx], logp_mle[idx,:] = opt_mle_mean(logp0,t_data[:,idx],c1_data_corr[:,idx],dag[idx],dosetimes)
end
display(mle)
display(exp.(logp_mle))


# plot results
c1_fit = zeros(size(c1_data))
for i in 1:cells
    c1_fit[:,i] = simulate_data_single(t_data[:,i], dag[i], mean(c1_data_corr[1:5,i]), exp.(logp_mle[i,1:3]), dosetimes, 0)
end

labels = reshape(["SOG","SAG"],1,2)
cells = 2
filename = "./plot/experiment/mean_fit_corrected.pdf"
plot_fit_single_mean(cells, t_data, c1_data, c1_fit.+correction, dosetimes, labels, filename)

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
        pj_arr, pf_arr, pl_arr = get_pl_mean(fdx,fep,t_data[:,idx],c1_data_corr[:,idx],dag[idx],dosetimes)
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

# plot profiles
n_profiles = 4
filename = "./plot/experiment/mean_fit_profiles_corrected.pdf"
labels = ["SOG","SAG"]
plot_profiles_single_mean(cells, n_profiles, n_params, pj_collect, pl_collect, mle, labels, filename)

