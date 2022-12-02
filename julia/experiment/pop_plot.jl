# load packages
using DataFrames
using CSV
using JLD

# include functions
include("../functions/simulate_models.jl")
include("../functions/profile_likelihoods.jl")
include("../functions/plotting_functions.jl")
include("../functions/extract_data.jl")

# index of each DAG in dataset
# 2 (MS46) cgDMG
# 5 (MS12) cgSAG
# 6 (MS47) cgDArG
# 7 (MS45) cgSLG
# 9 (MS44) cgOSG
# 11 (MS6) cgSOG

# experimentally determined dag concentrations
dag_mean = [1.824,2.424,2.203,1.850,1.054,2.208]
dag_sd = [0.302,0.402,0.365,0.307,0.175,0.367]

# plot results
for (i,idx) in enumerate([2,5,6,7,9,11])
    # load data
    t_data, c1_data, c1, cells, c1_recruit, dosetimes, set_id = load_dataset(idx)
    # load results
    r_params = load(string("./out/experiment/r_params_",set_id,".jld"))["r_params"]
    f_params = load(string("./out/experiment/f_params_",set_id,".jld"))["f_params"]
    obj = load(string("./out/experiment/obj_",set_id,".jld"))["obj"]
    r_opt = load(string("./out/experiment/r_",set_id,".jld"))["r_opt"]
    # simulate fit
    c1_fit = simulate_data(t_data[:,1:cells], r_params[end,:,1],  mean(c1_data[1:5,1:cells],dims=1), exp.(f_params[end,1:3]), dosetimes, zeros(cells))
    # plot fits
    labels = "None"
    filename = string("./plot/experiment/",set_id,"_fit.pdf")
    cells_to_plot = 10
    plot_fit(cells_to_plot, t_data, c1_data, c1_fit, dosetimes, labels, filename)
    # parameter profile likelihoods
    pl = zeros(4,11)
    pj = zeros(4,11)
    pr = zeros(4,11,cells,1)
    pf = zeros(4,11,4)
    for fdx in 1:4
        pl[fdx,:] = load(string("./out/experiment/pl_",set_id,"_",string(fdx),".jld"))["pl_arr"]
        pj[fdx,:] = load(string("./out/experiment/pj_",set_id,"_",string(fdx),".jld"))["pj_arr"]
        pr[fdx,:,:,1] = load(string("./out/experiment/pr_",set_id,"_",string(fdx),".jld"))["pr_arr"]
        pf[fdx,:,:] = load(string("./out/experiment/pf_",set_id,"_",string(fdx),".jld"))["pf_arr"]
    end
    # confidence intervals
    n_params = 4
    ci_collect = get_ci(cells, n_params-1, pl, pj)
    # DAG profile likelihoods
    pl_dag = zeros(5,11)
    pj_dag = zeros(5,11)
    pf_dag = zeros(5,11,4)
    pr_dag = zeros(5,11,cells)
    for fdx in 1:5
        pl_dag[fdx,:] = load(string("./out/experiment/pl_DAG_",set_id,"_",string(fdx),".jld"))["pl_arr"]
        pj_dag[fdx,:] = load(string("./out/experiment/pj_DAG_",set_id,"_",string(fdx),".jld"))["pj_arr"]
        pf_dag[fdx,:,:] = load(string("./out/experiment/pf_DAG_",set_id,"_",string(fdx),".jld"))["pf_arr"]
        pr_dag[fdx,:,:] = load(string("./out/experiment/pr_DAG_",set_id,"_",string(fdx),".jld"))["pr_arr"]
    end
    # confidence intervals
    n_params = 5
    ci_dag = get_ci(cells, n_params-1, pl_dag, log.(pj_dag))
    filename = string("./plot/experiment/",set_id,"_allprofiles.pdf")
    plot_profiles_expt_dag(cells, n_params, pj, pl,pj_dag,pl_dag, obj[end], exp.(f_params[end,:]), r_params[end,1:5], filename)
    filename = string("./plot/experiment/",set_id,"_allprofiles_align.pdf")
    plot_profiles_expt_dag_align(cells, n_params, pj, pl,pj_dag,pl_dag, obj[end], exp.(f_params[end,:]), r_params[end,1:5], filename)
    # pairplot
    dag_err1 = r_params[end,1:5] .- ci_dag[:,1]
    dag_err2 = ci_dag[:,2] .- r_params[end,1:5]
    err1_replace = findall(j->(j==-Inf),ci_dag[:,1])
    err2_replace = findall(j->(j==Inf),ci_dag[:,2])
    dag_err1[err1_replace] = r_params[end,err1_replace]
    dag_err2[err2_replace] .= 100
    dag_bins = -0.5:0.5:11
    c1_bins = 0:0.2:3
    dag_lims = (-0.5,11)
    c1_lims = (0,3)
    filename = string("./plot/experiment/",set_id,"_pairplot.pdf")
    pairplot_expt(c1_recruit,r_params,dag_err1,dag_err2,dag_bins,c1_bins,dag_lims,c1_lims,dag_mean[i],dag_sd[i],filename,cells)
    # display(exp.(f_params[end,:]))
    # display(ci_collect)
    # display(ci_dag)
    # display(r_params[end,1:5])
    # display(mean(r_params[end,:]))
end


# collate results for comparison plot of parameters
mle_params = zeros(6,4)
ci_params = zeros(6,4,2)
for (i,idx) in enumerate([2,9,11,7,5,6])
    # collect mle params
    t_data, c1_data, c1, cells, c1_recruit, dosetimes, set_id = load_dataset(idx)
    r_params = load(string("./out/experiment/r_params_",set_id,".jld"))["r_params"]
    f_params = load(string("./out/experiment/f_params_",set_id,".jld"))["f_params"]
    obj = load(string("./out/experiment/obj_",set_id,".jld"))["obj"]
    r_opt = load(string("./out/experiment/r_",set_id,".jld"))["r_opt"]
    mle_params[i,:] = f_params[end,:]
    # parameter profile likelihoods
    pl = zeros(4,11)
    pj = zeros(4,11)
    pr = zeros(4,11,cells,1)
    pf = zeros(4,11,4)
    for fdx in 1:4
        pl[fdx,:] = load(string("./out/experiment/pl_",set_id,"_",string(fdx),".jld"))["pl_arr"]
        pj[fdx,:] = load(string("./out/experiment/pj_",set_id,"_",string(fdx),".jld"))["pj_arr"]
        pr[fdx,:,:,1] = load(string("./out/experiment/pr_",set_id,"_",string(fdx),".jld"))["pr_arr"]
        pf[fdx,:,:] = load(string("./out/experiment/pf_",set_id,"_",string(fdx),".jld"))["pf_arr"]
    end
    # confidence intervals
    n_params = 4
    ci_collect = get_ci(cells, n_params-1, pl, pj)
    ci_params[i,:,:] = ci_collect
end
# remove non-identifiable parameters for plotting
ci_params[6,1,2]=1000

x = ["DMG","OSG","SOG","SLG","SAG","DArG"]
fig = plot(layout=(1,4),size=(1200,250),grid=:y,gridalpha=0.3,bottom_margin=10mm,left_margin=10mm,right_margin=3mm,top_margin=2mm,framestyle=:box)
scatter!(x,exp.(mle_params[:,1]),yerr = (exp.(mle_params[:,1])-ci_params[:,1,1],ci_params[:,1,2]-exp.(mle_params[:,1])), ylims=(-0.05,0.50), labelfont = "Helvetica",axis=(font(13)), subplot = 1,ylabel="kin (1/s)",legend=false,ms=5,mc="red",xrotation=60)
scatter!(x,exp.(mle_params[:,2]),yerr = (exp.(mle_params[:,2])-ci_params[:,2,1],ci_params[:,2,2]-exp.(mle_params[:,2])), ylims=(-0.005,0.05), labelfont = "Helvetica",axis=(font(13)), subplot = 2,ylabel="kmet (1/s)",legend=false,ms=5,mc="red",xrotation=60)
scatter!(x,exp.(mle_params[:,3]),yerr = (exp.(mle_params[:,3])-ci_params[:,3,1],ci_params[:,3,2]-exp.(mle_params[:,3])), ylims=(-2,35), labelfont = "Helvetica",axis=(font(13)), subplot = 3,ylabel="Kd (uM)",legend=false,ms=5,mc="red",xrotation=60)
scatter!(x,exp.(mle_params[:,4]),yerr = (exp.(mle_params[:,4])-ci_params[:,4,1],ci_params[:,4,2]-exp.(mle_params[:,4])), ylims=(-0.01,0.16), labelfont = "Helvetica",axis=(font(13)), subplot = 4,ylabel="sd (uM)",xrotation=60,legend=false,ms=5,mc="red")
display(fig)
savefig(fig,"./plot/experiment/compare_params.pdf")