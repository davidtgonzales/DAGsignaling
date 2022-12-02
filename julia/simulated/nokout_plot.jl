# load packages
using DataFrames
using CSV
using JLD

# include functions
include("../functions/profile_likelihoods.jl")
include("../functions/simulate_models.jl")
include("../functions/plotting_functions.jl")

###########################################
# Plotting for simulated cell populations #
###########################################
# set number of cells
for cells in [10,50,100,200]
    set = string(string(cells),"cells")
    # load data
    t_data = load("./data/simulated/cells_t_data.jld")["t_data"][:,1:cells]
    c1_data = load("./data/simulated/cells_c1_data.jld")["c1_data"][:,1:cells]
    dag = load("./data/simulated/cells_dag.jld")["dag"][1:cells]
    c1_recruit = transpose(mean(c1_data[1:5,:],dims=1)-minimum(c1_data,dims=1))[1:cells]
    # load results
    r_params = load(string("./out/simulated/r_params_",set,".jld"))["r_params"]
    f_params = load(string("./out/simulated/f_params_",set,".jld"))["f_params"]
    obj = load(string("./out/simulated/obj_",set,".jld"))["obj"]
    # simulate fit
    dosetimes = [15]
    c1_fit = simulate_data(t_data[:,1:cells], r_params[end,:,1], mean(c1_data[1:5,1:cells],dims=1), exp.(f_params[end,1:3]), dosetimes, zeros(cells))
    # plot fits
    labels = "None"
    filename = string("./plot/simulated/",set,"_fit.pdf")
    cells_to_plot = 10
    plot_fit(cells_to_plot, t_data, c1_data, c1_fit, dosetimes, labels, filename)
    # parameter profile likelihoods
    pl = zeros(4,11)
    pj = zeros(4,11)
    pr = zeros(4,11,cells,1)
    pf = zeros(4,11,4)
    for fdx in 1:4
        pl[fdx,:] = load(string("./out/simulated/pl_",set,"_",string(fdx),".jld"))["pl_arr"]
        pj[fdx,:] = load(string("./out/simulated/pj_",set,"_",string(fdx),".jld"))["pj_arr"]
        pr[fdx,:,:,1] = load(string("./out/simulated/pr_",set,"_",string(fdx),".jld"))["pr_arr"]
        pf[fdx,:,:] = load(string("./out/simulated/pf_",set,"_",string(fdx),".jld"))["pf_arr"]
    end
    # confidence intervals
    n_params = 4
    ci_collect = get_ci(cells, n_params-1, pl, pj)
    # DAG profile likelihoods
    pl_dag = zeros(5,11)
    pj_dag = zeros(5,11)
    pr_dag = zeros(5,11,cells)
    pf_dag = zeros(5,11,4)
    for fdx in 1:5
        pl_dag[fdx,:] = load(string("./out/simulated/pl_DAG_",set,"_",string(fdx),".jld"))["pl_arr"]
        pj_dag[fdx,:] = load(string("./out/simulated/pj_DAG_",set,"_",string(fdx),".jld"))["pj_arr"]
        pr_dag[fdx,:,:,1] = load(string("./out/simulated/pr_DAG_",set,"_",string(fdx),".jld"))["pr_arr"]
        pf_dag[fdx,:,:] = load(string("./out/simulated/pf_DAG_",set,"_",string(fdx),".jld"))["pf_arr"]
    end
    # confidence intervals
    n_params = 5
    ci_dag = get_ci(cells, n_params-1, pl_dag, log.(pj_dag))
    # plot profiles
    kin = 0.098
    kmet = 0.01823
    Kd = 0.866
    sd = 0.1
    filename = string("./plot/simulated/",set,"_allprofiles.pdf")
    plot_profiles_sim_dag(cells, n_params, pj, pl,pj_dag,pl_dag, obj[end], exp.(f_params[end,:]), r_params[end,1:5], filename,kin,kmet,Kd,sd,dag)
    filename = string("./plot/simulated/",set,"_allprofiles_align.pdf")
    plot_profiles_sim_dag_align(cells, n_params, pj, pl,pj_dag,pl_dag, obj[end], exp.(f_params[end,:]), r_params[end,1:5], filename,kin,kmet,Kd,sd,dag)
    # pairplotting
    dag_err1 = r_params[end,1:5] .- ci_dag[:,1]
    dag_err2 = ci_dag[:,2] .- r_params[end,1:5]
    dag_bins = 0.0:0.05:5
    c1_bins = 0.5:0.05:1.25
    dag_lims = (0,5)
    c1_lims = (0.5,1.25)
    filename = string("./plot/simulated/",set,"_pairplot.pdf")
    pairplot_sim(dag,c1_recruit,r_params,dag_err1,dag_err2,dag_bins,c1_bins,dag_lims,c1_lims,filename,cells)
    # display(exp.(f_params[end,:]))
    # display(ci_collect)
    # display(ci_dag)
    # display(r_params[end,1:5])
end