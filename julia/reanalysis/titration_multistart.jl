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

# short traces laser power titration
LP_list = [0,0.1,0.2,0.5,1,5,10,20,40]
LP_label = ["00","01","02","05","1","5","10","20","40"]
labels = ["0% LP" "0.1" "0.2" "0.5" "1" "5" "10" "20" "40"]

# long traces laser power titration
LP_list_long = [10,40]
LP_label_long = ["10","40"]

# load normalized data
t_mean = load(string("./data/reanalysis/t_mean.jld"))["t_mean"][1:3,:,:]
c1_norm = load(string("./data/reanalysis/c1_norm.jld"))["c1_norm"]
t_mean_long = load(string("./data/reanalysis/t_mean_long.jld"))["t_mean_long"][1:3,:,:]
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

# initialize multistart
n = 100 # 100 trials of random initial conditions
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
p_mat = zeros(n,4)
p_mat[:,1] = kin_draw
p_mat[:,2] = kmet_draw
p_mat[:,3] = Kd_draw
p_mat[:,4] = sd_draw

# measured DAG uncaging
SAG = [0,3.94E5,6.93E5,9.79E5,1.15E6,1.37E6,1.56E6,1.93E6,2.3E6]./6.022E23*1E15*1000000/3053.819
SOG = [0,1.75E5,3.94E5,5E5,6.39E5,1.16E6,1.38E6,2.12E6,2.96E6]./6.022E23*1E15*1000000/3053.819
DOG = [0,3.34E5,6.27E5,7.68E5,1.12E6,1.44E6,1.78E6,2.34E6,2.96E6]./6.022E23*1E15*1000000/3053.819
DAG = [SAG, SOG, DOG]
SAG_long = [1.56E6,2.3E6]./6.022E23*1E15*1000000/3053.819
SOG_long = [1.38E6,2.96E6]./6.022E23*1E15*1000000/3053.819
DOG_long = [1.78E6,2.96E6]./6.022E23*1E15*1000000/3053.819
DAG_long = [SAG_long,SOG_long,DOG_long]

filename_labels = ["SAG","SOG","DOG"]
mle_collect = zeros(3,n)
logp_collect = zeros(3,n,4)
for i in 1:3
    cells = 11
    dosetimes = [14]
    r_max = 1
    mle_multi = zeros(n)
    logp_multi = zeros(n,4)
    for j in 1:n
        display(j) 
        logp0 = log.(p_mat[j,:])
        mle_multi[j], logp_multi[j,:] = opt_mle_mean_titration(log.(p_mat[j,:]),t_mean[i,:,:],c1_uM[i,:,:].-DAG_plat[i],t_mean_long[i,:,:],c1_uM_long[i,:,:].-DAG_plat[i],DAG[i],DAG_long[i],dosetimes,cells)
    end
    mle_collect[i,:] = mle_multi
    logp_collect[i,:,:] = logp_multi
    min_idx = collect(getindex.(argmin(mle_multi,dims=1),1))
    display(exp.(logp_multi[min_idx,:]))
    c1_sim = simulate_data(t_mean[i,:,:], DAG[i], c1_uM[i,5,:].-DAG_plat[i], vec(exp.(logp_multi[min_idx,1:3])), [14], zeros(9))
    fig = plot(layout=(1,1),size=(500,280),grid=false,bottom_margin=5mm,left_margin=5mm,right_margin=5mm,top_margin=2mm,framestyle=:box)
    plot!(t_mean[i,:,:],c1_uM[i,:,:],labelfont = "Helvetica",axis=(font(16)),legend = :outertopright,lw=1.5, palette=:YlGn_9, foreground_color_legend = nothing, background_color_legend=nothing, legendfontsize=11,label=labels,xlabel="Time (s)",ylabel="C1-EGFP-NES (uM)")
    plot!(t_mean[i,:,:],c1_sim.+DAG_plat[i],ls=:dash, label="",labelfont= "Helvetica",axis=(font(16)),color="black")
    display(fig)
    savefig(string("./plot/reanalysis/",filename_labels[i],"_fit.pdf"))
end

DAG_labels = ["SAG","SOG","DOG"]
fig = plot(layout=(1,3),size=(1200,300),grid=false,bottom_margin=10mm,left_margin=10mm,right_margin=5mm,top_margin=5mm,framestyle=:box)
for i in 1:3
    mle_sort = mle_collect[i,sortperm(mle_collect[i,:])]
    scatter!(mle_sort,color="black",subplot=i,xlabel="Sorted fit",ylabel="-log(Likelihood)",legend=false,labelfont = "Helvetica",axis=(font(17)),title = DAG_labels[i],titlefontsize=20,ms=3,msw=0)
end
display(fig)
savefig(string("./plot/reanalysis/multistart.pdf"))

save("./out/reanalysis/mle_collect.jld", "mle_collect", mle_collect)
save("./out/reanalysis/logp_collect.jld", "logp_collect", logp_collect)
save("./out/reanalysis/p_mat.jld", "p_mat", p_mat)