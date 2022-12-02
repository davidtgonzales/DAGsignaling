# load packages
using DataFrames
using CSV
using JLD

# include functions
include("../functions/simulate_models.jl")
include("../functions/profile_likelihoods.jl")
include("../functions/plotting_functions.jl")
include("../functions/extract_data.jl")

fig = plot(layout=(3,2),size=(800,800),bottom_margin=2mm,left_margin=10mm,right_margin=3mm,top_margin=2mm,framestyle=:box)
labels = ["DMG","OSG","SOG","SLG","SAG","DArG"]
for (idx,i) in enumerate([2,9,11,7,5,6])
    t_data, c1_data, c1, cells, c1_recruit, dosetimes, set_id = load_dataset(i)
    # load results
    r_params = load(string("./out/experiment/r_params_",set_id,".jld"))["r_params"]
    f_params = load(string("./out/experiment/f_params_",set_id,".jld"))["f_params"]
    obj = load(string("./out/experiment/obj_",set_id,".jld"))["obj"]
    # simulate fit
    c1_fit = simulate_data(t_data[:,1:cells], r_params[end,1:cells,1],  mean(c1_data[1:5,1:cells],dims=1), exp.(f_params[end,1:3]), dosetimes, zeros(cells))
    # get residuals
    c1_res = c1_data[:,1:cells] .- c1_fit
    violin!(string.(transpose(floor.(Int,t_data[collect(1:10:120),1]))),transpose(c1_res[collect(1:10:120),:]), lw=0, color = "grey",side=:right, linewidth=0, label="",title=labels[idx],ylabel="Residuals (uM)",xlabel="Time (s)",subplot=idx,xrotation=45,ylims=(-0.6,0.6))
    hline!([0],ls=:dash,color="black",label="",subplot=idx)
end
display(fig)
savefig(fig,"./plot/experiment/rss_hist.pdf")


fig = plot(layout=(1,1),size=(600,200),bottom_margin=10mm,left_margin=10mm,right_margin=3mm,top_margin=2mm,framestyle=:box)
labels = ["DMG","OSG","SOG","SLG","SAG","DArG"]
for (idx,i) in enumerate([2,9,11,7,5,6])
    t_data, c1_data, c1, cells, c1_recruit, dosetimes, set_id = load_dataset(i)
    cells = 80
    # load results
    r_params = load(string("./out/experiment/r_params_",set_id,".jld"))["r_params"]
    f_params = load(string("./out/experiment/f_params_",set_id,".jld"))["f_params"]
    obj = load(string("./out/experiment/obj_",set_id,".jld"))["obj"]
    # simulate fit
    c1_fit = simulate_data(t_data[:,1:cells], r_params[end,1:cells,1],  mean(c1_data[1:5,1:cells],dims=1), exp.(f_params[end,1:3]), dosetimes, zeros(cells))
    # get residuals
    c1_res = c1_data[:,1:cells] .- c1_fit
    plot!(t_data[1:120,1],sum(c1_res.*c1_res,dims=2),ylabel="RSS",xlabel="Time (s)",palette = palette(:rainbow),  label=labels[idx],foreground_color_legend = nothing, background_color_legend=nothing,legend = :outertopright)
end
display(fig)
savefig(fig,"./plot/experiment/rss.pdf")
