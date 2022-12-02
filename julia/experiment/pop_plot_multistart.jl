# load packages
using DataFrames
using CSV
using JLD

# include functions
include("../functions/simulate_models.jl")
include("../functions/mle_functions.jl")
include("../functions/extract_data.jl")
include("../functions/plotting_functions.jl")

# load data
# index of each DAG in dataset
# 2 (MS46) cgDMG
# 5 (MS12) cgSAG
# 6 (MS47) cgDArG
# 7 (MS45) cgSLG
# 9 (MS44) cgOSG
# 11 (MS6) cgSOG

filename = "./plot/experiment/multistart_pop.pdf"
labels = reshape(["DMG", "OSG","SOG","SLG","SAG","DArG"],1,6) # arranged according to main text (DMG, OSG, SOG,SLG, SAG, DArG)
fig = plot(layout=(2,3),size=(1200,600),grid=false,bottom_margin=10mm,left_margin=10mm,right_margin=5mm,top_margin=2mm,framestyle=:box)

for (i,idx) in enumerate([2,9,11,7,5,6])
    # load data
    t_data, c1_data, c1, cells, c1_recruit, dosetimes, set_id = load_dataset(idx)
    # open results
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
    # plot
    n = size(p_mat)[1]
    mle_sort = mle_multi[sortperm(mle_multi)]
    scatter!(mle_sort,color="black",subplot=i,xlabel="Sorted fit",ylabel="-log(Likelihood)",legend=false,labelfont = "Helvetica",axis=(font(17)),title = labels[i],titlefontsize=20,ms=3,msw=0)
end
display(fig)
savefig(fig,filename)