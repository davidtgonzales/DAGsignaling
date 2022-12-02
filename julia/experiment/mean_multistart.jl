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

t_data = zeros(120,6)
c1_data = zeros(120,6)
c1_init = zeros(6)
c1_recruit = zeros(6)
for (i,idx) in enumerate([2,9,11,7,5,6]) # arranged according to main text (DMG, OSG, SOG,SLG, SAG, DArG)
    t_datai, c1_datai, c1i, cells, c1_recruiti, dosetimes, set_id = load_dataset(idx)
    # get mean values
    t_data[:,i] = mean(t_datai,dims=2)
    c1_data[:,i] = mean(c1_datai,dims=2)
    c1_init[i] = mean(c1i)
    c1_recruit[i] = mean(c1_recruiti)
end

n = 100 # 100 trials of random initial conditions
mle_multi = zeros(n,6)
logp_multi = zeros(n,6,4)

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

for i in 1:n
    display(i)
    kin = kin_draw[i]
    kmet = kmet_draw[i]
    Kd = Kd_draw[i]
    sd = sd_draw[i]
    # dag = [1.2106,0.6996 ,1.465,1.227,1.6086,1.4617] # average measured concentrations of DAG
    dag = [1.824383581,1.054389975,2.207953554,1.849868644,2.424218508,2.202893915] # average measured concentrations of DAG
    dosetimes = [14]
    cells = 6
    mle = zeros(cells)
    logp_mle = zeros((cells,4))
    c1_sim = zeros(size(t_data)[1],cells)
    for idx in 1:cells
        logp0 = log.(vcat(kin,kmet,Kd,sd))
        mle[idx], logp_mle[idx,:] = opt_mle_mean(logp0,t_data[:,idx],c1_data[:,idx],dag[idx],dosetimes)
    end
    mle_multi[i,:] = mle
    logp_multi[i,:,:] = logp_mle
end

# save results
p_init = hcat(kin_draw,kmet_draw,Kd_draw,sd_draw)
save(string("./out/experiment/p_init_multi_mean.jld"), "p_init", p_init)
save(string("./out/experiment/mle_multi_mean.jld"), "mle_multi", mle_multi)
save(string("./out/experiment/logp_multi_mean.jld"), "logp_multi", logp_multi)

labels = reshape(["DMG", "OSG","SOG","SLG","SAG","DArG"],1,6) # arranged according to main text (DMG, OSG, SOG,SLG, SAG, DArG)
filename = "./plot/experiment/multistart_mean.pdf"

fig = plot(layout=(2,3),size=(1200,600),grid=false,bottom_margin=10mm,left_margin=10mm,right_margin=5mm,top_margin=5mm,framestyle=:box)
mle_sort = mle_multi[sortperm(mle_multi[:,1]),1]
scatter!(mle_sort,color="black",subplot=1,xlabel="Sorted fit",ylabel="-log(Likelihood)",legend=false,labelfont = "Helvetica",axis=(font(17)),title = labels[1],titlefontsize=20,ms=3,msw=0)
mle_sort = mle_multi[sortperm(mle_multi[:,2]),2]
scatter!(mle_sort,color="black",subplot=2,xlabel="Sorted fit",ylabel="-log(Likelihood)",legend=false,labelfont = "Helvetica",axis=(font(17)),title = labels[2],titlefontsize=20,ms=3,msw=0)
mle_sort = mle_multi[sortperm(mle_multi[:,3]),3]
scatter!(mle_sort,color="black",subplot=3,xlabel="Sorted fit",ylabel="-log(Likelihood)",legend=false,labelfont = "Helvetica",axis=(font(17)),title = labels[3],titlefontsize=20,ms=3,msw=0)
mle_sort = mle_multi[sortperm(mle_multi[:,4]),4]
scatter!(mle_sort,color="black",subplot=4,xlabel="Sorted fit",ylabel="-log(Likelihood)",legend=false,labelfont = "Helvetica",axis=(font(17)),title = labels[4],titlefontsize=20,ms=3,msw=0)
mle_sort = mle_multi[sortperm(mle_multi[:,5]),5]
scatter!(mle_sort,color="black",subplot=5,xlabel="Sorted fit",ylabel="-log(Likelihood)",legend=false,labelfont = "Helvetica",axis=(font(17)),title = labels[5],titlefontsize=20,ms=3,msw=0)
mle_sort = mle_multi[sortperm(mle_multi[:,6]),6]
scatter!(mle_sort,color="black",subplot=6,xlabel="Sorted fit",ylabel="-log(Likelihood)",legend=false,labelfont = "Helvetica",axis=(font(17)),title = labels[6],titlefontsize=20,ms=3,msw=0)
display(fig)
savefig(fig,filename)

