# load packages
using MAT
using JLD
using DataFrames
using CSV
using Distributions

# include functions
include("../functions/extract_data.jl")

# open raw data
raw = matread("./raw/DAG_traces.mat")
data = raw["Data"]

# restructure data
cells = get_totalcells(data,["Cells","Normalized_profile"])
timepoints = 120
times = zeros(timepoints,cells)
concentrations = zeros(timepoints,cells)
compounds = Array{String,1}(undef,cells)
areas = zeros(cells)
k = 1
for i in 1:length(data["Cells"])
    cell = data["Cells"][i]
    cell_ids = unique(data["Normalized_profile"][i][:,2])
    df = data["Normalized_profile"][i]
    for j in cell_ids
        cell_profile = df[findall(x->x==j,df[:,2]),:]
        times[:,k] = cell_profile[1:timepoints,1]
        concentrations[:,k] = cell_profile[1:timepoints,3]
        compounds[k] = data["Compound"][i]
        areas[k] = cell_profile[1,4]
        k = k + 1
    end
end

# filter data by removing cells that have a higher C1 than initial concentration by 0.2uM
filter_idx = vec(mean(concentrations[1:5,:],dims=1).+0.2 .<= maximum(concentrations,dims=1))
times = times[:,findall(x->x ==0, filter_idx)]
concentrations = concentrations[:,findall(x->x ==0, filter_idx)]
compounds = compounds[findall(x->x ==0, filter_idx)]
areas = areas[findall(x->x ==0, filter_idx)]

# save restructured data
save("./data/experiment/times.jld", "times", times)
save("./data/experiment/concentrations.jld", "concentrations", concentrations)
save("./data/experiment/compounds.jld", "compounds", compounds)
save("./data/experiment/areas.jld", "areas", areas)