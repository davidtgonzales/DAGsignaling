# load packages
using DataFrames
using CSV
using JLD
using Distributions

function load_dataset(idx)
    # index of each DAG in dataset
    # 2 (MS46) cgDMG
    # 5 (MS12) cgSAG
    # 6 (MS47) cgDArG
    # 7 (MS45) cgSLG
    # 9 (MS44) cgOSG
    # 11 (MS6) cgSOG
    times = load("./data/experiment/times.jld")["times"]
    concentrations = load("./data/experiment/concentrations.jld")["concentrations"]
    compounds = load("./data/experiment/compounds.jld")["compounds"]
    list_comp = unique([x[2] for x in [split.(unique(compounds),"_")][1]])
    comp = list_comp[idx]
    index_comp = findall(x->occursin(comp,x), compounds)
    t_data = times[:,index_comp]
    c1_data = concentrations[:,index_comp]
    c1 = mean(c1_data[1:5,:],dims=1)
    cells = size(t_data)[2]
    c1_recruit = transpose(maximum(c1_data,dims=1)-minimum(c1_data,dims=1))
    dosetimes = [14]
    set_id = comp
    return t_data, c1_data, c1, cells, c1_recruit, dosetimes, set_id
end

function get_totalcells(data,keys)
    cells = 0
    for i in 1:length(data[keys[1]])
        cell_ids = unique(data[keys[2]][i][:,2])
        cells = cells + length(cell_ids)
    end
    return cells
end