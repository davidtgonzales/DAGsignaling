# load packages
using MAT
using JLD
using DataFrames
using CSV
using Distributions

# read data
collected_df = DataFrame(time=Float64[],channel=String[],ROI=String[],LP=String[],Probe=String[],Mean=Float64[])
raw_path = "./raw/Schuhmacher2020/"
files = readdir(raw_path)
for file in files
    df = DataFrame(CSV.File(string(raw_path,file)))
    rename!(df,[:time,:channel,:ROI,:LP,:Probe,:Mean])
    append!(collected_df,df)
end

for MS in ["MS12","MS6","MS18","MS19"] # SAG, SOG, 1,3DOG, DOG
    # subset to different probes
    df_MS = filter(:Probe => n -> n == MS, collected_df)
    # rearrange datasets for MS
    rois = unique(df_MS[!,:ROI])
    short_idxs =  Vector{Int}()
    long_idxs = Vector{Int}()
    # get index of cells for short ot long trace
    for (idx,roi) in enumerate(rois)
        subset_df = filter(:ROI => n -> n == roi, df_MS)
        if subset_df[!,:time][end]==413
            append!(long_idxs,idx)
        elseif subset_df[!,:time][end]==102
            append!(short_idxs,idx)
        end
    end
    # short traces
    cells = length(rois[short_idxs])
    t_data = zeros(30,cells)
    c1_rfu = zeros(30,cells)
    lp_data = zeros(cells)
    for (idx,roi) in enumerate(rois[short_idxs])
        subset_df = filter(:ROI => n -> n == roi, df_MS)
        t_data[:,idx] = subset_df[!,:time][1:30]
        c1_rfu[:,idx] = subset_df[!,:Mean][1:30]
        if subset_df[!,:LP][1] == "LP0"
            lp_data[idx] = 0
        elseif subset_df[!,:LP][1] == "LP01"
            lp_data[idx] = 0.1
        elseif subset_df[!,:LP][1] == "LP02"
            lp_data[idx] = 0.2
        elseif subset_df[!,:LP][1] == "LP05"
            lp_data[idx] = 0.5
        elseif subset_df[!,:LP][1] == "LP1"
            lp_data[idx] = 1
        elseif subset_df[!,:LP][1] == "LP5"
            lp_data[idx] = 5
        elseif subset_df[!,:LP][1] == "LP10"
            lp_data[idx] = 10
        elseif subset_df[!,:LP][1] == "LP20"
            lp_data[idx] = 20
        elseif subset_df[!,:LP][1] == "LP40"
            lp_data[idx] = 40
        end
    end
    # long traces
    cells = length(rois[long_idxs])
    t_data_long = zeros(120,cells)
    c1_rfu_long = zeros(120,cells)
    lp_data_long = zeros(cells)
    for (idx,roi) in enumerate(rois[long_idxs])
        subset_df = filter(:ROI => n -> n == roi, df_MS)
        t_data_long[:,idx] = subset_df[!,:time][1:120]
        c1_rfu_long[:,idx] = subset_df[!,:Mean][1:120]
        if subset_df[!,:LP][1] == "LP0"
            lp_data_long[idx] = 0
        elseif subset_df[!,:LP][1] == "LP01"
            lp_data_long[idx] = 0.1
        elseif subset_df[!,:LP][1] == "LP02"
            lp_data_long[idx] = 0.2
        elseif subset_df[!,:LP][1] == "LP05"
            lp_data_long[idx] = 0.5
        elseif subset_df[!,:LP][1] == "LP1"
            lp_data_long[idx] = 1
        elseif subset_df[!,:LP][1] == "LP5"
            lp_data_long[idx] = 5
        elseif subset_df[!,:LP][1] == "LP10"
            lp_data_long[idx] = 10
        elseif subset_df[!,:LP][1] == "LP20"
            lp_data_long[idx] = 20
        elseif subset_df[!,:LP][1] == "LP40"
            lp_data_long[idx] = 40
        end
    end
    # save data
    save(string("./data/reanalysis/times_",MS,".jld"), "t_data", t_data)
    save(string("./data/reanalysis/concentrations_",MS,"_rfu.jld"), "c1_rfu", c1_rfu)
    save(string("./data/reanalysis/lp_",MS,".jld"), "lp_data", lp_data)
    save(string("./data/reanalysis/times_",MS,"_long.jld"), "t_data_long", t_data_long)
    save(string("./data/reanalysis/concentrations_",MS,"_rfu_long.jld"), "c1_rfu_long", c1_rfu_long)
    save(string("./data/reanalysis/lp_",MS,"_long.jld"), "lp_data_long", lp_data_long)
end