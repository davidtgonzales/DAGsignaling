# load packages
using DataFrames
using CSV
using JLD
using Plots

# include functions
include("../functions/extract_data.jl")

# short traces laser power titration
LP_list = [0,0.1,0.2,0.5,1,5,10,20,40]
LP_label = ["00","01","02","05","1","5","10","20","40"]

# long traces laser power titration
LP_list_long = [10,40]
LP_label_long = ["10","40"]

# get data and averages
t_mean = zeros(4,30,9)
c1_mean = zeros(4,30,9)
t_mean_long = zeros(4,120,2)
c1_mean_long = zeros(4,120,2)
for (i,MS) in enumerate(["MS12","MS6","MS18","MS19"])
    t_data,c1_data,cells,lasers,dosetimes,set_label,set_id = load_dataset_reanalysis(MS,"short")
    t_data_long,c1_data_long,cells_long,lasers_long,dosetimes_long,set_label_long,set_id_long = load_dataset_reanalysis(MS,"long")
    for (j,LP) in enumerate(LP_list)
        # collect data at LP
        lp_idx = findall(x->x==LP, lasers)
        # threshold 500<=c1(5)<=1500
        upper_threshold_idx = findall(x->x<=1500,c1_data[5,:])
        lower_threshold_idx = findall(x->x>=500,c1_data[5,:])
        idx_intersect = intersect(lp_idx,upper_threshold_idx,lower_threshold_idx)
        # get mean values
        t_mean[i,:,j] = mean(t_data[:,idx_intersect],dims=2)
        c1_mean[i,:,j] = mean(c1_data[:,idx_intersect],dims=2)
    end
    for (j,LP) in enumerate(LP_list_long)
        # collect data at LP
        lp_idx = findall(x->x==LP, lasers_long)
        # threshold 500<=c1(5)<=1500
        upper_threshold_idx = findall(x->x<=1500,c1_data_long[5,:])
        lower_threshold_idx = findall(x->x>=500,c1_data_long[5,:])
        idx_intersect = intersect(lp_idx,upper_threshold_idx,lower_threshold_idx)
        # get mean values
        t_mean_long[i,:,j] = mean(t_data_long[:,idx_intersect],dims=2)
        c1_mean_long[i,:,j] = mean(c1_data_long[:,idx_intersect],dims=2)
    end
end

# bleach correction
bleach_correction = zeros(30,9)
for j in 1:9
    bleach_correction[:,j] = c1_mean[3,:,j]./c1_mean[3,1,j]
end
bleach_correction_long = zeros(120,2)
for j in 1:2
    bleach_correction_long[:,j] = c1_mean_long[3,:,j]./c1_mean_long[3,1,j]
end
c1_corr = zeros(4,30,9)
for i in 1:4
    for j in 1:9
        c1_corr[i,:,j] = c1_mean[i,:,j]./bleach_correction[:,j]
    end
end
c1_corr_long = zeros(4,120,2)
for i in 1:4
    for j in 1:2
        c1_corr_long[i,:,j] = c1_mean_long[i,:,j]./bleach_correction_long[:,j]
    end
end

# normalize
c1_norm = zeros(4,30,9)
for i in 1:4
    for j in 1:9
        c1_norm[i,:,j] = c1_corr[i,:,j]./c1_corr[i,5,j]
    end
end
c1_norm_long = zeros(4,120,2)
for i in 1:4
    for j in 1:2
        c1_norm_long[i,:,j] = c1_corr_long[i,:,j]./c1_corr_long[i,5,j]
    end
end

save(string("./data/reanalysis/t_mean.jld"), "t_mean", t_mean)
save(string("./data/reanalysis/c1_norm.jld"), "c1_norm", c1_norm)
save(string("./data/reanalysis/t_mean_long.jld"), "t_mean_long", t_mean_long)
save(string("./data/reanalysis/c1_norm_long.jld"), "c1_norm_long", c1_norm_long)
