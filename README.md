# Quantifying single cell lipid signaling kinetics after photo-stimulation
This repository contains code and data used for the paper 'Quantifying single cell lipid signaling kinetics after photo-stimulation' (add link or doi). The provided code can be used to reproduce the parameter inference and profile likelihood analysis shown in the paper and is not meant to be a general tool. For further information, contact Christoph Zechner (zechner@mpi-cbg.de) and André Nadler (nadler@mpi-cbg.de).

# Content
This repository contains the following directories:
1. raw - results from image analysis and calibration.
2. data - Julia data formatted (JLD) of raw data.
3. julia - Julia scripts to run analysis for simulated and experimental data.
4. out - output results of analysis of Julia scripts.
5. plot - output plots of Julia scripts.

For raw microscopy data, image analysis, and calibration, we refer you to the complete repository found in (add link or doi in Edmond).

# Installation
The provided code uses Julia (v1.7.2). The required packages are CSV, DataFrames, DifferentialEquations, Distributions, JLD, Optim, Plots, Plots.Measures, Random, and StatsPlots

# Usage
To re-run the analysis, copy all directories into one location and set this as the working directory. For the simulated datasets, run ./julia/simulated/singletrace_tests.jl to recreate Fig. 2B-E and Fig. S1. Use ./julia/simulated/generate_data.jl to recreate simulated datasets for maximum likelihood estimation and profile likelihood analysis. To run maximum likelihood estimation and profile likelihood analysis, use ./julia/simulated/nokout_mle.jl and ./julia/simulated/nokout_profiles.jl and ./julia/simulated/nokout_DAGprofiles.jl. Note that you can change the number of cells analyzed in these scripts to be 10, 50, 100, or 200 cells. To recreate the plots in Fig. 2F and Fig. S2-S4, run ./julia/simulated/kout_plot.jl. For the same analysis with the DAG model with kout, use the scripts with the prefix “nokout_”. All output results and plots are saved in ./output/simulated/ and ./plots/simulated/, respectively.

For the experimental datasets, use ./julia/experiment/pop_data.jl to convert the Matlab file ./raw/DAG_traces.mat containing the single cell traces of uncaging experiments to JLD format. Use ./julia/experiment/pop_multistart.jl to run multistart analysis. The number of trials or multistarts is currently set at 20. Next, run MLE using ./julia/experiment/pop_mle.jl, which uses the best initial condition from the multistart analysis. Run the scripts ./julia/experiment/pop_profile.jl and ./julia/experiment/pop_DAGprofile.jl for profile likelihood analysis. This analysis pipeline should be run for each DAG species. Each script asks for an index to specify the DAG species to be analyzed. The index for each DAG species are:

- 2 cgDMG
- 5 cgSAG
- 6 cgDArG
- 7 cgSLG
- 9 cgOSG
- 11 cgSOG

In addition, scripts for profile likelihood also asks for an additional arguement to specify which parameter should be analyzed. For ./julia/experiment/pop_profile.jl, the indices are: 

- 1 kin
- 2 kmet
- 3 Kd
- 4 sd

For ./julia/experiment/pop_DAGprofile.jl, the indices are the initial uncaged DAG of each cell in the population. Scripts for profile likelihood analysis take several hours (> 5 hours) as it is scanning a parameter range and repeatedly running MLE. The scripts ./julia/experiment/pop_plot_multistart.jl and ./julia/experiment/pop_plot.jl will recreate Fig. 5, Fig. S5-S12, and Fig. S15. Use ./julia/experiment/residual_analysis.jl to run the residual analysis in Fig. S18.

Use ./julia/experiment/mean_multistart.jl and ./julia/experiment/mean_fit.jl to repeat analysis but with population-averaged data. This will recreate the plots in Fig. S13 and S17. The C1-corrected analysis in Fig. S14 and S17 can be recreated using ./julia/experiment/mean_fit_corrected.jl. All output results and plots are saved in ./output/experiment/ and ./plots/experiment/, respectively. 
