# load packages
using DifferentialEquations
using Distributions
using Random
using Optim

##################################################
# functions for optimization on individual cells #
##################################################
function loglike(data, y, sd)
    logpdf_arr = fill(0.0,length(data))
    for (idx,i) in enumerate(y)
        dist = Normal(i,sd)
        logpdf_arr[idx] = logpdf(dist,data[idx])
    end
    return -sum(logpdf_arr)
end

function opt_mle(fep0,t_data,c1_data,dosetimes)
    if length(fep0) == 5 # model without kout (kin,kmet,Kd,sd,DAG)
        res_fep = optimize(fep->get_ll(fep,t_data,c1_data,dosetimes),fep0)
    elseif length(fep0) == 6 # model with kout (kin,kout,kmet,Kd,sd,DAG)
        res_fep = optimize(fep->get_ll_kout(fep,t_data,c1_data,dosetimes),fep0)
    end
    return Optim.minimum(res_fep), Optim.minimizer(res_fep)
end

function get_ll(fep,t_data,c1_data,dosetimes)
    efep = exp.(fep)
    c_init = mean(c1_data[1:5])  # average the first 5 timepoints to get initial C1
    params = vcat(efep[1:3],c_init)
    tspan = (0.0,t_data[end])
    u0 = [0.0,c_init]
    prob = ODEProblem(model,u0,tspan,params)
    cb = PresetTimeCallback(dosetimes,integrator->integrator.u[1] = integrator.u[1] + efep[4])
    alg = Rodas4()
    sol = solve(prob, alg, callback=cb, reltol=1e-6)
    c_loglike = loglike(c1_data,sol(t_data,idxs=2).u,efep[end])
    return sum(c_loglike)
end

function get_ll_kout(fep,t_data,c1_data,dosetimes)
    efep = exp.(fep)
    c_init = mean(c1_data[1:5])  # average the first 5 timepoints to get initial C1
    params = vcat(efep[1:4],c_init)
    tspan = (0.0,t_data[end])
    u0 = [0.0,c_init]
    prob = ODEProblem(model_kout,u0,tspan,params)
    cb = PresetTimeCallback(dosetimes,integrator->integrator.u[1] = integrator.u[1] + efep[5])
    alg = Rodas4()
    sol = solve(prob, alg, callback=cb, reltol=1e-6)
    c_loglike = loglike(c1_data,sol(t_data,idxs=2).u,efep[end])
    return sum(c_loglike)
end

################################################
# functions for optimization on multiple cells #
################################################
function objfun_re(rep,fep,t_i,c_i,dosetimes)
    efep = exp.(fep)
    c_init = mean(c_i[1:5]) # average the first 5 timepoints to get initial C1
    params = vcat(efep[1:3],c_init)
    tspan = (0.0,t_i[end])
    u0 = [0.0,c_init]
    prob = ODEProblem(model,u0,tspan,params)
    cb = PresetTimeCallback(dosetimes,integrator->integrator.u[1] = integrator.u[1] + rep[1])
    alg = Rodas4()
    sol = solve(prob, alg, callback=cb, reltol=1e-6, isoutofdomain=(u,p,t) -> any(x -> x < 0, u))
    c_loglike = loglike(c_i,sol(t_i,idxs=2).u,efep[end])
    return c_loglike
end

function opt_re(rep0,fep,t_i,c_i,dosetimes)
    res_re = optimize(rep->objfun_re(rep,fep,t_i,c_i,dosetimes),0,10) # range between 0-10uM uncaged DAG
    return Optim.minimum(res_re), Optim.minimizer(res_re)
end

function loglike_fe(fep,re_sub,cells,t_data,c_data,dosetimes)
    loglikes = zeros(Float64, cells)
    for i in 1:cells
        rep = re_sub[i,:]
        efep = exp.(fep)
        c_init = mean(c_data[1:5,i])  # average the first 5 timepoints to get initial C1
        params = vcat(efep[1:3],c_init)
        tspan = (0.0,t_data[end,i])
        u0 = [0.0,c_init]
        prob = ODEProblem(model,u0,tspan,params)
        cb = PresetTimeCallback(dosetimes,integrator->integrator.u[1] = integrator.u[1] + rep[1])
        alg = Rodas4()
        sol = solve(prob, alg, callback=cb, reltol=1e-6, isoutofdomain=(u,p,t) -> any(x -> x < 0, u))
        c_loglike = loglike(c_data[:,i],sol(t_data[:,i],idxs=2).u,efep[end])
        loglikes[i] = c_loglike
    end
    return sum(loglikes)
end

function opt_fe(fep0,re_sub,cells,t_data,c_data,dosetimes)
    res_fep = optimize(fep->loglike_fe(fep,re_sub,cells,t_data,c_data,dosetimes),fep0)
    return Optim.minimum(res_fep), Optim.minimizer(res_fep)
end

function re_iteration(re_init, fek, cells, t_data, c_data, dosetimes)
    re_sub = zeros(cells,size(re_init)[2])
    for i in 1:cells
        rei = re_init[i,:]
        t_i = t_data[:,i]
        c_i = c_data[:,i]
        opt_out = opt_re(rei,fek,t_i,c_i,dosetimes)
        re_sub[i,:] .= opt_out[2]
    end
    return re_sub
end

function objfcn_r(r, t_data, c1_data, cells, c1_recruit, dosetimes, p_init)
    iters = 10
    res = 1 # DAGext (random params)
    fes = 4 # kin,kmet,Kd,sd (fixed params)
    re_arr = ones(Float64, iters, cells, res)
    re_arr[1,:,:] = c1_recruit/r # initialize uncaged DAG using r
    fe_arr = ones(Float64, iters, fes)
    obj_arr = zeros(Float64, iters)
    fe_arr[1,:] = log.(p_init)
    display("Starting new iteration:")
    for m in 1:iters-1
        display(m)
        opt_out = opt_fe(fe_arr[m,:],re_arr[m,:,:],cells,t_data,c1_data,dosetimes)
        obj_arr[m+1] = opt_out[1]
        fe_res = opt_out[2]
        fe_arr[m+1,:] = fe_res
        re_res = re_iteration(re_arr[m,:,:],fe_arr[m+1,:],cells,t_data,c1_data,dosetimes)
        re_arr[m+1,:,:] = re_res
    end
    display("Finished.")
    obj = obj_arr[end]
    return obj
end

function optimized_r(r, t_data, c1_data, cells, c1_recruit, dosetimes, p_init)
    iters = 10
    res = 1 # DAGext (random params)
    fes = 4 # kin,kmet,Kd,sd (fixed params)
    re_arr = ones(Float64, iters, cells, res)
    re_arr[1,:,:] = c1_recruit/r[1] # initialize uncaged DAG using r
    fe_arr = ones(Float64, iters, fes)
    obj_arr = zeros(Float64, iters)
    fe_arr[1,:] = log.(p_init)
    for m in 1:iters-1
        opt_out = opt_fe(fe_arr[m,:],re_arr[m,:,:],cells,t_data,c1_data,dosetimes)
        obj_arr[m+1] = opt_out[1]
        fe_res = opt_out[2]
        fe_arr[m+1,:] = fe_res
        re_res = re_iteration(re_arr[m,:,:],fe_arr[m+1,:],cells,t_data,c1_data,dosetimes)
        re_arr[m+1,:,:] = re_res
        fe_arr[m+1,:] = fe_res
    end
    r_params = re_arr
    f_params = fe_arr
    obj = obj_arr
    return r_params, f_params, obj
end

##########################################################
# functions for optimization on multiple cells with kout #
##########################################################
function objfun_re_kout(rep,fep,t_i,c_i,dosetimes)
    efep = exp.(fep)
    c_init = mean(c_i[1:5]) # average the first 5 timepoints to get initial C1
    params = vcat(efep[1:4],c_init)
    tspan = (0.0,t_i[end])
    u0 = [0.0,c_init]
    prob = ODEProblem(model_kout,u0,tspan,params)
    cb = PresetTimeCallback(dosetimes,integrator->integrator.u[1] = integrator.u[1] + rep[1])
    alg = Rodas4()
    sol = solve(prob, alg, callback=cb, reltol=1e-6, isoutofdomain=(u,p,t) -> any(x -> x < 0, u))
    c_loglike = loglike(c_i,sol(t_i,idxs=2).u,efep[end])
    return c_loglike
end

function opt_re_kout(rep0,fep,t_i,c_i,dosetimes)
    res_re = optimize(rep->objfun_re_kout(rep,fep,t_i,c_i,dosetimes),0,10) # range between 0-10uM uncaged DAG
    return Optim.minimum(res_re), Optim.minimizer(res_re)
end

function loglike_fe_kout(fep,re_sub,cells,t_data,c_data,dosetimes)
    loglikes = zeros(Float64, cells)
    for i in 1:cells
        rep = re_sub[i,:]
        efep = exp.(fep)
        c_init = mean(c_data[1:5,i])  # average the first 5 timepoints to get initial C1
        params = vcat(efep[1:4],c_init)
        tspan = (0.0,t_data[end,i])
        u0 = [0.0,c_init]
        prob = ODEProblem(model_kout,u0,tspan,params)
        cb = PresetTimeCallback(dosetimes,integrator->integrator.u[1] = integrator.u[1] + rep[1])
        alg = Rodas4()
        sol = solve(prob, alg, callback=cb, reltol=1e-6, isoutofdomain=(u,p,t) -> any(x -> x < 0, u))
        c_loglike = loglike(c_data[:,i],sol(t_data[:,i],idxs=2).u,efep[end])
        loglikes[i] = c_loglike
    end
    return sum(loglikes)
end

function opt_fe_kout(fep0,re_sub,cells,t_data,c_data,dosetimes)
    res_fep = optimize(fep->loglike_fe_kout(fep,re_sub,cells,t_data,c_data,dosetimes),fep0)
    return Optim.minimum(res_fep), Optim.minimizer(res_fep)
end

function re_iteration_kout(re_init, fek, cells, t_data, c_data, dosetimes)
    re_sub = zeros(cells,size(re_init)[2])
    for i in 1:cells
        rei = re_init[i,:]
        t_i = t_data[:,i]
        c_i = c_data[:,i]
        opt_out = opt_re_kout(rei,fek,t_i,c_i,dosetimes)
        re_sub[i,:] .= opt_out[2]
    end
    return re_sub
end

function objfcn_r_kout(r, t_data, c1_data, cells, c1_recruit, dosetimes, p_init)
    iters = 10
    res = 1 # DAGext (random params)
    fes = 5 # kin,kout,kmet,Kd,sd (fixed params)
    re_arr = ones(Float64, iters, cells, res)
    re_arr[1,:,:] = c1_recruit/r # initialize uncaged DAG using r
    fe_arr = ones(Float64, iters, fes)
    obj_arr = zeros(Float64, iters)
    fe_arr[1,:] = log.(p_init)
    display("Starting new iteration:")
    for m in 1:iters-1
        display(m)
        opt_out = opt_fe_kout(fe_arr[m,:],re_arr[m,:,:],cells,t_data,c1_data,dosetimes)
        obj_arr[m+1] = opt_out[1]
        fe_res = opt_out[2]
        fe_arr[m+1,:] = fe_res
        re_res = re_iteration_kout(re_arr[m,:,:],fe_arr[m+1,:],cells,t_data,c1_data,dosetimes)
        re_arr[m+1,:,:] = re_res
    end
    display("Finished.")
    obj = obj_arr[end]
    return obj
end

function optimized_r_kout(r, t_data, c1_data, cells, c1_recruit, dosetimes, p_init)
    iters = 10
    res = 1 # DAGext (random params)
    fes = 5 # kin,kout,kmet,Kd,sd (fixed params)
    re_arr = ones(Float64, iters, cells, res)
    re_arr[1,:,:] = c1_recruit/r[1] # initialize uncaged DAG using r
    fe_arr = ones(Float64, iters, fes)
    obj_arr = zeros(Float64, iters)
    fe_arr[1,:] = log.(p_init)
    for m in 1:iters-1
        opt_out = opt_fe_kout(fe_arr[m,:],re_arr[m,:,:],cells,t_data,c1_data,dosetimes)
        obj_arr[m+1] = opt_out[1]
        fe_res = opt_out[2]
        fe_arr[m+1,:] = fe_res
        re_res = re_iteration_kout(re_arr[m,:,:],fe_arr[m+1,:],cells,t_data,c1_data,dosetimes)
        re_arr[m+1,:,:] = re_res
        fe_arr[m+1,:] = fe_res
    end
    r_params = re_arr
    f_params = fe_arr
    obj = obj_arr
    return r_params, f_params, obj
end

####################################################################
# functions for optimization on population averaged data (no kout) #
####################################################################
function opt_mle_mean(fep0,t_data,c1_data,dag,dosetimes)
    res_fep = optimize(fep->get_ll_mean(fep,t_data,c1_data,dag,dosetimes),fep0)
    return Optim.minimum(res_fep), Optim.minimizer(res_fep)
end

function get_ll_mean(fep,t_data,c1_data,dag,dosetimes)
    efep = exp.(fep)
    c_init = mean(c1_data[1:5])  # average the first 5 timepoints to get initial C1
    params = vcat(efep[1:3],c_init)
    tspan = (0.0,t_data[end])
    u0 = [0.0,c_init]
    prob = ODEProblem(model,u0,tspan,params)
    cb = PresetTimeCallback(dosetimes,integrator->integrator.u[1] = integrator.u[1] + dag)
    alg = Rodas4()
    sol = solve(prob, alg, callback=cb, reltol=1e-6)
    c_loglike = loglike(c1_data,sol(t_data,idxs=2).u,efep[end])
    return sum(c_loglike)
end

function get_pl_mean(fdx,fep,t_data,c1_data,dag,dosetimes)
    n = 11
    lower = log(exp(fep[fdx])/10)
    upper = log(exp(fep[fdx])*10)
    pj_arr = range(lower,upper, length = n) |> collect
    pl_arr = zeros(n)
    pf_arr = zeros(n,length(fep)-1)
    for (jdx, pj) in enumerate(pj_arr)
         pl_arr[jdx],pf_arr[jdx,:] = opt_pl_mean(pj,fdx,fep,t_data,c1_data,dag,dosetimes)
    end
    return pj_arr, pf_arr, pl_arr
end

function opt_pl_mean(pj,fdx,fep,t_data,c1_data,dag,dosetimes)
    fep0 = fep[1:end .!=fdx]
    res_fep = optimize(fep->loglike_pl_mean(fep,t_data,c1_data,pj,fdx,dag,dosetimes),fep0)
    return Optim.minimum(res_fep), Optim.minimizer(res_fep)
end

function loglike_pl_mean(fep,t_data,c1_data,pj,fdx,dag,dosetimes)
    # kin, kmet, Kd, sigma
    fep_temp = fep[1:end]
    efep = exp.(insert!(fep_temp,fdx,pj))
    # kin,kmet,Kd,C1_i
    params = vcat(efep[1:3],c1_data[1])
    tspan = (0.0,t_data[end])
    u0 = [0.0,c1_data[1]]
    prob = ODEProblem(model,u0,tspan,params)
    cb = PresetTimeCallback(dosetimes,integrator->integrator.u[1] = integrator.u[1] + dag)
    alg = Rodas4()
    sol = solve(prob, alg, callback=cb, reltol=1e-6)
    c_loglike = loglike(c1_data,sol(t_data,idxs=2).u,efep[end])
    return sum(c_loglike)
end