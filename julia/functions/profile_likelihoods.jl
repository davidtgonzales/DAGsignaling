# load packages
using DifferentialEquations
using Distributions
using Random
using Optim

############################################################################
# functions for profile likelihoods on individual cells without kout model #
############################################################################
function get_pl(fdx,fep,t_data,c1_data,dosetimes)
    n = 11
    lower = log(exp(fep[fdx])/10)
    upper = log(exp(fep[fdx])*10)
    pj_arr = range(lower,upper, length = n) |> collect
    pl_arr = zeros(n)
    pf_arr = zeros(n,length(fep)-1)
    for (jdx, pj) in enumerate(pj_arr)
         pl_arr[jdx],pf_arr[jdx,:] = opt_pl(pj,fdx,fep,t_data,c1_data,dosetimes)
    end
    return pj_arr, pf_arr, pl_arr
end

function loglike_pl(fep,t_data,c1_data,pj,fdx,dosetimes)
    fep_temp = fep[1:end] # kin,kmet,Kd,sd
    efep = exp.(insert!(fep_temp,fdx,pj))
    params = vcat(efep[1:3],c1_data[1]) # kin,kmet,Kd,C1_i
    tspan = (0.0,t_data[end])
    u0 = [0.0,c1_data[1]]
    prob = ODEProblem(model,u0,tspan,params)
    cb = PresetTimeCallback(dosetimes,integrator->integrator.u[1] = integrator.u[1] + efep[4])
    alg = Rodas4()
    sol = solve(prob, alg, callback=cb, reltol=1e-6)
    c_loglike = loglike(c1_data,sol(t_data,idxs=2).u,efep[end])
    return sum(c_loglike)
end

function opt_pl(pj,fdx,fep,t_data,c1_data,dosetimes)
    fep0 = fep[1:end .!=fdx]
    if length(fep) == 5
        res_fep = optimize(fep->loglike_pl(fep,t_data,c1_data,pj,fdx,dosetimes),fep0)
    elseif length(fep) == 6
        res_fep = optimize(fep->loglike_pl_kout(fep,t_data,c1_data,pj,fdx,dosetimes),fep0)
    end
    return Optim.minimum(res_fep), Optim.minimizer(res_fep)
end

#########################################################################
# functions for profile likelihoods on individual cells with kout model #
#########################################################################
function loglike_pl_kout(fep,t_data,c1_data,pj,fdx,dosetimes)
    fep_temp = fep[1:end] # kin,kout,kmet,Kd,sd
    efep = exp.(insert!(fep_temp,fdx,pj))
    params = vcat(efep[1:4],c1_data[1]) # kin,kout,kmet,Kd,C1_i
    tspan = (0.0,t_data[end])
    u0 = [0.0,c1_data[1]]
    prob = ODEProblem(model_kout,u0,tspan,params)
    cb = PresetTimeCallback(dosetimes,integrator->integrator.u[1] = integrator.u[1] + efep[5])
    alg = Rodas4()
    sol = solve(prob, alg, callback=cb, reltol=1e-6)
    c_loglike = loglike(c1_data,sol(t_data,idxs=2).u,efep[end])
    return sum(c_loglike)
end

##########################################################################
# functions for profile likelihoods on multiple cells without kout model #
##########################################################################
function get_fixedpl(fdx,fep,cells,t_data,c1_data,lower_r,upper_r,c1_recruit,dosetimes,p_init)
    n = 11
    lower = log(exp(fep[fdx])/15)
    upper = log(exp(fep[fdx])*15)
    pj_arr = range(lower,upper, length = n) |> collect
    pl_arr = zeros(n)
    pf_arr = zeros(n,4)
    pr_arr = zeros(n,cells,1)
    display("Starting new profile:")
    for (jdx, pj) in enumerate(pj_arr)
        display(jdx)
        res_r = optimize(r->objfcn_pl_r(r,pj,fdx,t_data,c1_data,cells,c1_recruit,dosetimes,p_init),lower_r,upper_r,abs_tol=0.01,rel_tol=0.001)
        r_opt = Optim.minimizer(res_r)
        r_params, f_params, obj = optimized_pl_r(r_opt, pj, fdx, t_data, c1_data, cells, c1_recruit, dosetimes, p_init)
        pl_arr[jdx] = obj[end]
        pf_arr[jdx,:] = f_params[end,:]
        pr_arr[jdx,:] = r_params[end,:,:]
    end
    display("Finished.")
    return pl_arr, pj_arr, pr_arr, pf_arr
end

function objfcn_pl_r(r, pj, fdx, t_data, c1_data, cells, c1_recruit,dosetimes, p_init)
    iters = 10
    res = 1 # DAGext (random params)
    fes = 4 # kin,kmet,Kd,sd (fixed params)
    re_arr = ones(Float64, iters, cells, res)
    re_arr[1,:,:] = c1_recruit/r # initialize uncaged DAG using r
    fe_arr = ones(Float64, iters, fes)
    obj_arr = zeros(Float64, iters)
    fep = log.(p_init)
    fep_temp = fep[1:end .!=fdx]
    fe_arr[1,:] = insert!(fep_temp,fdx,pj)
    for m in 1:iters-1
        opt_out = opt_fepl(fe_arr[m,1:end .!=fdx],re_arr[m,:,:],cells,t_data,c1_data,dosetimes,pj,fdx)
        obj_arr[m+1] = opt_out[1]
        fe_res = opt_out[2]
        fe_arr[m+1,:] = insert!(fe_res,fdx,pj)
        re_res = re_iteration(re_arr[m,:,:],fe_arr[m+1,:],cells,t_data,c1_data,dosetimes)
        re_arr[m+1,:,:] = re_res
    end
    obj = obj_arr[end]
    return obj
end

function optimized_pl_r(r, pj, fdx, t_data, c1_data, cells, c1_recruit,dosetimes,p_init)
    iters = 10
    res = 1 # DAGext (random params)
    fes = 4 # kin,kmet,Kd,sd (fixed params)
    re_arr = ones(Float64, iters, cells, res)
    re_arr[1,:,:] = c1_recruit/r # initialize uncaged DAG using r
    fe_arr = ones(Float64, iters, fes)
    obj_arr = zeros(Float64, iters)
    fep = log.(p_init)
    fep_temp = fep[1:end .!=fdx]
    fe_arr[1,:] = insert!(fep_temp,fdx,pj)
    for m in 1:iters-1
        opt_out = opt_fepl(fe_arr[m,1:end .!=fdx],re_arr[m,:,:],cells,t_data,c1_data,dosetimes,pj,fdx)
        obj_arr[m+1] = opt_out[1]
        fe_res = opt_out[2]
        fe_arr[m+1,:] = insert!(fe_res,fdx,pj)
        re_res = re_iteration(re_arr[m,:,:],fe_arr[m+1,:],cells,t_data,c1_data,dosetimes)
        re_arr[m+1,:,:] = re_res
    end
    r_params = re_arr
    f_params = fe_arr
    obj = obj_arr
    return r_params, f_params, obj
end

function opt_fepl(fep0,re_sub,cells,t_data,c1_data,dosetimes,pj,fdx)
    res_fep = optimize(fep->loglike_fepl(fep,re_sub,cells,t_data,c1_data,dosetimes,pj,fdx),fep0)
    return Optim.minimum(res_fep), Optim.minimizer(res_fep)
end

function loglike_fepl(fep,re_sub,cells,t_data,c_data,dosetimes,pj,fdx)
    loglikes = zeros(Float64, cells)
    for i in 1:cells
        rep = re_sub[i,:]
        fep_temp = fep[1:end]
        efep = exp.(insert!(fep_temp,fdx,pj))
        t_i = t_data[:,i]
        c_i = c_data[:,i]
        c_init = mean(c_i[1:5])
        params = vcat(efep[1:3],c_init)
        tspan = (0.0,t_i[end])
        u0 = [0.0,c_init]
        prob = ODEProblem(model,u0,tspan,params)
        cb = PresetTimeCallback(dosetimes,integrator->integrator.u[1] = integrator.u[1] + rep[1])
        alg = Rodas4()
        sol = solve(prob, alg, callback=cb, reltol=1e-6, isoutofdomain=(u,p,t) -> any(x -> x < 0, u))
        c_loglike = loglike(c_i,sol(t_i,idxs=2).u,efep[end])
        loglikes[i] = c_loglike
    end
    return sum(loglikes)
end

##########################################################
# functions for optimization on multiple cells with kout #
##########################################################
function get_fixedpl_kout(fdx,fep,cells,t_data,c1_data,lower_r,upper_r,c1_recruit,dosetimes,p_init)
    n = 11
    lower = log(exp(fep[fdx])/10)
    upper = log(exp(fep[fdx])*10)
    pj_arr = range(lower,upper, length = n) |> collect
    pl_arr = zeros(n)
    pf_arr = zeros(n,5)
    pr_arr = zeros(n,cells,1)
    display("Starting new profile:")
    for (jdx, pj) in enumerate(pj_arr)
        display(jdx)
        res_r = optimize(r->objfcn_pl_r_kout(r,pj,fdx,t_data,c1_data,cells,c1_recruit,dosetimes,p_init),lower_r,upper_r,abs_tol=0.01,rel_tol=0.001)
        r_opt = Optim.minimizer(res_r)
        r_params, f_params, obj = optimized_pl_r_kout(r_opt, pj, fdx, t_data, c1_data, cells, c1_recruit, dosetimes, p_init)
        pl_arr[jdx] = obj[end]
        pf_arr[jdx,:] = f_params[end,:]
        pr_arr[jdx,:] = r_params[end,:,:]
    end
    display("Finished.")
    return pl_arr, pj_arr, pr_arr, pf_arr
end

function objfcn_pl_r_kout(r, pj, fdx, t_data, c1_data, cells, c1_recruit,dosetimes, p_init)
    iters = 10
    res = 1 # DAGext (random params)
    fes = 5 # kin,kout,kmet,Kd,sd (fixed params)
    re_arr = ones(Float64, iters, cells, res)
    re_arr[1,:,:] = c1_recruit/r # initialize uncaged DAG using r
    fe_arr = ones(Float64, iters, fes)
    obj_arr = zeros(Float64, iters)
    fep = log.(p_init)
    fep_temp = fep[1:end .!=fdx]
    fe_arr[1,:] = insert!(fep_temp,fdx,pj)
    for m in 1:iters-1
        opt_out = opt_fepl_kout(fe_arr[m,1:end .!=fdx],re_arr[m,:,:],cells,t_data,c1_data,dosetimes,pj,fdx)
        obj_arr[m+1] = opt_out[1]
        fe_res = opt_out[2]
        fe_arr[m+1,:] = insert!(fe_res,fdx,pj)
        re_res = re_iteration_kout(re_arr[m,:,:],fe_arr[m+1,:],cells,t_data,c1_data,dosetimes)
        re_arr[m+1,:,:] = re_res
    end
    obj = obj_arr[end]
    return obj
end

function optimized_pl_r_kout(r, pj, fdx, t_data, c1_data, cells, c1_recruit,dosetimes,p_init)
    iters = 10
    res = 1 # DAGext (random params)
    fes = 5 # kin,kout,kmet,Kd,sd (fixed params)
    re_arr = ones(Float64, iters, cells, res)
    re_arr[1,:,:] = c1_recruit/r # initialize uncaged DAG using r
    fe_arr = ones(Float64, iters, fes)
    obj_arr = zeros(Float64, iters)
    fep = log.(p_init)
    fep_temp = fep[1:end .!=fdx]
    fe_arr[1,:] = insert!(fep_temp,fdx,pj)
    for m in 1:iters-1
        opt_out = opt_fepl_kout(fe_arr[m,1:end .!=fdx],re_arr[m,:,:],cells,t_data,c1_data,dosetimes,pj,fdx)
        obj_arr[m+1] = opt_out[1]
        fe_res = opt_out[2]
        fe_arr[m+1,:] = insert!(fe_res,fdx,pj)
        re_res = re_iteration_kout(re_arr[m,:,:],fe_arr[m+1,:],cells,t_data,c1_data,dosetimes)
        re_arr[m+1,:,:] = re_res
    end
    r_params = re_arr
    f_params = fe_arr
    obj = obj_arr
    return r_params, f_params, obj
end

function opt_fepl_kout(fep0,re_sub,cells,t_data,c1_data,dosetimes,pj,fdx)
    res_fep = optimize(fep->loglike_fepl_kout(fep,re_sub,cells,t_data,c1_data,dosetimes,pj,fdx),fep0)
    return Optim.minimum(res_fep), Optim.minimizer(res_fep)
end

function loglike_fepl_kout(fep,re_sub,cells,t_data,c_data,dosetimes,pj,fdx)
    loglikes = zeros(Float64, cells)
    for i in 1:cells
        rep = re_sub[i,:]
        fep_temp = fep[1:end]
        efep = exp.(insert!(fep_temp,fdx,pj))
        t_i = t_data[:,i]
        c_i = c_data[:,i]
        c_init = mean(c_i[1:5])
        params = vcat(efep[1:4],c_init)
        tspan = (0.0,t_i[end])
        u0 = [0.0,c_init]
        prob = ODEProblem(model_kout,u0,tspan,params)
        cb = PresetTimeCallback(dosetimes,integrator->integrator.u[1] = integrator.u[1] + rep[1])
        alg = Rodas4()
        sol = solve(prob, alg, callback=cb, reltol=1e-6, isoutofdomain=(u,p,t) -> any(x -> x < 0, u))
        c_loglike = loglike(c_i,sol(t_i,idxs=2).u,efep[end])
        loglikes[i] = c_loglike
    end
    return sum(loglikes)
end

#####################################################################
# functions for profile likelihoods of DAG using model without kout #
#####################################################################
function get_pl_DAG(fdx,dag,cells,t_data,c1_data,lower_r,upper_r,c1_recruit,dosetimes,p_init)
    n = 11
    lower = log(dag/10)
    upper = log(dag*10)
    pj_arr = exp.(range(lower,upper, length = n) |> collect)
    pl_arr = zeros(n)
    pf_arr = zeros(n,4)
    pr_arr = zeros(n,cells,1)
    display("Starting new profile:")
    for (jdx, pj) in enumerate(pj_arr)
        display(jdx)
        res_r = optimize(r->objfcn_DAGpl_r(r,pj,fdx,t_data,c1_data,cells,c1_recruit,dosetimes,p_init),lower_r,upper_r,abs_tol=0.01,rel_tol=0.001)
        r_opt = Optim.minimizer(res_r)
        r_params, f_params, obj = optimized_DAGpl_r(r_opt, pj, fdx, t_data, c1_data, cells, c1_recruit, dosetimes, p_init)
        pl_arr[jdx] = obj[end]
        pf_arr[jdx,:] = f_params[end,:]
        pr_arr[jdx,:] = r_params[end,:,:]
    end
    display("Finished.")
    return pl_arr, pj_arr, pr_arr, pf_arr
end


function objfcn_DAGpl_r(r, pj, fdx, t_data, c1_data, cells, c1_recruit,dosetimes, p_init)
    iters = 10
    res = 1 # DAGext (random params)
    fes = 4 # kin,kmet,Kd,sd (fixed params)
    re_arr = ones(Float64, iters, cells, res)
    re_arr[1,:,:] = c1_recruit/r # initialize uncaged DAG using r
    re_arr[:,fdx,1] .= pj
    fe_arr = ones(Float64, iters, fes)
    obj_arr = zeros(Float64, iters)
    fe_arr[1,:] = log.(p_init)
    for m in 1:iters-1
        opt_out = opt_fepl_DAG(fe_arr[m,1:end],re_arr[m,:,:],cells,t_data,c1_data,dosetimes)
        obj_arr[m+1] = opt_out[1]
        fe_res = opt_out[2]
        fe_arr[m+1,:] = fe_res
        re_res = re_iteration_DAG(re_arr[m,:,:],fe_arr[m+1,:],cells,t_data,c1_data,dosetimes,fdx,pj)
        re_arr[m+1,:,:] = re_res
    end
    obj = obj_arr[end]
    return obj
end

function optimized_DAGpl_r(r, pj, fdx, t_data, c1_data, cells, c1_recruit,dosetimes,p_init)
    iters = 10
    res = 1 # DAGext (random params)
    fes = 4 # kin,kmet,Kd,sd (fixed params)
    re_arr = ones(Float64, iters, cells, res)
    re_arr[1,:,:] = c1_recruit/r # initialize uncaged DAG using r
    re_arr[:,fdx,1] .= pj
    fe_arr = ones(Float64, iters, fes)
    obj_arr = zeros(Float64, iters)
    fe_arr[1,:] = log.(p_init)
    for m in 1:iters-1
        opt_out = opt_fepl_DAG(fe_arr[m,1:end],re_arr[m,:,:],cells,t_data,c1_data,dosetimes)
        obj_arr[m+1] = opt_out[1]
        fe_res = opt_out[2]
        fe_arr[m+1,:] = fe_res
        re_res = re_iteration_DAG(re_arr[m,:,:],fe_arr[m+1,:],cells,t_data,c1_data,dosetimes,fdx,pj)
        re_arr[m+1,:,:] = re_res
    end
    r_params = re_arr
    f_params = fe_arr
    obj = obj_arr
    return r_params, f_params, obj
end

function opt_fepl_DAG(fep0,re_sub,cells,t_data,c1_data,dosetimes)
    res_fep = optimize(fep->loglike_fepl_DAG(fep,re_sub,cells,t_data,c1_data,dosetimes),fep0)
    return Optim.minimum(res_fep), Optim.minimizer(res_fep)
end

function loglike_fepl_DAG(fep,re_sub,cells,t_data,c_data,dosetimes)
    loglikes = zeros(Float64, cells)
    for i in 1:cells
        rep = re_sub[i,:]
        efep = exp.(fep[1:end])
        t_i = t_data[:,i]
        c_i = c_data[:,i]
        c_init = mean(c_i[1:5])
        params = vcat(efep[1:3],c_init)
        tspan = (0.0,t_i[end])
        u0 = [0.0,c_init]
        prob = ODEProblem(model,u0,tspan,params)
        cb = PresetTimeCallback(dosetimes,integrator->integrator.u[1] = integrator.u[1] + rep[1])
        alg = Rodas4()
        sol = solve(prob, alg, callback=cb, reltol=1e-6, isoutofdomain=(u,p,t) -> any(x -> x < 0, u))
        c_loglike = loglike(c_i,sol(t_i,idxs=2).u,efep[end])
        loglikes[i] = c_loglike
    end
    return sum(loglikes)
end

function re_iteration_DAG(re_init, fek, cells, t_data, c_data, dosetimes,fdx,pj)
    re_sub = zeros(cells,size(re_init)[2])
    for i in 1:cells
        if i == fdx
            re_sub[i,1] = pj 
        else
            rei = re_init[i,:]
            t_i = t_data[:,i]
            c_i = c_data[:,i]
            opt_out = opt_re_DAG(rei,fek,t_i,c_i,dosetimes)
            re_sub[i,:] .= opt_out[2]
        end
    end
    return re_sub
end

function opt_re_DAG(rep0,fep,t_i,c_i,dosetimes)
    res_re = optimize(rep->objfun_re_DAG(rep,fep,t_i,c_i,dosetimes),0,10)
    return Optim.minimum(res_re), Optim.minimizer(res_re)
end

function objfun_re_DAG(rep,fep,t_i,c_i,dosetimes)
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

##################################################################
# functions for profile likelihoods of DAG using model with kout #
##################################################################
function get_pl_DAG_kout(fdx,dag,cells,t_data,c1_data,lower_r,upper_r,c1_recruit,dosetimes,p_init)
    n = 11
    lower = log(dag/10)
    upper = log(dag*10)
    pj_arr = exp.(range(lower,upper, length = n) |> collect)
    pl_arr = zeros(n)
    pf_arr = zeros(n,5)
    pr_arr = zeros(n,cells,1)
    display("Starting new profile:")
    for (jdx, pj) in enumerate(pj_arr)
        display(jdx)
        res_r = optimize(r->objfcn_DAGpl_r_kout(r,pj,fdx,t_data,c1_data,cells,c1_recruit,dosetimes,p_init),lower_r,upper_r,abs_tol=0.01,rel_tol=0.001)
        r_opt = Optim.minimizer(res_r)
        r_params, f_params, obj = optimized_DAGpl_r_kout(r_opt, pj, fdx, t_data, c1_data, cells, c1_recruit, dosetimes, p_init)
        pl_arr[jdx] = obj[end]
        pf_arr[jdx,:] = f_params[end,:]
        pr_arr[jdx,:] = r_params[end,:,:]
    end
    display("Finished.")
    return pl_arr, pj_arr, pr_arr, pf_arr
end


function objfcn_DAGpl_r_kout(r, pj, fdx, t_data, c1_data, cells, c1_recruit,dosetimes, p_init)
    iters = 10
    res = 1 # DAGext (random params)
    fes = 5 # kin,kout,kmet,Kd,sd (fixed params)
    re_arr = ones(Float64, iters, cells, res)
    re_arr[1,:,:] = c1_recruit/r # initialize uncaged DAG using r
    re_arr[:,fdx,1] .= pj
    fe_arr = ones(Float64, iters, fes)
    obj_arr = zeros(Float64, iters)
    fe_arr[1,:] = log.(p_init)
    for m in 1:iters-1
        opt_out = opt_fepl_DAG_kout(fe_arr[m,1:end],re_arr[m,:,:],cells,t_data,c1_data,dosetimes)
        obj_arr[m+1] = opt_out[1]
        fe_res = opt_out[2]
        fe_arr[m+1,:] = fe_res
        re_res = re_iteration_DAG_kout(re_arr[m,:,:],fe_arr[m+1,:],cells,t_data,c1_data,dosetimes,fdx,pj)
        re_arr[m+1,:,:] = re_res
    end
    obj = obj_arr[end]
    return obj
end

function optimized_DAGpl_r_kout(r, pj, fdx, t_data, c1_data, cells, c1_recruit,dosetimes,p_init)
    iters = 10
    res = 1 # DAGext (random params)
    fes = 5 # kin,kout,kmet,Kd,sd (fixed params)
    re_arr = ones(Float64, iters, cells, res)
    re_arr[1,:,:] = c1_recruit/r # initialize uncaged DAG using r
    re_arr[:,fdx,1] .= pj
    fe_arr = ones(Float64, iters, fes)
    obj_arr = zeros(Float64, iters)
    fe_arr[1,:] = log.(p_init)
    for m in 1:iters-1
        opt_out = opt_fepl_DAG_kout(fe_arr[m,1:end],re_arr[m,:,:],cells,t_data,c1_data,dosetimes)
        obj_arr[m+1] = opt_out[1]
        fe_res = opt_out[2]
        fe_arr[m+1,:] = fe_res
        re_res = re_iteration_DAG_kout(re_arr[m,:,:],fe_arr[m+1,:],cells,t_data,c1_data,dosetimes,fdx,pj)
        re_arr[m+1,:,:] = re_res
    end
    r_params = re_arr
    f_params = fe_arr
    obj = obj_arr
    return r_params, f_params, obj
end

function opt_fepl_DAG_kout(fep0,re_sub,cells,t_data,c1_data,dosetimes)
    res_fep = optimize(fep->loglike_fepl_DAG_kout(fep,re_sub,cells,t_data,c1_data,dosetimes),fep0)
    return Optim.minimum(res_fep), Optim.minimizer(res_fep)
end

function loglike_fepl_DAG_kout(fep,re_sub,cells,t_data,c_data,dosetimes)
    loglikes = zeros(Float64, cells)
    for i in 1:cells
        rep = re_sub[i,:]
        efep = exp.(fep[1:end])
        t_i = t_data[:,i]
        c_i = c_data[:,i]
        c_init = mean(c_i[1:5])
        params = vcat(efep[1:4],c_init)
        tspan = (0.0,t_i[end])
        u0 = [0.0,c_init]
        prob = ODEProblem(model_kout,u0,tspan,params)
        cb = PresetTimeCallback(dosetimes,integrator->integrator.u[1] = integrator.u[1] + rep[1])
        alg = Rodas4()
        sol = solve(prob, alg, callback=cb, reltol=1e-6, isoutofdomain=(u,p,t) -> any(x -> x < 0, u))
        c_loglike = loglike(c_i,sol(t_i,idxs=2).u,efep[end])
        loglikes[i] = c_loglike
    end
    return sum(loglikes)
end

function re_iteration_DAG_kout(re_init, fek, cells, t_data, c_data, dosetimes,fdx,pj)
    re_sub = zeros(cells,size(re_init)[2])
    for i in 1:cells
        if i == fdx
            re_sub[i,1] = pj 
        else
            rei = re_init[i,:]
            t_i = t_data[:,i]
            c_i = c_data[:,i]
            opt_out = opt_re_DAG_kout(rei,fek,t_i,c_i,dosetimes)
            re_sub[i,:] .= opt_out[2]
        end
    end
    return re_sub
end

function opt_re_DAG_kout(rep0,fep,t_i,c_i,dosetimes)
    res_re = optimize(rep->objfun_re_DAG_kout(rep,fep,t_i,c_i,dosetimes),0,10)
    return Optim.minimum(res_re), Optim.minimizer(res_re)
end

function objfun_re_DAG_kout(rep,fep,t_i,c_i,dosetimes)
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

#######################################################################
# functions for getting confidence intervals from profile likelihoods #
#######################################################################
function get_ci(cells, n_params, pl_arr, pj_arr)
    n = floor(Int,(size(pl_arr)[2]-1)/2+1)
    ci_arr = zeros(n_params+1,2)
    df = n_params + cells
    threshold = minimum(pl_arr,dims=2) .+ quantile(Chisq(df),0.95)
    threshold_values = convert(Array{Float64}, pl_arr .< threshold)
    for i in 1:n_params+1
        display(i)
        pl = pl_arr[i,:]
        pj = exp.(pj_arr[i,:]) # input pj_arr values are in log scale
        n = findall(x->x==minimum(pl_arr,dims=2)[i],pl_arr[i,:])[1]
        # left ci
        if sum(threshold_values[i,1:n-1]) == n-1
            ci_arr[i,1] = -Inf
        elseif sum(threshold_values[i,1:n-1]) < n-1
            j = maximum(findall(x->x==0,threshold_values[i,1:n-1]))
            ci_arr[i,1] = pj[j]+(threshold[i]-pl[j])*(pj[j+1]-pj[j])/(pl[j+1]-pl[j])
        end
        # right ci
        if sum(threshold_values[i,n+1:end]) == 11-n
            ci_arr[i,2] = Inf
        elseif sum(threshold_values[i,n+1:end]) < 11-n
            j = minimum(findall(x->x==0,threshold_values[i,n:end]))+n-2
            ci_arr[i,2] = pj[j]+(threshold[i]-pl[j])*(pj[j+1]-pj[j])/(pl[j+1]-pl[j])
        end
    end
    return ci_arr
end