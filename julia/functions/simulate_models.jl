# load packages
using DifferentialEquations
using Distributions
using Random

# model without kout
function model(du,u,p,t)
    kin,kmet,Kd,C1_0 = p
    du[1] = -kin*u[1] # d[DAGext]/dt
    du[2] = (-kin*u[1] + kmet*Kd*((C1_0/u[2])-1))/((Kd*C1_0/(u[2]^2))+1) # d[C1]/dt
end

# model with kout
function model_kout(du,u,p,t)
    kin,kout,kmet,Kd,C1_0 = p
    du[1] = -kin*u[1] + kout*Kd*((C1_0/u[2])-1)# d[DAGext]/dt
    du[2] = (-kin*u[1] + (kmet+kout)*Kd*((C1_0/u[2])-1))/((Kd*C1_0/(u[2]^2))+1) # d[C1]/dt
end

# simulate model for multiple cells using the same parameter set but different intial conditions (C1 and DAGext) and noise levels (sd)
function simulate_data(t_data, dag_uncaged, c1_init, params, dosetimes, sd)
    c1_sim = zeros(size(t_data))
    for i in 1:size(t_data)[2]
        p = vcat(params,c1_init[i])
        tspan = (0.0,t_data[end,i])
        u0 = [0.0,c1_init[i]]
        if length(params) == 3 # no kout model
            prob = ODEProblem(model,u0,tspan,p)
        elseif length(params) == 4 # with kout model
            prob = ODEProblem(model_kout,u0,tspan,p)
        end
        cb = PresetTimeCallback(dosetimes[1],integrator->integrator.u[1] = integrator.u[1] + dag_uncaged[i])
        alg = Rodas4()
        sol = solve(prob, alg, callback=cb, reltol=1e-6)
        c1_sim[:,i] = sol(t_data[:,i],idxs=2).u + rand(Normal(0, sd[i]),size(t_data[:,i]))
    end
    return c1_sim
end

# simulate model for one cell using the same parameter set but different intial conditions (C1 and DAGext) and noise levels (sd)
function simulate_data_single(t_data, dag_uncaged, c1_init, params, dosetimes, sd)
    c1_sim = zeros(length(t_data))
    p = vcat(params,c1_init)
    tspan = (0.0,t_data[end])
    u0 = [0.0,c1_init]
    if length(params) == 3 # no kout model
        prob = ODEProblem(model,u0,tspan,p)
    elseif length(params) == 4 # with kout model
        prob = ODEProblem(model_kout,u0,tspan,p)
    end
    cb = PresetTimeCallback(dosetimes[1],integrator->integrator.u[1] = integrator.u[1] + dag_uncaged)
    alg = Rodas4()
    sol = solve(prob, alg, callback=cb, reltol=1e-6)
    c1_sim = sol(t_data,idxs=2).u + rand(Normal(0, sd),length(t_data))
    return c1_sim
end