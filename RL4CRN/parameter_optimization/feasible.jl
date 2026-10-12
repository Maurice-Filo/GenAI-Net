# Optional feasibility-only backtracking. The original optimizer stays unchanged.
function optimize_feasible(model, key, start, checkpoint, steps, config; evaluation_limit=nothing)
    started = time_ns()
    result = optimize_one(model, key, start, checkpoint, 0, config)
    result["trials"] = Any[]
    result["requested_steps"] = steps
    result["step_policy"] = "feasible"
    result["status"] == "ok" || return result
    state = result["state"]
    current = state["current"]
    lo, hi = Float64.(config["bounds"])
    rates = clamp.(exp.(state["theta"]), lo, hi)
    evaluations = result["evaluations"]
    completed = result["completed_evaluations"]
    solve_counter = Ref(result["scenario_solves"])
    limit = config["max_evaluations"] === nothing ? typemax(Int) : Int(config["max_evaluations"])
    # A dispatch budget is not optimizer state: continuation may have less credit.
    if evaluation_limit !== nothing
        Int(evaluation_limit) >= 1 || error("Invalid runtime evaluation limit")
        limit = min(limit, Int(evaluation_limit))
    end
    status, failure = "ok", nothing
    for row in result["history"]
        row["evaluations"] = evaluations
        row["scenario_solves"] = solve_counter[]
        row["elapsed_s"] = (time_ns()-started)/1e9
    end
    for _ in 1:steps
        # Tentative optimizer memory is committed only after a feasible step.
        tentative = deepcopy(state)
        tentative["step"] += 1
        gradient = rates .* Float64.(current["gradient"])
        lr = Float64(config["learning_rate"])
        delta = if config["method"] == "adagrad"
            coefficient = adagrad_update_get_coefficient(rates, gradient, tentative["accumulator"], lr)
            coefficient .* gradient
        else
            ADAM_update_get_coefficient(rates, gradient, tentative["moment"], tentative["velocity"],
                                       lr, tentative["step"], .9, .999, 1e-8)
        end
        all(isfinite, delta) || error("Nonfinite optimizer step")
        accepted = false
        for attempt in 0:Int(config["max_backtracks"])
            if evaluations >= limit
                status, failure = "evaluation_budget_exhausted", "Per-start evaluation budget reached"
                break
            end
            scale = 0.5^attempt
            theta = clamp.(state["theta"] - scale .* delta, log(lo), log(hi))
            if theta == state["theta"]
                status, failure = "stalled", "Projected step is numerically unchanged"
                break
            end
            candidate = clamp.(exp.(theta), lo, hi)
            evaluations += 1
            trial = Dict{String,Any}("step"=>tentative["step"], "backtracks"=>attempt,
                "scale"=>scale, "rates"=>candidate, "evaluation"=>evaluations,
                "loss"=>nothing, "error"=>nothing)
            proposal = try
                value = evaluate(model, candidate; solve_counter=solve_counter)
                completed += 1
                value
            catch error_value
                error_value isa EvaluationFailure || rethrow()
                trial["error"] = sprint(showerror, error_value)
                nothing
            end
            trial["scenario_solves"] = solve_counter[]
            trial["elapsed_s"] = (time_ns()-started)/1e9
            if proposal === nothing
                trial["status"] = "rejected"
                push!(result["trials"], trial)
                continue
            end
            trial["status"], trial["loss"] = "accepted", proposal["loss"]
            push!(result["trials"], trial)
            tentative["theta"], tentative["current"] = theta, proposal
            if proposal["loss"] < tentative["best_loss"]
                tentative["best_loss"], tentative["best_rates"] = proposal["loss"], copy(candidate)
            end
            state, current, rates = tentative, proposal, candidate
            push!(result["history"], Dict("step"=>state["step"], "loss"=>current["loss"],
                "best_loss"=>state["best_loss"], "evaluations"=>evaluations,
                "scenario_solves"=>solve_counter[], "elapsed_s"=>(time_ns()-started)/1e9))
            accepted = true
            break
        end
        if !accepted
            if status == "ok"
                status, failure = "backtracking_exhausted", "No feasible trial within the backtracking limit"
            end
            break
        end
    end
    merge!(result, Dict("status"=>status, "error"=>failure, "state"=>state,
        "best_loss"=>state["best_loss"], "best_rates"=>state["best_rates"],
        "final_rates"=>rates, "final_loss"=>current["loss"], "evaluations"=>evaluations,
        "completed_evaluations"=>completed, "scenario_solves"=>solve_counter[],
        "elapsed_s"=>(time_ns()-started)/1e9))
    return result
end
