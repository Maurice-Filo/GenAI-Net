#!/usr/bin/env julia

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const RNE_SRC = joinpath(ROOT, "comparisons", "rpa_search", "julia", "ReactionNetworkEvolution", "src")

include(joinpath(RNE_SRC, "process_output.jl"))

function parse_fitness(astr::String)
    for line in split(astr, "\n")
        if startswith(line, "#fitness")
            parts = split(line)
            length(parts) >= 2 || return missing
            try
                return parse(Float64, parts[2])
            catch
                return missing
            end
        end
    end
    return missing
end

function evaluate_file(path::String)
    seed = splitext(basename(path))[1]
    astr = read(path, String)
    fitness = parse_fitness(astr)
    loss = ismissing(fitness) || fitness == 0 ? missing : 1 / fitness

    parse_ok = true
    good = false
    broken = false
    fixed = false
    final_success = false
    error = ""

    try
        good = is_oscillator(astr)
        if good
            final_success = true
        else
            broken = is_broken_oscillator(astr)
            if broken
                fixed_astr = fix_broken_oscillator(astr)
                fixed = fixed_astr != "FAIL"
                final_success = fixed
            end
        end
    catch e
        parse_ok = false
        error = sprint(showerror, e)
    end

    return (
        seed = seed,
        file = path,
        fitness = fitness,
        loss = loss,
        parse_ok = parse_ok,
        rne_is_oscillator = good,
        rne_is_broken_oscillator = broken,
        rne_fixed_by_reaction_removal = fixed,
        rne_posthoc_success = final_success,
        error = error,
    )
end

function csv_escape(x)
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s) || occursin("\r", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_rows_csv(path::String, rows)
    columns = [
        :seed,
        :file,
        :fitness,
        :loss,
        :parse_ok,
        :rne_is_oscillator,
        :rne_is_broken_oscillator,
        :rne_fixed_by_reaction_removal,
        :rne_posthoc_success,
        :error,
    ]
    open(path, "w") do io
        println(io, join(string.(columns), ","))
        for row in rows
            vals = [getfield(row, col) for col in columns]
            println(io, join(csv_escape.(vals), ","))
        end
    end
end

function main()
    if length(ARGS) < 2
        println("usage: julia evaluate_rne_posthoc.jl INPUT_ANT_DIR OUTPUT_CSV")
        exit(2)
    end

    input_dir = ARGS[1]
    output_csv = ARGS[2]
    files = sort([
        joinpath(input_dir, file)
        for file in readdir(input_dir)
        if endswith(file, ".ant")
    ])

    rows = [evaluate_file(path) for path in files]
    mkpath(dirname(output_csv))
    write_rows_csv(output_csv, rows)

    n = length(rows)
    successes = sum(row -> row.rne_posthoc_success, rows)
    println("Evaluated $n networks")
    println("RNE posthoc successes: $successes/$n")
    println("Output: $output_csv")
end

main()
