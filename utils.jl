module Utils

using Printf
using DataFrames
using CSV

export SEED, HsTpLookup, get_hs_tp

const SEED = 42

# ==============================================================================
# HsTpLookup
# ==============================================================================

mutable struct HsTpLookup
    csv_path::String
    cache::Union{DataFrame, Nothing}
end

HsTpLookup(csv_path::String = "signals2/info.csv") = HsTpLookup(csv_path, nothing)

function (lookup::HsTpLookup)(u10::Real, fet::Real; seed::Int = SEED)
    if isnothing(lookup.cache)
        if !isfile(lookup.csv_path)
            return (nothing, nothing)
        end
        lookup.cache = CSV.read(lookup.csv_path, DataFrame; delim=';')
    end

    df = lookup.cache
    mask = isapprox.(df[!, "U10 [m/s]"], u10) .&
           isapprox.(df[!, "Fet [m]"], fet) .&
           isapprox.(df[!, "seed"], seed)

    sub = df[mask, ["Hs [m]", "Tp [s]"]]
    if nrow(sub) == 0
        return (nothing, nothing)
    end

    return (sub[1, "Hs [m]"], sub[1, "Tp [s]"])
end

const get_hs_tp = HsTpLookup()

end