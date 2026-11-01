"""Task 001 competitor: Julia FixedEffectModels 2.1.0 (Julia 1.13.1), called in-process via juliacall.

Follows the 14 October 2026 study's Round 2 adapter (research/2026-10-14/native-extension/task001/
benchmark.jl in the companion repository): a default `reg(df, formula, Vcov.cluster(:cl))` fit,
then the study's `correction` function (copied unchanged below) to rescale FixedEffectModels'
small-sample factor to the task's absorbed-df rule. Building the DataFrame from the NumPy inputs
is inside the timed call. TIGHT selects the study's tightened setting (tol 1e-10, maxiter 1e5).
"""
import os

import numpy as np

TIGHT = False
_STUDY = "/path/to/repro-socialscibench/research/2026-10-14/native-extension/task001"
_SHARED = "/path/to/SocialSciBench/solutions/001/.venv"
os.environ.setdefault("PYTHON_JULIAPKG_EXE", _STUDY + "/downloads/julia-1.13.1/bin/julia")
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", _SHARED + "/julia-env")
os.environ.setdefault("JULIA_DEPOT_PATH", _SHARED + "/julia-depot:" + _STUDY + "/julia-depot")
os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")
from juliacall import Main as jl  # noqa: E402

jl.seval(f'push!(LOAD_PATH, "{_STUDY}/julia-project")')
jl.seval("using FixedEffectModels, DataFrames, LinearAlgebra")
jl.seval(f'BLAS.set_num_threads({int(os.environ.get("OMP_NUM_THREADS", "1"))})')
jl.seval(r"""
function recode(ids)
    levels = Dict{Int64,Int}()
    out = Vector{Int}(undef, length(ids))
    for i in eachindex(ids)
        out[i] = get!(levels, ids[i], length(levels)+1)
    end
    out
end

function nested(a, b)
    last = zeros(Int, maximum(a))
    for i in eachindex(a)
        if last[a[i]] == 0
            last[a[i]] = b[i]
        elseif last[a[i]] != b[i]
            return false
        end
    end
    true
end

function ncomponents(a, b)
    la, lb = maximum(a), maximum(b)
    parent = collect(1:la+lb)
    sizes = ones(Int, la+lb)
    function root!(j)
        while parent[j] != j
            parent[j] = parent[parent[j]]
            j = parent[j]
        end
        j
    end
    components = la+lb
    for i in eachindex(a)
        ra, rb = root!(a[i]), root!(la+b[i])
        if ra != rb
            if sizes[ra] < sizes[rb]
                ra, rb = rb, ra
            end
            parent[rb] = ra
            sizes[ra] += sizes[rb]
            components -= 1
        end
    end
    components
end

function correction(df, k, nf)
    fes = [recode(df[!, Symbol("fe", j)]) for j in 1:nf]
    keep = trues(nrow(df))
    changed = true
    while changed
        changed = false
        for fe in fes
            counts = zeros(Int, maximum(fe))
            for i in eachindex(keep)
                keep[i] && (counts[fe[i]] += 1)
            end
            for i in eachindex(keep)
                if keep[i] && counts[fe[i]] == 1
                    keep[i] = false
                    changed = true
                end
            end
        end
    end
    if !all(keep)
        fes = [recode(fe[keep]) for fe in fes]
    end
    cl = recode(df[keep, :cl])
    n, G = count(keep), maximum(cl)
    seen = Vector{Vector{Int}}()
    df_a, native_total = 0, 0
    for fe in fes
        L = maximum(fe)
        if nested(fe, cl)
            native_total += 1
            continue
        end
        native_total += L
        if isempty(seen)
            df_a += L
        elseif length(seen) == 1
            df_a += L-ncomponents(seen[1], fe)
        else
            nested_levels = [maximum(a) for a in seen if nested(fe, a)]
            df_a += L-maximum(vcat([1], nested_levels))
        end
        push!(seen, fe)
    end
    !isempty(seen) && (df_a -= 1)
    q = (n-1)/(n-k-1-df_a)*G/(G-1)
    native_q = (n-1)/(n-k-native_total)*G/(G-1)
    (nobs=n, df_a=df_a, n_clusters=G, q=q, native_df_a=native_total-1, native_q=native_q)
end
""")
jl.seval(r"""
function task_fit(y, X, ids, cl, tight)
    n, k = size(X)
    nf = size(ids, 2)
    df = DataFrame(:y => Vector{Float64}(y), :cl => Vector{Int64}(cl))
    for j in 1:k
        df[!, Symbol("x", j)] = Vector{Float64}(X[:, j])
    end
    for j in 1:nf
        df[!, Symbol("fe", j)] = Vector{Int64}(ids[:, j])
    end
    f = term(:y) ~ sum(term(Symbol("x", j)) for j in 1:k) + sum(fe(Symbol("fe", j)) for j in 1:nf)
    m = tight ? reg(df, f, Vcov.cluster(:cl); tol=1e-10, maxiter=100000, progress_bar=false) :
                reg(df, f, Vcov.cluster(:cl); progress_bar=false)
    m.converged || error("FixedEffectModels did not converge")
    c = correction(df, k, nf)
    @assert nobs(m) == c.nobs
    @assert m.nclusters.cl == c.n_clusters
    (coef(m), vcov(m) / c.native_q * c.q, c.nobs, c.df_a, c.n_clusters)
end
""")


def run(y, X, fe_ids, cluster_ids, tol=1e-8, maxiter=100_000):
    beta, V, n, df_a, G = jl.task_fit(np.ascontiguousarray(y), np.asfortranarray(X), np.asfortranarray(fe_ids),
                                      np.ascontiguousarray(cluster_ids), TIGHT)
    return dict(beta=np.array(beta, dtype=float), vcov=np.array(V, dtype=float),
                nobs=int(n), df_a=int(df_a), n_clusters=int(G))
