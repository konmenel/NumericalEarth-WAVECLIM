using AbstractGPs
using KernelFunctions
using ParameterHandling
using Zygote
using Optimisers
using DataFrames
using CSV
using Statistics
using LinearAlgebra
using Printf
using Plots
using PythonPlot
using PolygonOps
using LaTeXStrings

# Include and use the custom Utils module containing ProgressBar
include("utils.jl")
using .Utils

const D = 2   # number of input dimensions

# ==============================================================================
# Model and Training Functions
# ==============================================================================

function train_gp_model(
    train_x::AbstractMatrix, # Shape: (N, 2) for N samples and 2 input features
    train_y::AbstractVector; # Shape: (N,)
    num_epochs::Int = 150,
    lr::Float64 = 0.1,
    verbose::Bool = false,
    bar_width::Int = 40
)
    params_init = (
        mean        = 0.0,
        lengthscale = positive(fill(0.693, D)),
        outputscale = positive(0.693),
        noise       = positive(0.693, exp, 1e-4),
    )
    
    params_flat_init, unflatten = ParameterHandling.value_flatten(params_init)
    params_flat = copy(params_flat_init)

    # Local loss function capturing unflatten in its closure scope
    function local_loss(p_flat)
        params = unflatten(p_flat)
        k = params.outputscale * with_lengthscale(Matern52Kernel(), params.lengthscale)
        f = GP(ConstMean(params.mean), k)
        fx = f(RowVecs(train_x), params.noise)
        return -logpdf(fx, train_y) / length(train_y)
    end

    opt_state = Optimisers.setup(Optimisers.Adam(lr), params_flat)

    function execute_training(bar)
        for i in 1:num_epochs
            loss, back = Zygote.pullback(local_loss, params_flat)
            grad = only(back(1.0))
            opt_state, params_flat = Optimisers.update(opt_state, params_flat, grad)

            # Yield control so background tasks can update the UI
            yield()

            if verbose && !isnothing(bar) && (i % 25 == 0)
                log_msg!(bar, "$(BOLD)$(YELLOW)INFO$(RESET) Epoch $i/$num_epochs Loss: $(round(loss; digits = 4))")
                update!(bar, 25)
            end
        end
    end

    if verbose
        pbar = ProgressBar(num_epochs; prefix="Training GP", bar_width=bar_width)
        with_progressbar(pbar) do bar
            execute_training(bar)
        end
    else
        execute_training(nothing)
    end

    optimal_params = ParameterHandling.value(unflatten(params_flat))
    trained_k = optimal_params.outputscale * with_lengthscale(Matern52Kernel(), optimal_params.lengthscale)
    trained_f = GP(ConstMean(optimal_params.mean), trained_k)
    trained_gp = posterior(trained_f(RowVecs(train_x), optimal_params.noise), train_y)
    
    return trained_gp, optimal_params
end

# ==============================================================================
# Single Point Prediction
# ==============================================================================

function predict_k660(
    u10::Float64,
    fet::Float64,
    posterior_gp::AbstractGPs.PosteriorGP,
    X_min::Vector{Float64},
    X_max::Vector{Float64},
    y_mean::Float64,
    y_std::Float64,
)
    x_norm = ([u10, fet] .- X_min) ./ (X_max .- X_min)
    x_row = RowVecs(reshape(x_norm, 1, 2)) # Size: (1, 2) row vector collection

    pred_mean = mean(posterior_gp(x_row))[1]
    log_knb = pred_mean * y_std + y_mean
    k_nb = exp(log_knb)

    return k_nb
end

# ==============================================================================
# Leave-One-Out Cross-Validation (LOOCV) with Progress Bar
# ==============================================================================

function run_loocv(x_np::Matrix{Float64}, y_np::Vector{Float64}; bar_width::Int = 40)
    n_samples = size(x_np, 1) # Samples are along rows (N, 2)

    println("Running LOOCV validation on $n_samples simulation runs...")
    loocv_preds = Float64[]
    loocv_stds = Float64[]

    # Initialize the custom progress bar matching the Python pipeline style
    pbar = ProgressBar(n_samples; prefix="Trained models", bar_width=bar_width)

    with_progressbar(pbar) do bar
        for i in 1:n_samples
            mask = trues(n_samples)
            mask[i] = false

            X_tr_raw = x_np[mask, :]       # Shape: (N-1, 2)
            y_tr_raw = y_np[mask]          # Shape: (N-1,)

            X_val_raw = x_np[i:i, :]       # Shape: (1, 2)

            X_min_f = minimum(X_tr_raw, dims=1)
            X_max_f = maximum(X_tr_raw, dims=1)
            y_mean_f = mean(y_tr_raw)
            y_std_f = std(y_tr_raw)

            X_tr = (X_tr_raw .- X_min_f) ./ (X_max_f .- X_min_f)
            y_tr = (y_tr_raw .- y_mean_f) ./ y_std_f
            X_val = (X_val_raw .- X_min_f) ./ (X_max_f .- X_min_f)

            posterior_gp, _ = train_gp_model(
                X_tr, y_tr; num_epochs=350, verbose=false, bar_width=bar_width
            )

            pred_dist = marginals(posterior_gp(RowVecs(X_val)))
            pred_mean = mean(first(pred_dist))
            pred_std = std(first(pred_dist))

            pred_mean_log = (pred_mean * y_std_f) + y_mean_f
            pred_std_log = pred_std * y_std_f

            push!(loocv_preds, exp(pred_mean_log))
            push!(loocv_stds, exp(pred_mean_log) * pred_std_log)

            update!(bar)
        end
    end

    y_np_physical = exp.(y_np)

    loocv_mae = mean(abs.(y_np_physical .- loocv_preds))
    loocv_nmae = loocv_mae / (maximum(y_np_physical) - minimum(y_np_physical))
    loocv_rmse = sqrt(mean((y_np_physical .- loocv_preds).^2))
    loocv_nrmse = loocv_rmse / (maximum(y_np_physical) - minimum(y_np_physical))
    
    ss_res = sum((y_np_physical .- loocv_preds).^2)
    ss_tot = sum((y_np_physical .- mean(y_np_physical)).^2)
    loocv_r2 = 1.0 - (ss_res / ss_tot)

    println("\n--- LOOCV Evaluation Metrics ---")
    @printf("LOOCV MAE   : %.3f cm/hr\n", loocv_mae)
    @printf("LOOCV NMAE  : %.1f%%\n", loocv_nmae * 100)
    @printf("LOOCV RMSE  : %.3f cm/hr\n", loocv_rmse)
    @printf("LOOCV NRMSE : %.1f%%\n", loocv_nrmse * 100)
    @printf("LOOCV R^2   : %.4f\n\n", loocv_r2)
end

# ==============================================================================
# Full Model Training and Visualization
# ==============================================================================

function run_and_plot_full_model(
    df::DataFrame,
    x_np::Matrix{Float64},
    y_np::Vector{Float64};
    bar_width::Int = 40,
    figtitle::Union{String, Nothing} = nothing,
    show_fig::Bool = false,
    save_fig::Bool = true,
    out_path::String = "./images",
    out_name::String = "Figure",
    out_exts::Vector{String} = ["pdf", "png"]
)
    println("Training final GP model on full dataset...")

    X_min = minimum(x_np, dims=1)
    X_max = maximum(x_np, dims=1)
    X = (x_np .- X_min) ./ (X_max .- X_min)

    y_mean = mean(y_np)
    y_std = std(y_np)
    y = (y_np .- y_mean) ./ y_std

    posterior_gp, optimal_params = train_gp_model(
        X, y; num_epochs=350, verbose=true, bar_width=bar_width
    )

    u10_test = range(5.0, 25.0, length=50)
    fet_test = range(10.0e3, 505.0e3, length=50)

    grid_u10 = [u for f in fet_test, u in u10_test]
    grid_fet = [f for f in fet_test, u in u10_test]

    test_pts_raw = hcat(vec(grid_u10), vec(grid_fet))
    test_pts = (test_pts_raw .- X_min) ./ (X_max .- X_min)

    observed_preds = marginals(posterior_gp(RowVecs(test_pts)))
    pred_means = mean.(observed_preds)
    pred_stds = std.(observed_preds)

    mean_log = (pred_means .* y_std) .+ y_mean
    std_log = pred_stds .* y_std

    mean_k660 = exp.(mean_log)

    println("\n--- Learned Physical Lengthscales ---")
    @printf("Lengthscale (U10)   : %.4f\n", optimal_params.lengthscale[1])
    @printf("Lengthscale (Fetch) : %.4f\n", optimal_params.lengthscale[2])

    boundary_pts = [
        (5.0, 10.0 * 1000),
        (8.33333333, 10.0 * 1000),
        (11.66666667, 10.0 * 1000),
        (15.0, 10.0 * 1000),
        (16.6666666666667, 10.0 * 1000),
        (18.33333333, 10.0 * 1000),
        (20.0, 10.0 * 1000),
        (21.6666666666667, 10.0 * 1000),
        (23.3333333333333, 10.0 * 1000),
        (25.0, 10.0 * 1000),
        (23.3333333333333, 92.5 * 1000),
        (21.6666666666667, 92.5 * 1000),
        (20.0, 92.5 * 1000),
        (18.33333333, 92.5 * 1000),
        (16.6666666666667, 175.0 * 1000),
        (15.0, 175.0 * 1000),
        (11.66666667, 340.0 * 1000),
        (8.33333333, 505.0 * 1000),
        (5.0, 505.0 * 1000),
        (5.0, 10.0 * 1000)
    ]

    inside_domain = [inpolygon((pt[1], pt[2]), boundary_pts) != 0 for pt in eachrow(test_pts_raw)]

    mean_grid = reshape(mean_k660, size(grid_u10))
    uncertainty_physical = reshape(2 .* mean_k660 .* std_log, size(grid_u10))

    domain_mask = reshape(inside_domain, size(grid_u10))
    mean_grid_masked = ifelse.(domain_mask, mean_grid, NaN)
    uncertainty_grid_masked = ifelse.(domain_mask, uncertainty_physical, NaN)

    # Qualified PythonPlot.pyplot.subplots
    fig, axes = PythonPlot.pyplot.subplots(1, 2, figsize=(12, 5), sharex=true, sharey=true)

    # Left Plot: Mean Prediction
    c1 = axes[0].contourf(
        grid_u10, grid_fet ./ 1000.0, mean_grid_masked, 
        levels=20, cmap="viridis"
    )
    fig.colorbar(c1, ax=axes[0], label=L"$k_{660}$ [cm/hr]")
    axes[0].scatter(
        df[!, "U10"],
        df[!, "Fetch"] ./ 1000.0,
        c="red",
        edgecolors="black",
        label="Simulations"
    )
    axes[0].set_title(L"GP Predicted Mean $k_{nb}$")
    axes[0].set_xlabel(L"$U_{10}$ [m/s]")
    axes[0].set_ylabel("Fetch [km]")
    axes[0].legend(loc="upper right")

    # Right Plot: Uncertainty
    c2 = axes[1].contourf(
        grid_u10,
        grid_fet ./ 1000.0,
        uncertainty_grid_masked, 
        levels=20,
        cmap="magma"
    )
    fig.colorbar(c2, ax=axes[1], label=L"Uncertainty $\pm 2\sigma$ [cm/hr]")
    axes[1].scatter(
        df[!, "U10"],
        df[!, "Fetch"] ./ 1000.0,
        c="white",
        edgecolors="black",
        label="Simulations"
    )
    axes[1].set_title(L"Prediction Uncertainty ($\pm 2\sigma$)")
    axes[1].set_xlabel(L"$U_{10}$ [m/s]")
    axes[1].legend(loc="upper right")

    if !isnothing(figtitle)
        fig.suptitle(figtitle)
    end

    PythonPlot.tight_layout()

    if show_fig
        PythonPlot.plotshow()
    end

    if save_fig
        mkpath(out_path)
        for ext in out_exts
            filename = joinpath(out_path, "$(out_name).$(ext)")
            fig.savefig(filename, dpi=300)
            println("Saved figure - '$filename'")
        end
    end
    PythonPlot.pyplot.close(fig)
end

# ==============================================================================
# Main Orchestrator
# ==============================================================================

function main()
    PythonPlot.pyplot.style.use("pps_white_grid.mplstyle")

    data_path = "/run/media/konmenel/OneTouch/data/DualSPH/Inishmaan"
    old_pwd = pwd()

    if isdir(data_path)
        cd(data_path)
    end

    BAR_WIDTH = 40

    df = CSV.read("k660_dataset.csv", DataFrame)

    X_np = Matrix{Float64}(df[:, ["U10", "Fetch"]])
    y_np = log.(Float64.(df[:, "k_nb_cm_hr"]))

    # ------------------------------------------------------------------
    # ALL POINTS
    # ------------------------------------------------------------------

    run_loocv(X_np, y_np, bar_width=BAR_WIDTH)

    run_and_plot_full_model(
        df, X_np, y_np;
        bar_width=BAR_WIDTH,
        figtitle="All Input",
        save_fig=true,
        out_path="./images/gp/julia",
        out_name="GP_Unfiltered_Results"
    )

    # ------------------------------------------------------------------
    # FILTERED
    # ------------------------------------------------------------------

    bad_points = [(5.0, 257.5 * 1000), (8.33333333, 10.0 * 1000)]
    good_mask = trues(nrow(df))

    for (u10_bad, fetch_bad) in bad_points
        match_u10 = isapprox.(df[!, "U10"], u10_bad)
        match_fetch = isapprox.(df[!, "Fetch"], fetch_bad)
        is_bad = match_u10 .& match_fetch
        good_mask .&= .!is_bad
    end

    df_filtered = df[good_mask, :]
    X_np_filtered = Matrix{Float64}(df_filtered[:, ["U10", "Fetch"]])
    y_np_filtered = log.(Float64.(df_filtered[:, "k_nb_cm_hr"]))

    run_loocv(X_np_filtered, y_np_filtered, bar_width=BAR_WIDTH)

    run_and_plot_full_model(
        df_filtered, X_np_filtered, y_np_filtered;
        bar_width=BAR_WIDTH,
        figtitle="Excluding Bad Input",
        save_fig=true,
        out_path="./images/gp/julia",
        out_name="GP_Filtered_Results"
    )

    cd(old_pwd)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end