module Utils

using Printf
using DataFrames
using CSV

export SEED, HsTpLookup, get_hs_tp, ProgressBar, update!, log_msg!, set_prefix!, set_postfix!, with_progressbar
export RESET, RESET_BOLD, BOLD, GRAY, RED, GREEN, YELLOW, BLUE

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


# ==============================================================================
# ProgressBar
# ==============================================================================

const RESET = "\033[0m"
const RESET_BOLD = "\033[22m"
const BOLD = "\033[1m"
const GRAY = "\033[90m"
const RED = "\033[31m"
const GREEN = "\033[32m"
const YELLOW = "\033[33m"
const BLUE = "\033[34m"

const DONE_CHAR = '-'
const REMAINING_CHAR = '·'
const SPINNER_CHARS = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
const PACMAN_CHARS = "CCCCCccccc"

mutable struct ProgressBar
    max_steps::Int
    width::Int
    curr_step::Int
    progress::Float64
    color_gradient::Bool
    show_spinner::Bool
    prefix::String
    postfix::String
    start_time::Float64
    spinner_idx::Int
    lock::ReentrantLock
    stop_event::Ref{Bool}
    animation_task::Union{Task, Nothing}
    supports_truecolor::Bool
end

function ProgressBar(
    max_steps::Int;
    bar_width::Int = 40,
    prefix::AbstractString = "",
    postfix::AbstractString = "",
    color_gradient::Bool = true,
    show_spinner::Bool = true
)
    p_prefix = (!isempty(prefix) && !isspace(last(prefix))) ? string(prefix, " ") : string(prefix)
    p_postfix = (!isempty(postfix) && !isspace(last(postfix))) ? string(postfix, " ") : string(postfix)

    supports_tc = lowercase(get(ENV, "COLORTERM", "")) in ("truecolor", "24bit")

    return ProgressBar(
        max(1, max_steps),
        bar_width,
        0,
        0.0,
        color_gradient,
        show_spinner,
        p_prefix,
        p_postfix,
        time(),
        1,
        ReentrantLock(),
        Ref(false),
        nothing,
        supports_tc
    )
end

function set_prefix!(pbar::ProgressBar, val::AbstractString)
    pbar.prefix = (!isempty(val) && !isspace(last(val))) ? string(val, " ") : string(val)
end

function set_postfix!(pbar::ProgressBar, val::AbstractString)
    pbar.postfix = (!isempty(val) && !isspace(last(val))) ? string(val, " ") : string(val)
end

function get_gradient_color(pbar::ProgressBar)
    r, g, b = 0, 0, 0
    if pbar.progress < 0.5
        fraction = pbar.progress / 0.5
        r = 255
        g = round(Int, fraction * 255)
        b = 0
    else
        fraction = (pbar.progress - 0.5) / 0.5
        r = round(Int, (1.0 - fraction) * 255)
        g = 255
        b = 0
    end

    if pbar.supports_truecolor
        return "\033[38;2;$r;$g;$(b)m"
    else
        r_idx = floor(Int, r / 255 * 5)
        g_idx = floor(Int, g / 255 * 5)
        b_idx = floor(Int, b / 255 * 5)
        color_id = 16 + (36 * r_idx) + (6 * g_idx) + b_idx
        return "\033[38;5;$(color_id)m"
    end
end

function draw_ui!(pbar::ProgressBar; force_complete::Bool = false)
    done_width = round(Int, pbar.progress * pbar.width)
    remain_width = max(0, pbar.width - done_width)

    color = pbar.color_gradient ? get_gradient_color(pbar) : ""
    reset = pbar.color_gradient ? RESET : ""

    pacman_char = PACMAN_CHARS[mod1(pbar.spinner_idx, length(PACMAN_CHARS))]
    filled_bar = done_width > 0 ? string(repeat(DONE_CHAR, done_width - 1), pacman_char) : ""
    empty_bar = repeat(REMAINING_CHAR, remain_width)

    elapsed = time() - pbar.start_time

    if pbar.curr_step > 0
        steps_per_sec = pbar.curr_step / elapsed
        remaining_steps = pbar.max_steps - pbar.curr_step
        eta_seconds = remaining_steps / steps_per_sec
        mins = div(floor(Int, eta_seconds), 60)
        secs = rem(floor(Int, eta_seconds), 60)
        eta_str = Printf.@sprintf("%02d:%02d", mins, secs)
    else
        eta_str = "--:--"
    end

    spinner_str = ""
    if pbar.show_spinner
        if pbar.curr_step >= pbar.max_steps || force_complete
            spinner_str = "$(color)✓$(reset) "
        else
            char = SPINNER_CHARS[mod1(pbar.spinner_idx, length(SPINNER_CHARS))]
            spinner_str = "$(color)$(char)$(reset) "
        end
    end

    pct_str = Printf.@sprintf("%4.0f%%", pbar.progress * 100)

    out_str = string(
        "\r\033[K", pbar.prefix,
        color, filled_bar, reset, empty_bar,
        " ", spinner_str,
        " ", color, pbar.curr_step, "/", pbar.max_steps, " ", pct_str, reset,
        " ", pbar.postfix, "(ETA: ", eta_str, ")"
    )

    print(stdout, out_str)
    flush(stdout)
end

function start!(pbar::ProgressBar)
    pbar.start_time = time()
    pbar.stop_event[] = false
    pbar.animation_task = @async begin
        while !pbar.stop_event[]
            lock(pbar.lock) do
                draw_ui!(pbar)
                if pbar.show_spinner
                    pbar.spinner_idx += 1
                end
            end
            sleep(0.05)
        end
    end
    return pbar
end

function finish!(pbar::ProgressBar; exception = nothing)
    pbar.stop_event[] = true
    if !isnothing(pbar.animation_task)
        wait(pbar.animation_task)
    end

    lock(pbar.lock) do
        if pbar.curr_step < pbar.max_steps && isnothing(exception)
            pbar.curr_step = pbar.max_steps
            pbar.progress = 1.0
        end
        draw_ui!(pbar; force_complete = isnothing(exception))
        println(stdout)
        flush(stdout)
    end
end

"""
    with_progressbar(f::Function, pbar::ProgressBar)

Executes `f(pbar)` inside a progress bar context (replaces Python's `with ProgressBar(...) as bar:`).
"""
function with_progressbar(f::Function, pbar::ProgressBar)
    start!(pbar)
    try
        f(pbar)
    catch e
        finish!(pbar; exception = e)
        rethrow(e)
    else
        finish!(pbar)
    end
end

function update!(pbar::ProgressBar, increment::Int = 1)
    lock(pbar.lock) do
        pbar.curr_step += increment
        if pbar.curr_step > pbar.max_steps
            pbar.curr_step = pbar.max_steps
        elseif pbar.curr_step < 0
            pbar.curr_step = 0
        end
        pbar.progress = pbar.curr_step / pbar.max_steps
    end
end

function log_msg!(pbar::ProgressBar, message::AbstractString)
    lock(pbar.lock) do
        print(stdout, "\r\033[K")
        println(stdout, message)
        draw_ui!(pbar)
    end
end

end