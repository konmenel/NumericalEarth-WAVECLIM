module Progressbar
using Base.Threads
using Printf

export ProgressBar, update!, draw_ui!, with_progressbar, log_msg!
export HIDE_CURSOR, SHOW_CURSOR, CLEAR_LINE
export RESET, RESET_BOLD, BOLD, GRAY, RED, GREEN, YELLOW, BLUE

Base.exit_on_sigint(false)

const HIDE_CURSOR = "\033[?25l"
const SHOW_CURSOR = "\033[?25h"
const CLEAR_LINE = "\033[K"

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
const PACMAN_CHARS = ['C', 'C', 'C','C', 'C', 'c', 'c', 'c', 'c', 'c']

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
    run_thread::Union{Task, Nothing}
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
    p_postfi = (!isempty(postfix) && !isspace(last(postfix))) ? string(postfix, " ") : string(postfix)

    supports_tc = lowercase(get(ENV, "COLORTERM", "")) in ("truecolor", "24bit")

    return ProgressBar(
        max(1, max_steps),
        bar_width,
        0,
        0.0,
        color_gradient,
        show_spinner,
        p_prefix,
        p_postfi,
        time(),
        1,
        ReentrantLock(),
        Ref(false),
        nothing,
        supports_tc
    )
end


function set_prefix!(bar::ProgressBar, val::AbstractString)
    lock(bar.lock) do
        bar.prefix = (!isempty(val) && !isspace(last(val))) ? string(val, " ") : string(val)
    end
end

function set_postfix!(bar::ProgressBar, val::AbstractString)
    lock(bar.lock) do
        bar.postfix = (!isempty(val) && !isspace(last(val))) ? string(val, " ") : string(val)
    end
end


function get_gradient_color(bar::ProgressBar)
    r, g, b = 0, 0, 0
    if bar.progress < 0.5
        fraction = bar.progress / 0.5
        r = 255
        g = round(Int, fraction * 255)
        b = 0
    else
        fraction = (bar.progress - 0.5) / 0.5
        r = round(Int, (1.0 - fraction) * 255)
        g = 255
        b = 0
    end

    if bar.supports_truecolor
        return "\033[38;2;$r;$g;$(b)m"
    else
        r_idx = floor(Int, r / 255 * 5)
        g_idx = floor(Int, g / 255 * 5)
        b_idx = floor(Int, b / 255 * 5)
        color_id = 16 + (36 * r_idx) + (6 * g_idx) + b_idx
        return "\033[38;5;$(color_id)m"
    end
end


function update!(bar, increment::Int=1)
    lock(bar.lock) do
        bar.curr_step += increment
        bar.progress = bar.curr_step / bar.max_steps
        draw_ui(bar)
    end
end


function draw_ui(bar; force_complete::Bool = false)
    done_width = max(1, round(Int, bar.progress * bar.width))
    remain_width = max(0, bar.width - done_width)

    color = bar.color_gradient ? get_gradient_color(bar) : ""
    reset = bar.color_gradient ? RESET : ""

    pacman_char = PACMAN_CHARS[mod1(bar.spinner_idx, length(PACMAN_CHARS))]
    filled_bar = done_width > 0 ? string(repeat(DONE_CHAR, done_width - 1), pacman_char) : ""
    empty_bar = repeat(REMAINING_CHAR, remain_width)

    elapsed = time() - bar.start_time

    if bar.curr_step > 0
        steps_per_sec = bar.curr_step / elapsed
        remaining_steps = bar.max_steps - bar.curr_step
        eta_seconds = remaining_steps / steps_per_sec
        mins = div(floor(Int, eta_seconds), 60)
        secs = rem(floor(Int, eta_seconds), 60)
        eta_str = Printf.@sprintf("%02d:%02d", mins, secs)
    else
        eta_str = "--:--"
    end

    spinner_str = ""
    if bar.show_spinner
        if bar.curr_step >= bar.max_steps || force_complete
            spinner_str = "$(color)✓$(reset) "
        else
            char = SPINNER_CHARS[mod1(bar.spinner_idx, length(SPINNER_CHARS))]
            spinner_str = "$(color)$(char)$(reset) "
        end
    end

    percent_str = Printf.@sprintf("%4.0f%%", bar.progress * 100)

    out_str = string(
        "\r", CLEAR_LINE, bar.prefix,
        color, filled_bar, reset, empty_bar,
        " ", spinner_str,
        " ", color, bar.curr_step, "/", bar.max_steps, " ", percent_str, reset,
        " ", bar.postfix, "(ETA: ", eta_str, ")"
    )

    print(stdout, out_str)
    flush(stdout)
end

function start!(bar::ProgressBar)
    bar.stop_event[] = false

    lock(bar.lock) do
        print(HIDE_CURSOR)
        flush(stdout)
    end

    bar.run_thread = Threads.@spawn while !bar.stop_event[]
        try
            lock(bar.lock) do
                draw_ui(bar)
                if bar.show_spinner
                    bar.spinner_idx += 1
                end
            end
            sleep(0.05)
        catch e
            if e isa InterruptException
                break
            else
                rethrow(e)
            end
        end
    end
end

function finish!(bar::ProgressBar; exception=nothing)
    bar.stop_event[] = true
    if !isnothing(bar.run_thread)
        try
            wait(bar.run_thread)
        catch
        end
    end

    lock(bar.lock) do
        if bar.curr_step < bar.max_steps && isnothing(exception)
            bar.curr_step = bar.max_steps
            bar.progress = 1.0
        end
        draw_ui(bar; force_complete = isnothing(exception))
        print(SHOW_CURSOR)
        println(stdout)
        flush(stdout)
    end
end

function with_progressbar(f::Function, bar::ProgressBar)
    start!(bar)
    try
        f(bar)
    catch e
        finish!(bar; exception=e)
        rethrow(e)
    else
        finish!(bar)
    end
end

function log_msg!(bar::ProgressBar, message::AbstractString)
    lock(bar.lock) do
        print(stdout, "\r$CLEAR_LINE")
        println(stdout, message)
        draw_ui(bar)
    end
end

end