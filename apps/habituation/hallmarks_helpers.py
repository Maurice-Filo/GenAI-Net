import numpy as np


def rectangular_pulse_program(A, T, Ton, n_pulses, points_per_time=100):
    """Build a repeated rectangular ON/OFF input protocol.

    Each pulse consists of an ON segment with input amplitude `A` for duration
    `Ton`, followed by an OFF segment with input amplitude zero for duration
    `T - Ton`. The returned objects use the nested piecewise format expected by
    `IOCRN.transient_response_piecewise`.

    Args:
        A: Input amplitude during each ON segment.
        T: Duration of one complete pulse period.
        Ton: Duration of the ON part of each pulse.
        n_pulses: Number of ON/OFF pulse periods to generate.
        points_per_time: Number of time-grid samples per unit time.

    Returns:
        Tuple containing:
            - inputs: A single-protocol `u_nested_list` with alternating ON/OFF
              input vectors.
            - segments: List of local time grids, one for each ON or OFF segment.
    """
    on_grid = np.linspace(0.0, Ton, max(3, int(np.ceil(Ton * points_per_time)) + 1))
    off_grid = np.linspace(0.0, T - Ton, max(3, int(np.ceil((T - Ton) * points_per_time)) + 1))
    inputs, segments = [], []
    for _ in range(n_pulses):
        inputs.extend([np.asarray([A]), np.asarray([0.0])])
        segments.extend([on_grid, off_grid])
    return [inputs], segments


def extract_period_peaks(time, response, T, n_pulses):
    """Extract the maximum response in each pulse period.

    Args:
        time: One-dimensional array of simulation time points.
        response: One-dimensional response trajectory sampled at `time`.
        T: Duration of one complete pulse period.
        n_pulses: Number of pulse periods to evaluate.

    Returns:
        Array containing one peak value per pulse period.
    """
    time = np.asarray(time, dtype=float)
    response = np.asarray(response, dtype=float)
    return np.asarray([
        response[(time >= pulse_idx * T) & (time <= (pulse_idx + 1) * T)].max()
        for pulse_idx in range(int(n_pulses))
    ])


def extract_period_troughs(time, response, T, n_pulses):
    """Extract the minimum response in each pulse period.

    Args:
        time: One-dimensional array of simulation time points.
        response: One-dimensional response trajectory sampled at `time`.
        T: Duration of one complete pulse period.
        n_pulses: Number of pulse periods to evaluate.

    Returns:
        Array containing one trough value per pulse period.
    """
    time = np.asarray(time, dtype=float)
    response = np.asarray(response, dtype=float)
    return np.asarray([
        response[(time >= pulse_idx * T) & (time <= (pulse_idx + 1) * T)].min()
        for pulse_idx in range(int(n_pulses))
    ])


def compute_peak_window_decrements(peaks, n_dec=1, eps=1e-12):
    """Compute peak-window decrements from local maxima and minima.

    For each sliding window containing `n_dec + 1` consecutive peaks, this
    computes the relative range

        (max(window) - min(window)) / max(window).

    The accompanying boolean array records whether the final peak in the window
    is no larger than the first peak, so small upward drifts are not treated as
    habituating decrements.

    Args:
        peaks: One-dimensional sequence of pulse-period peak responses.
        n_dec: Number of peak-to-peak intervals in each window. The default
            value `1` compares adjacent peaks.
        eps: Small positive value used to avoid division by zero.

    Returns:
        Tuple containing:
            - window_decrements: Relative max-min range in each window.
            - window_maxima: Maximum peak in each window.
            - window_minima: Minimum peak in each window.
            - window_is_nonincreasing: Boolean trend check for each window.
    """
    peaks = np.asarray(peaks, dtype=float)
    n_dec = int(n_dec)
    if n_dec < 1:
        raise ValueError("n_dec must be at least 1.")

    n_windows = max(peaks.size - n_dec, 0)
    window_decrements = np.full(n_windows, np.inf, dtype=float)
    window_maxima = np.full(n_windows, np.nan, dtype=float)
    window_minima = np.full(n_windows, np.nan, dtype=float)
    window_is_nonincreasing = np.zeros(n_windows, dtype=bool)

    for start_idx in range(n_windows):
        window = peaks[start_idx:start_idx + n_dec + 1]
        if not np.all(np.isfinite(window)):
            continue
        window_max = float(np.max(window))
        window_min = float(np.min(window))
        window_maxima[start_idx] = window_max
        window_minima[start_idx] = window_min
        window_is_nonincreasing[start_idx] = bool(window[-1] <= window[0])
        if window_max > float(eps):
            window_decrements[start_idx] = (window_max - window_min) / window_max

    return window_decrements, window_maxima, window_minima, window_is_nonincreasing


def compute_peak_tube_decrements(peaks, min_remaining_intervals=1, eps=1e-12):
    """Compute relative tube widths for suffixes of a peak sequence.

    For each candidate start index, this computes the relative width of the
    smallest horizontal tube containing all peaks from that index through the
    final pulse:

        (max(tail) - min(tail)) / max(max(tail), eps).

    Args:
        peaks: One-dimensional sequence of pulse-period peak responses.
        min_remaining_intervals: Minimum number of peak-to-peak intervals that
            must remain in the suffix. The default value `1` requires at least
            two peaks in the tube.
        eps: Small positive value used to avoid division by zero.

    Returns:
        Tuple containing:
            - tube_decrements: Relative tube width for each qualifying suffix
              start. Non-qualifying starts are set to `np.inf`.
            - tube_maxima: Maximum peak in each suffix.
            - tube_minima: Minimum peak in each suffix.
    """
    peaks = np.asarray(peaks, dtype=float)
    min_remaining_intervals = int(min_remaining_intervals)
    if min_remaining_intervals < 1:
        raise ValueError("min_remaining_intervals must be at least 1.")

    tube_decrements = np.full(peaks.size, np.inf, dtype=float)
    tube_maxima = np.full(peaks.size, np.nan, dtype=float)
    tube_minima = np.full(peaks.size, np.nan, dtype=float)

    last_start = peaks.size - min_remaining_intervals - 1
    for start_idx in range(max(last_start + 1, 0)):
        tail = peaks[start_idx:]
        if not np.all(np.isfinite(tail)):
            continue
        tube_max = float(np.max(tail))
        tube_min = float(np.min(tail))
        tube_maxima[start_idx] = tube_max
        tube_minima[start_idx] = tube_min
        if tube_max > float(eps):
            tube_decrements[start_idx] = (tube_max - tube_min) / tube_max

    return tube_decrements, tube_maxima, tube_minima


def compute_habituation_time(
    time,
    response,
    T,
    n_pulses,
    tolerance=0.01,
    blowup_threshold=1e2,
    n_dec=1,
    habituation_time_method="window",
):
    """Compute habituation time from response peaks across repeated pulses.

    The peak response is measured over each pulse period. Habituation time is
    defined using one of two methods. The default `"window"` method uses the
    first pulse index for which the relative max-min range over a local window
    of `n_dec + 1` consecutive peaks is smaller than `tolerance`, and the
    window ends no higher than it starts. The `"tube"` method uses the first
    pulse index after which all remaining peaks fit inside a relative tube of
    width `tolerance`.

    Args:
        time: One-dimensional array of simulation time points.
        response: One-dimensional output trajectory sampled at `time`.
        T: Duration of one complete pulse period.
        n_pulses: Number of pulse periods to evaluate.
        tolerance: Maximum relative decrement used to identify habituation.
        blowup_threshold: If any absolute response value reaches this
            threshold, treat the trajectory as unstable and return `np.inf`.
        n_dec: Number of peak-to-peak intervals in each window. The default
            value `1` recovers the ordinary adjacent-peak test. For the
            `"tube"` method, this is the minimum number of peak-to-peak
            intervals that must remain after the detected pulse.
        habituation_time_method: Either `"window"` for the historical local
            window criterion or `"tube"` for the suffix-tube criterion.

    Returns:
        Tuple containing:
            - habituation_time: First qualifying pulse index, or `np.inf` if the
              response does not habituate.
            - peaks: Array containing the maximum response in each pulse period.
    """
    response = np.asarray(response, dtype=float)
    n_dec = int(n_dec)
    if n_dec < 1:
        raise ValueError("n_dec must be at least 1.")
    habituation_time_method = str(habituation_time_method).lower()
    if habituation_time_method not in {"window", "tube"}:
        raise ValueError("habituation_time_method must be either 'window' or 'tube'.")

    if (
        response.size == 0
        or not np.all(np.isfinite(response))
        or np.nanmax(np.abs(response)) >= float(blowup_threshold)
    ):
        return np.inf, np.asarray([], dtype=float)

    peaks = extract_period_peaks(time, response, T, n_pulses)
    if not np.all(np.isfinite(peaks)) or np.any(np.abs(peaks) >= float(blowup_threshold)):
        return np.inf, peaks

    if habituation_time_method == "tube":
        tube_decrements, _, _ = compute_peak_tube_decrements(
            peaks,
            min_remaining_intervals=n_dec,
        )
        if tube_decrements.size == 0:
            return np.inf, peaks
        qualifying_indices = np.flatnonzero(tube_decrements < tolerance)
        habituation_time = int(qualifying_indices[0]) if len(qualifying_indices) else np.inf
        return habituation_time, peaks

    window_decrements, _, _, window_is_nonincreasing = compute_peak_window_decrements(
        peaks,
        n_dec=n_dec,
    )
    if window_decrements.size == 0:
        return np.inf, peaks

    qualifying_indices = np.flatnonzero(
        window_is_nonincreasing & (window_decrements < tolerance)
    )
    habituation_time = int(qualifying_indices[0] + n_dec) if len(qualifying_indices) else np.inf
    return habituation_time, peaks

def compute_recovery_time(iocrn, A, T, Ton, habituation_time, x0,
                          recovery_tolerance=0.05, max_gap=4000.0,
                          search_depth=16, return_info=False):
    """Compute recovery time after habituation.

    The circuit is first trained for `habituation_time` pulse periods. Each
    training period ends at the final 0.01-spaced sample before the next period, 
    so recovery starts from the sampled state at approximately `habituation_time * T - 0.01`. 
    The system is then held at zero input for a candidate gap and tested with one additional pulse. 
    Recovery time is the smallest gap, found by binary search, whose test response peak is
    within `recovery_tolerance` of the first pulse response.

    Args:
        iocrn: IOCRN-like object implementing `transient_response_piecewise`.
        A: Input amplitude during each ON segment.
        T: Duration of one complete pulse period.
        Ton: Duration of the ON part of each pulse.
        habituation_time: Number of pulse periods used for the training phase.
        x0: Initial condition for the training phase.
        recovery_tolerance: Relative tolerance for recovery to the first peak.
        max_gap: Largest OFF gap tested before giving up.
        search_depth: Number of binary-search refinements.
        return_info: If True, also return a dictionary with the full recovery
            protocol time, input signal, response, and trajectory, including
            the training pulses before the recovery gap.

    Returns:
        Tuple containing:
            - recovery_time: Smallest recovered gap found, or `np.inf` if
              recovery is not reached by `max_gap`.
            - recovery_difference: Relative difference at the returned gap.
            - recovery_peak: Test-pulse peak at the returned gap.
            - info: Only when `return_info` is True. Dictionary containing
              recovery simulation arrays.
    """
    points_per_time = 100
    train_pulses = int(habituation_time)

    on_steps = int(Ton * points_per_time)
    period_steps = int(T * points_per_time)
    on_grid = np.arange(on_steps + 1) / points_per_time
    off_grid = np.arange(period_steps - on_steps) / points_per_time

    on_input = np.asarray([A])
    off_input = np.asarray([0.0])
    train_inputs = [[value for _ in range(train_pulses) for value in (on_input, off_input)]]
    train_segments = [grid for _ in range(train_pulses) for grid in (on_grid, off_grid)]

    _, train_trajectories, train_outputs, _ = iocrn.transient_response_piecewise(
        u_nested_list=train_inputs,
        x0_list=[x0],
        nested_time_horizon=train_segments,
        force=True,
    )
    habituated_state = train_trajectories[0][:, -1]

    first_period_samples = len(on_grid) + len(off_grid)
    first_peak = train_outputs[0][0, :first_period_samples].max()

    test_inputs, test_segments = rectangular_pulse_program(A, T, Ton, 1)

    def probe(gap):
        gap_grid = np.asarray([0.0, float(gap)])
        inputs = [[off_input] + test_inputs[0]]
        segments = [gap_grid] + test_segments
        time, _, outputs, _ = iocrn.transient_response_piecewise(
            u_nested_list=inputs,
            x0_list=[habituated_state],
            nested_time_horizon=segments,
            force=True,
        )
        test_peak = outputs[0][0][time >= gap].max()
        relative_difference = abs(test_peak - first_peak) / max(abs(first_peak), 1e-12)
        return relative_difference <= recovery_tolerance, relative_difference, test_peak

    def simulate_full_protocol(gap):
        gap_grid = np.asarray([0.0, float(gap)])
        inputs = train_inputs[0] + [off_input] + test_inputs[0]
        segments = train_segments + [gap_grid] + test_segments
        time, trajectories, outputs, _ = iocrn.transient_response_piecewise(
            u_nested_list=[inputs],
            x0_list=[x0],
            nested_time_horizon=segments,
            force=True,
        )
        input_signal = _piecewise_input_signal(time, inputs, segments)
        return {
            "time": time,
            "input_signal": input_signal,
            "response": outputs[0][0],
            "trajectory": trajectories[0],
        }

    recovered, relative_difference, test_peak = probe(max_gap)
    if not recovered:
        if return_info:
            info = simulate_full_protocol(max_gap)
            return np.inf, relative_difference, test_peak, info
        return np.inf, relative_difference, test_peak

    low, high = 0.0, float(max_gap)
    best = (high, relative_difference, test_peak)
    for _ in range(search_depth):
        midpoint = 0.5 * (low + high)
        recovered, relative_difference, test_peak = probe(midpoint)
        if recovered:
            high = midpoint
            best = (high, relative_difference, test_peak)
        else:
            low = midpoint
    if return_info:
        info = simulate_full_protocol(best[0])
        return best + (info,)
    return best


def render_habituation(iocrn, figsize=None):
    """Render cached habituation hallmark simulations.

    The function reads `iocrn.last_task_info`, as produced by
    `habituation_hallmarks_loss`, and creates a single dashboard figure with
    the same plotting style used in `test_hallmarks.ipynb`. Each hallmark
    section title includes the corresponding component loss.

    Args:
        iocrn: IOCRN-like object whose `last_task_info` contains the combined
            habituation hallmark cache.
        figsize: Optional matplotlib figure size. If None, choose a height
            based on the number of panels.

    Returns:
        Tuple `(fig, axes)` containing the matplotlib figure and axes.
    """
    import matplotlib.pyplot as plt

    task_info = getattr(iocrn, "last_task_info", None)
    if not isinstance(task_info, dict):
        raise ValueError("No last_task_info found on the IOCRN.")

    info = task_info.get("hallmark_info", {})
    hallmark_info = info.get("hallmark_info", info)
    component_losses = task_info.get("component_losses", info.get("component_losses", {}))
    if not isinstance(hallmark_info, dict) or not hallmark_info:
        raise ValueError("No habituation hallmark info found in last_task_info.")

    A = float(info.get("A", np.nan))
    T = float(info.get("T", np.nan))
    h1_info = hallmark_info.get("hallmark1", {})
    h2_info = hallmark_info.get("hallmark2", {})
    h3_info = hallmark_info.get("hallmark3", {})
    h4_info = hallmark_info.get("hallmark4", {})
    h5_info = hallmark_info.get("hallmark5", {})
    h6_info = hallmark_info.get("hallmark6", {})

    n_h4 = len(h4_info.get("runs", []))
    n_h5 = len(h5_info.get("runs", []))
    n_panels = 3 + n_h4 + n_h5 + 2
    if figsize is None:
        figsize = (12, max(3.0 * n_panels, 6.0))
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=False, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ax_idx = 0

    def loss_text(name):
        loss = component_losses.get(name, np.nan)
        return f"loss={float(loss):.4g}" if np.isfinite(loss) else "loss=nan"

    def response_max(response):
        response = np.asarray(response, dtype=float)
        if response.size == 0:
            return 1.0
        value = float(np.nanmax(response))
        return max(value, 1e-12) if np.isfinite(value) else 1.0

    def plot_response_with_input(ax, time, response, input_signal, title):
        response = np.asarray(response, dtype=float)
        input_signal = np.asarray(input_signal, dtype=float)
        ax.plot(time, response, label="response", color="C0")
        max_response = response_max(response)
        if input_signal.size:
            scaled_input = input_signal / max(float(np.nanmax(input_signal)), 1e-12) * max_response
            ax.step(time, scaled_input, where="post", label="input signal (scaled)", color="C1", alpha=0.4)
        ax.set_title(title)
        ax.set_ylabel("response")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="upper right")
        return max_response

    def annotate_marker(ax, x, y, text, color="black", xytext=(8, -18)):
        if not np.isfinite(x) or not np.isfinite(y):
            return
        ax.axvline(x, color=color, linestyle="--", linewidth=1.5)
        ax.annotate(
            text,
            xy=(x, y),
            xytext=xytext,
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", linestyle="--", color=color),
            ha="left",
            va="top",
            color=color,
            fontsize=8,
            annotation_clip=True,
        )

    # Hallmark 1.
    ax = axes[ax_idx]
    ax_idx += 1
    if h1_info:
        time = np.asarray(h1_info["time"], dtype=float)
        response = np.asarray(h1_info["response"], dtype=float)
        max_response = plot_response_with_input(
            ax,
            time,
            response,
            h1_info.get("input_signal", np.zeros_like(time)),
            f"Hallmark 1: Progressive Response Decrement ({loss_text('hallmark1')})",
        )
        habituation_time = h1_info.get("habituation_time", np.inf)
        if np.isfinite(habituation_time) and np.isfinite(T):
            habituation_time_abs = (float(habituation_time) - 1.0) * T
            annotate_marker(
                ax,
                habituation_time_abs,
                max_response,
                f"habituation time = {int(habituation_time)} pulses",
                color="black",
            )
    else:
        ax.set_title(f"Hallmark 1 ({loss_text('hallmark1')})")

    # Hallmark 2.
    ax = axes[ax_idx]
    ax_idx += 1
    if h2_info and "time" in h2_info and "response" in h2_info:
        time = np.asarray(h2_info["time"], dtype=float)
        response = np.asarray(h2_info["response"], dtype=float)
        max_response = plot_response_with_input(
            ax,
            time,
            response,
            h2_info.get("input_signal", np.zeros_like(time)),
            f"Hallmark 2: Spontaneous Recovery ({loss_text('hallmark2')})",
        )
        recovery_time = h2_info.get("recovery_time", np.inf)
        training_end = h1_info.get("habituation_time", np.inf) * T - 0.01
        recovery_end = training_end + recovery_time
        if np.isfinite(training_end):
            ax.axvline(training_end, color="black", linestyle="--", linewidth=1.5)
        if np.isfinite(recovery_end):
            annotate_marker(
                ax,
                recovery_end,
                max_response,
                f"recovery time = {recovery_time:.1f}",
                color="C3",
            )
    else:
        ax.set_title(f"Hallmark 2 ({loss_text('hallmark2')})")

    # Hallmark 3.
    ax = axes[ax_idx]
    ax_idx += 1
    if h3_info and "time" in h3_info and "response" in h3_info:
        time = np.asarray(h3_info["time"], dtype=float)
        response = np.asarray(h3_info["response"], dtype=float)
        max_response = plot_response_with_input(
            ax,
            time,
            response,
            h3_info.get("input_signal", np.zeros_like(time)),
            f"Hallmark 3: Potentiation of Habituation ({loss_text('hallmark3')})",
        )
        n_pulses = h3_info.get("n_pulses", 0)
        recovery_gap = h3_info.get("recovery_gap", 0.0)
        if np.isfinite(T):
            series_period = n_pulses * T + recovery_gap
            for series_idx, ht in enumerate(h3_info.get("habituation_times", [])):
                series_start = series_idx * series_period
                ax.axvline(series_start, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
                if np.isfinite(ht):
                    ht_time = series_start + (float(ht) - 1.0) * T
                    ax.axvline(ht_time, color="C3", linestyle="--", linewidth=1.2, alpha=0.8)
                    ax.text(ht_time, 0.9 * max_response, f"h={int(ht)}", color="C3", ha="left")
    else:
        ax.set_title(f"Hallmark 3 ({loss_text('hallmark3')})")

    # Hallmark 4.
    for idx, run in enumerate(h4_info.get("runs", [])):
        ax = axes[ax_idx]
        ax_idx += 1
        color = colors[idx % len(colors)]
        T_run = float(run.get("T", np.nan))
        h_run = h4_info.get("habituation_times", [np.nan] * n_h4)[idx]
        time = np.asarray(run["time"], dtype=float)
        response = np.asarray(run["response"], dtype=float)
        max_response = response_max(response)
        title = "Hallmark 4: Frequency Sensitivity"
        if idx == 0:
            title += f" ({loss_text('hallmark4')})"
        ax.plot(time, response, color=color, label=f"T={T_run:g}")
        if np.isfinite(h_run) and np.isfinite(T_run):
            ht_time = (float(h_run) - 1.0) * T_run
            ax.axvline(ht_time, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
            ax.text(ht_time, 0.9 * max_response, f"h={int(h_run)}", color=color, ha="left")
        ax.set_title(title)
        ax.set_ylabel("response")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="upper right")

    # Hallmark 5.
    for idx, run in enumerate(h5_info.get("runs", [])):
        ax = axes[ax_idx]
        ax_idx += 1
        color = colors[idx % len(colors)]
        A_run = float(run.get("A", np.nan))
        h_run = h5_info.get("habituation_times", [np.nan] * n_h5)[idx]
        time = np.asarray(run["time"], dtype=float)
        response = np.asarray(run["response"], dtype=float)
        max_response = response_max(response)
        title = "Hallmark 5: Intensity Sensitivity"
        if idx == 0:
            title += f" ({loss_text('hallmark5')})"
        ax.plot(time, response, color=color, label=f"A={A_run:g}")
        if np.isfinite(h_run) and np.isfinite(T):
            ht_time = (float(h_run) - 1.0) * T
            ax.axvline(ht_time, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
            ax.text(ht_time, 0.9 * max_response, f"h={int(h_run)}", color=color, ha="left")
        ax.set_title(title)
        ax.set_ylabel("response")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="upper right")

    # Hallmark 6.
    normal_training_end = h1_info.get("habituation_time", np.inf) * T - 0.01
    continued_training_end = h6_info.get(
        "stricter_habituation_time",
        h6_info.get("continued_habituation_time", np.inf),
    ) * T - 0.01
    h6_plots = [
        (h2_info, h6_info.get("recovery_time", np.inf), normal_training_end, "C2", "normal recovery"),
        (
            h6_info,
            h6_info.get("continued_recovery_time", h6_info.get("stricter_recovery_time", np.inf)),
            continued_training_end,
            "C3",
            "continued stimulation",
        ),
    ]
    for idx, (plot_info, recovery_time, training_end, color, label) in enumerate(h6_plots):
        ax = axes[ax_idx]
        ax_idx += 1
        title = label
        if idx == 0:
            title = f"Hallmark 6: Subliminal Accumulation ({loss_text('hallmark6')})"
        if plot_info and "time" in plot_info and "response" in plot_info:
            time = np.asarray(plot_info["time"], dtype=float)
            response = np.asarray(plot_info["response"], dtype=float)
            max_response = plot_response_with_input(
                ax,
                time,
                response,
                plot_info.get("input_signal", np.zeros_like(time)),
                title,
            )
            recovery_end = training_end + recovery_time
            if np.isfinite(training_end):
                ax.axvline(training_end, color="black", linestyle="--", linewidth=1.5)
            if np.isfinite(recovery_end):
                annotate_marker(
                    ax,
                    recovery_end,
                    0.85 * max_response,
                    f"{label} RT = {recovery_time:.1f}",
                    color=color,
                )
        else:
            ax.set_title(title)

    for ax in axes:
        ax.set_xlabel("time")
    fig.suptitle("Habituation Hallmarks")
    return fig, axes


def render_hallmark_loss_summary(iocrn, figsize=(9, 4)):
    """Render the six weighted hallmark losses for one IOCRN.

    The plot reads the cached `component_losses` and `component_weights`
    produced by `habituation_hallmarks_loss`. It is intended for RL logging:
    all six components are shown for the same CRN, with each legend entry
    reporting the weight used in the total objective.

    Args:
        iocrn: IOCRN-like object with populated `last_task_info`.
        figsize: Matplotlib figure size.

    Returns:
        Tuple `(fig, ax)` containing the matplotlib figure and axis.
    """
    import matplotlib.pyplot as plt

    task_info = getattr(iocrn, "last_task_info", None)
    if not isinstance(task_info, dict):
        raise ValueError("No last_task_info found on the IOCRN.")

    info = task_info.get("hallmark_info", {})
    component_losses = task_info.get("component_losses", info.get("component_losses", {}))
    component_weights = task_info.get("component_weights", info.get("component_weights", {}))
    if not isinstance(component_losses, dict) or not component_losses:
        raise ValueError("No component losses found in last_task_info.")

    names = [f"hallmark{i}" for i in range(1, 7)]
    labels = [f"H{i}" for i in range(1, 7)]
    losses = np.asarray([float(component_losses.get(name, np.nan)) for name in names], dtype=float)
    weights = np.asarray([float(component_weights.get(name, 1.0)) for name in names], dtype=float)
    weighted_losses = losses * weights

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x = np.arange(len(names))
    for idx, (label, loss, weighted_loss, weight) in enumerate(zip(labels, losses, weighted_losses, weights)):
        color = colors[idx % len(colors)]
        ax.bar(
            x[idx],
            weighted_loss,
            color=color,
            label=f"{label}: w={weight:g}",
        )
        if np.isfinite(weighted_loss):
            ax.text(
                x[idx],
                weighted_loss,
                f"{loss:.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("weighted loss")
    ax.set_title("Hallmark Loss Components")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    return fig, ax


def _piecewise_input_signal(time, inputs, segments):
    """Sample a one-dimensional piecewise-constant input on a stitched time grid."""
    input_signal = np.zeros_like(time, dtype=float)
    start = 0.0
    for input_value, segment in zip(inputs, segments):
        end = start + float(np.asarray(segment, dtype=float)[-1])
        mask = (time >= start) & (time <= end)
        input_signal[mask] = float(np.asarray(input_value, dtype=float).reshape(-1)[0])
        start = end
    return input_signal
