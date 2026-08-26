import numpy as np

from apps.habituation.hallmarks_helpers import (
    compute_habituation_time,
    compute_peak_tube_decrements,
    compute_peak_window_decrements,
    compute_recovery_time,
    extract_period_peaks,
    extract_period_troughs,
    rectangular_pulse_program,
)


def hallmark1_loss(
    iocrn,
    A,
    T,
    Ton,
    n_pulses,
    x0,
    tolerance=0.01,
    n_dec=1,
    habituation_time_method="window",
    n_min=4,
    i_max=2,
    first_peak_fraction=0.5,
    monotone_drop_tol=0.0,
    n_post_min=2,
    n_peak_trough_checks=5,
    delta_min=0.8,
    delta_t=0.05,
    trough_thr=0.6,
    trough_tail=0.02,
    trough_count_thr=0.10,
    trough_count_max=5,
    level_weight=1.0,
    time_weight=1.0,
    alpha=0.1,
    eps=1e-12,
    LARGE_NUMBER=1e4,
    return_info=False,
):
    """Compute a habituation-time loss for habituation hallmark 1.

    Hallmark 1 is defined as repeated application of a stimulus producing
    a progressive decrease in some parameter of a response to an asymptotic
    level. This function first applies a list of trajectory filters, then 
    scores the accepted trajectory using habituation time.

    For peaks p_0, ..., p_n, the default habituation-time method lets h be the
    first index satisfying

        (p_{h-1} - p_h) / p_{h-1} < tolerance.

    If `n_dec > 1`, the criterion is applied to the max-min range over a window
    of `n_dec + 1` consecutive peaks, and h is the pulse index at the end of the
    first qualifying window. If `habituation_time_method="tube"`, h is instead
    the first pulse index such that all remaining peaks fit inside a relative
    tube of width `tolerance`.

    The loss combines a habituated-level term and a habituation-time term:

        alpha * (level_weight * p_h / p_star
                 + time_weight * h / (n_pulses - 1))
                / (level_weight + time_weight),

    clipped to [0, 1]. If no habituation time is found, or if both weights are
    zero, the loss is 1.

    Args:
        iocrn: IOCRN-like object implementing `transient_response_piecewise`.
        A: Input amplitude during each ON segment.
        T: Duration of one complete pulse period.
        Ton: Duration of the ON part of each pulse.
        n_pulses: Number of pulse periods to evaluate.
        x0: Initial condition for the simulation.
        tolerance: Relative decrement threshold used to detect habituation.
        n_dec: Number of peak-to-peak intervals in the max-min window used to
            detect habituation. For the tube method, this is the minimum number
            of peak-to-peak intervals that must remain after the detected
            pulse.
        habituation_time_method: Either `"window"` for the historical local
            window criterion or `"tube"` for the suffix-tube criterion.
        n_min: Minimum acceptable habituation time. If the detected
            habituation time is smaller than this, the loss is one.
        i_max: Latest allowed zero-based index of the maximum peak.
        first_peak_fraction: Minimum allowed first-peak/max-peak ratio.
        monotone_drop_tol: Minimum relative post-maximum drop. The paper's
            monotonic decrease condition corresponds to zero.
        n_post_min: Minimum number of peaks required after the maximum peak.
        n_peak_trough_checks: Number of post-maximum peak-trough separations
            checked against `delta_t`.
        delta_min: Maximum allowed minimum post-maximum peak/max-peak ratio.
        delta_t: Minimum peak-trough separation relative to the maximum peak.
        trough_thr: Maximum allowed early trough/max-peak ratio.
        trough_tail: Maximum allowed final trough/max-peak ratio.
        trough_count_thr: Threshold for counting high troughs.
        trough_count_max: Maximum allowed number of high troughs.
        level_weight: Weight on the habituated response level term.
        time_weight: Weight on the habituation time term.
        alpha: Global weight applied to the whole hallmark 1 loss.
        eps: Small positive value used to avoid division by zero.
        LARGE_NUMBER: Invalid peak sentinel.
        return_info: If True, return `(loss, info)`. Otherwise return only
            `loss`.

    Returns:
        Scalar loss in [0, 1], or `(loss, info)` when `return_info` is True.
    """
    def finish(loss, info):
        loss = float(np.clip(loss, 0.0, 1.0))
        return (loss, info) if return_info else loss

    u_program, time_segments = rectangular_pulse_program(A, T, Ton, n_pulses)
    time, _, outputs, _ = iocrn.transient_response_piecewise(
        u_nested_list=u_program,
        x0_list=[x0],
        nested_time_horizon=time_segments,
        force=True,
    )
    response = outputs[0][0]
    input_signal = np.zeros_like(time, dtype=float)
    for pulse_idx in range(int(n_pulses)):
        on_mask = (time >= pulse_idx * T) & (time <= pulse_idx * T + Ton)
        input_signal[on_mask] = A

    habituation_time, peaks = compute_habituation_time(
        time,
        response,
        T,
        n_pulses,
        tolerance=tolerance,
        blowup_threshold=LARGE_NUMBER,
        n_dec=n_dec,
        habituation_time_method=habituation_time_method,
    )
    info = {
        "habituation_time": habituation_time,
        "time": time,
        "input_signal": input_signal,
        "response": response,
        "peaks": peaks,
        "n_dec": int(n_dec),
        "habituation_time_method": str(habituation_time_method),
    }
    if np.isfinite(habituation_time) and int(habituation_time) < int(n_min):
        info["too_early_habituation"] = True
        info["n_min"] = int(n_min)

    peaks = np.asarray(peaks, dtype=float)
    troughs = extract_period_troughs(time, response, T, n_pulses)
    info["troughs"] = troughs
    relative_decrements = np.divide(
        peaks[:-1] - peaks[1:],
        peaks[:-1],
        out=np.full(max(peaks.size - 1, 0), np.inf, dtype=float),
        where=np.abs(peaks[:-1]) > eps,
    )
    info["relative_decrements"] = relative_decrements
    (
        window_decrements,
        window_maxima,
        window_minima,
        window_is_nonincreasing,
    ) = compute_peak_window_decrements(peaks, n_dec=n_dec, eps=eps)
    info["window_relative_decrements"] = window_decrements
    info["window_peak_maxima"] = window_maxima
    info["window_peak_minima"] = window_minima
    info["window_is_nonincreasing"] = window_is_nonincreasing
    tube_decrements, tube_maxima, tube_minima = compute_peak_tube_decrements(
        peaks,
        min_remaining_intervals=n_dec,
        eps=eps,
    )
    info["tube_relative_decrements"] = tube_decrements
    info["tube_peak_maxima"] = tube_maxima
    info["tube_peak_minima"] = tube_minima

    # 1. At least two peaks and two troughs are required.
    if peaks.size < 2 or not np.all(np.isfinite(peaks)):
        return finish(1.0, info)
    if troughs.size < 2 or not np.all(np.isfinite(troughs)):
        return finish(1.0, info)
    if np.any(peaks >= LARGE_NUMBER) or np.any(troughs >= LARGE_NUMBER):
        return finish(1.0, info)

    i_star = int(np.argmax(peaks))
    p_star = max(float(peaks[i_star]), eps)
    first_peak = max(float(peaks[0]), eps)
    tail = peaks[i_star:]
    tail_troughs = troughs[i_star:min(troughs.size, peaks.size)]

    # 2. The maximum peak must occur no later than i_max
    if i_star > int(i_max):
        return finish(1.0, info)
    
    # 3. The first peak must be at least a fraction of the maximum peak.
    if first_peak / p_star < float(first_peak_fraction):
        return finish(1.0, info)

    # 4. The tail must be monotone decreasing within the specified tolerance.
    relative_tail_drops = (tail[:-1] - tail[1:]) / p_star
    monotone_slack = max(float(tolerance), eps)
    if np.any(relative_tail_drops < float(monotone_drop_tol) - monotone_slack):
        return finish(1.0, info)
    
    # 5. The tail must contain at least n_post_min + 1 peaks.
    if tail.size < int(n_post_min) + 1:
        return finish(1.0, info)
    
    # 6. The minimum peak in the tail must be less than delta_min times the maximum peak.
    if np.min(tail[1:]) / p_star > float(delta_min):
        return finish(1.0, info)

    # 7. The separation between each of the first n_peak_trough_checks troughs and peaks must be at least delta_t.
    k_sep = min(int(n_peak_trough_checks), tail.size, tail_troughs.size)
    if k_sep < 1:
        return finish(1.0, info)
    separations = (tail[:k_sep] - tail_troughs[:k_sep]) / p_star
    if np.min(separations) <= float(delta_t):
        return finish(1.0, info)
    
    # 8. The maximum trough in the tail must be less than trough_thr times the maximum peak.
    if np.max(tail_troughs / p_star) > float(trough_thr):
        return finish(1.0, info)
    
    # 9. The final trough must be less than trough_tail times the maximum peak.
    if float(troughs[-1]) / p_star > float(trough_tail):
        return finish(1.0, info)
    
    # 10. The number of troughs exceeding trough_count_thr times the maximum peak must be at most trough_count_max.
    high_trough_count = int(np.sum((troughs / p_star) > float(trough_count_thr)))
    if high_trough_count > int(trough_count_max):
        return finish(1.0, info)

    if not np.isfinite(habituation_time):
        return finish(1.0, info)
    if int(habituation_time) < int(n_min):
        return finish(1.0, info)

    habituation_index = int(habituation_time)
    if habituation_index >= peaks.size:
        return finish(1.0, info)
    if habituation_index <= i_star:
        return finish(1.0, info)

    time_loss = habituation_index / max(int(n_pulses) - 1, 1)
    level_loss = float(peaks[habituation_index]) / p_star
    total_weight = float(level_weight) + float(time_weight)
    if total_weight <= 0.0:
        return finish(1.0, info)
    loss = (
        float(level_weight) * level_loss
        + float(time_weight) * time_loss
    ) / total_weight
    loss *= float(alpha)
    return finish(loss, info)


def hallmark2_loss(
    iocrn,
    A,
    T,
    Ton,
    habituation_time,
    x0,
    recovery_tolerance=0.05,
    max_gap=4000.0,
    search_depth=16,
    return_info=False,
):
    """Compute a spontaneous-recovery loss for habituation hallmark 2.

    Hallmark 2 says that if the stimulus is withheld after response decrement,
    the response recovers at least partially over the observation time. This
    function uses `compute_recovery_time`, which searches for the smallest OFF
    gap after habituation whose test-pulse response is within
    `recovery_tolerance` of the first pulse response.

    The loss is

        recovery_time / max_gap,

    clipped to [0, 1]. If recovery is not reached by `max_gap`, the loss is 1.

    Args:
        iocrn: IOCRN-like object implementing `transient_response_piecewise`.
        A: Input amplitude during each ON segment.
        T: Duration of one complete pulse period.
        Ton: Duration of the ON part of each pulse.
        habituation_time: Number of pulse periods used for the training phase.
        x0: Initial condition for the training phase.
        recovery_tolerance: Relative tolerance for recovery to the first peak.
        max_gap: Largest OFF gap treated as the observation window.
        search_depth: Number of binary-search refinements.
        return_info: If True, return `(loss, info)`. Otherwise return only
            `loss`.

    Returns:
        Scalar loss in [0, 1], or `(loss, info)` when `return_info` is True.
    """
    if not np.isfinite(habituation_time):
        info = {"recovery_time": np.inf}
        return (1.0, info) if return_info else 1.0

    recovery_time, _, _, recovery_info = compute_recovery_time(
        iocrn,
        A,
        T,
        Ton,
        habituation_time,
        x0,
        recovery_tolerance=recovery_tolerance,
        max_gap=max_gap,
        search_depth=search_depth,
        return_info=True,
    )
    info = {"recovery_time": recovery_time}
    info.update(recovery_info)

    if not np.isfinite(recovery_time):
        return (1.0, info) if return_info else 1.0
    if float(max_gap) <= 0.0:
        return (1.0, info) if return_info else 1.0

    loss = float(recovery_time) / float(max_gap)
    loss = float(np.clip(loss, 0.0, 1.0))
    return (loss, info) if return_info else loss


def hallmark3_loss(
    iocrn,
    A,
    T,
    Ton,
    habituation_time,
    recovery_time,
    x0,
    n_series=2,
    recovery_gap_fraction=0.5,
    tolerance=0.01,
    n_dec=1,
    habituation_time_method="window",
    margin=0.0,
    eps=1e-12,
    return_info=False,
):
    """Compute a potentiation loss for habituation hallmark 3.

    Hallmark 3 says that after multiple series of stimulus repetitions and
    spontaneous recoveries, the response decrement becomes successively more
    rapid and/or more pronounced. This function simulates repeated pulse
    trains separated by OFF recovery gaps, then scores the ratios of
    consecutive habituation times. Each train contains `habituation_time`
    pulses. The OFF gap between trains is
    `recovery_gap_fraction * recovery_time`.

    For habituation times h_0, ..., h_{n-1}, the loss is

        mean(min(h_{i+1} / h_i, 1))

    so smaller values mean stronger potentiation.

    Args:
        iocrn: IOCRN-like object implementing `transient_response_piecewise`.
        A: Input amplitude during each ON segment.
        T: Duration of one complete pulse period.
        Ton: Duration of the ON part of each pulse.
        habituation_time: Baseline habituation time. This also sets the number
            of pulses in each stimulus series.
        recovery_time: Recovery time used to set the OFF gap between series.
        x0: Initial condition for the first series.
        n_series: Number of repeated stimulus series.
        recovery_gap_fraction: Fraction of `recovery_time` used as the OFF gap
            between series.
        tolerance: Relative decrement threshold used to detect habituation.
        n_dec: Number of peak-to-peak intervals in the max-min window used to
            detect habituation. For the tube method, this is the minimum number
            of peak-to-peak intervals that must remain after the detected
            pulse.
        habituation_time_method: Either `"window"` for the historical local
            window criterion or `"tube"` for the suffix-tube criterion.
        margin: Deprecated compatibility argument. The ratio loss does not use
            it.
        eps: Small positive value used to avoid division by zero.
        return_info: If True, return `(loss, info)`. Otherwise return only
            `loss`.

    Returns:
        Scalar loss in [0, 1], or `(loss, info)` when `return_info` is True.
    """
    def finish(loss, info):
        loss = float(np.clip(loss, 0.0, 1.0))
        return (loss, info) if return_info else loss

    n_series = int(n_series)
    if not np.isfinite(habituation_time) or not np.isfinite(recovery_time):
        return finish(1.0, {})
    n_pulses = int(habituation_time)
    if n_series < 2 or n_pulses < 2:
        return finish(1.0, {})

    recovery_gap = float(recovery_gap_fraction) * float(recovery_time)
    if recovery_gap < 0.0 or not np.isfinite(recovery_gap):
        return finish(1.0, {})

    one_train_inputs, one_train_segments = rectangular_pulse_program(A, T, Ton, n_pulses)
    off_input = np.asarray([0.0])
    gap_segment = np.asarray([0.0, recovery_gap])

    inputs = []
    segments = []
    series_starts = []
    current_start = 0.0
    for series_idx in range(n_series):
        series_starts.append(current_start)
        inputs.extend(one_train_inputs[0])
        segments.extend(one_train_segments)
        current_start += float(n_pulses) * float(T)
        if series_idx < n_series - 1:
            inputs.append(off_input)
            segments.append(gap_segment)
            current_start += recovery_gap

    time, trajectories, outputs, _ = iocrn.transient_response_piecewise(
        u_nested_list=[inputs],
        x0_list=[x0],
        nested_time_horizon=segments,
        force=True,
    )
    response = outputs[0][0]
    input_signal = _sample_piecewise_input_signal(time, inputs, segments)

    habituation_times = [float(habituation_time)]
    peak_lists = []
    for series_idx, series_start in enumerate(series_starts):
        series_end = series_start + float(n_pulses) * float(T)
        mask = (time >= series_start) & (time <= series_end)
        local_time = time[mask] - series_start
        local_response = response[mask]
        series_habituation_time, peaks = compute_habituation_time(
            local_time,
            local_response,
            T,
            n_pulses,
            tolerance=tolerance,
            n_dec=n_dec,
            habituation_time_method=habituation_time_method,
        )
        peaks = np.asarray(peaks, dtype=float)
        peak_lists.append(peaks)
        if series_idx > 0:
            habituation_times.append(series_habituation_time)

    habituation_times = np.asarray(habituation_times, dtype=float)
    info = {
        "habituation_times": habituation_times,
        "recovery_gap": recovery_gap,
        "n_pulses": n_pulses,
        "time": time,
        "input_signal": input_signal,
        "response": response,
        "trajectory": trajectories[0],
        "peaks": peak_lists,
        "n_dec": int(n_dec),
        "habituation_time_method": str(habituation_time_method),
    }

    if not np.all(np.isfinite(habituation_times)):
        return finish(1.0, info)

    ratio_terms = habituation_times[1:] / np.maximum(habituation_times[:-1], eps)
    transition_losses = np.minimum(ratio_terms, 1.0)
    info["ratio_terms"] = ratio_terms
    info["transition_losses"] = transition_losses

    return finish(float(np.mean(transition_losses)), info)


def hallmark4_loss(
    iocrn,
    A,
    T_values,
    Ton,
    n_pulses,
    x0,
    tolerance=0.01,
    n_dec=1,
    habituation_time_method="window",
    include_recovery=False,
    recovery_tolerance=0.05,
    max_gap=4000.0,
    search_depth=16,
    habituation_weight=1.0,
    recovery_weight=1.0,
    eps=1e-12,
    return_info=False,
):
    """Compute the frequency-sensitivity loss for hallmark 4.

    The default behavior is the historical Equation 1 frequency term

        ht_f1 / ht_f2 + ht_f2 / ht_f3

    where f1 > f2 > f3. Since f = 1 / T, this function sorts `T_values` in
    increasing order and averages the two adjacent habituation-time ratios:

        0.5 * (ht(T_1) / ht(T_2) + ht(T_2) / ht(T_3)).

    If `include_recovery` is True, the loss also includes the recovery-time
    part of hallmark 4: higher-frequency stimulation should lead to faster
    spontaneous recovery after the response decrement has reached its
    asymptotic level. In that case the function also computes

        mean(rt(T_i) / rt(T_{i+1}))

    over adjacent sorted periods, and combines the habituation-time and
    recovery-time terms using `habituation_weight` and `recovery_weight`.

    Args:
        iocrn: IOCRN-like object implementing `transient_response_piecewise`.
        A: Input amplitude during each ON segment.
        T_values: Sequence of pulse periods to compare.
        Ton: Duration of the ON part of each pulse.
        n_pulses: Number of pulse periods to evaluate for habituation.
        x0: Initial condition for each simulation.
        tolerance: Relative decrement threshold used to detect habituation.
        n_dec: Number of peak-to-peak intervals in the max-min window used to
            detect habituation. For the tube method, this is the minimum number
            of peak-to-peak intervals that must remain after the detected
            pulse.
        habituation_time_method: Either `"window"` for the historical local
            window criterion or `"tube"` for the suffix-tube criterion.
        include_recovery: If True, also penalize failure of higher-frequency
            stimulation to produce shorter recovery times. If False, preserve
            the previous habituation-time-only loss.
        recovery_tolerance: Relative recovery tolerance passed to
            `compute_recovery_time` when `include_recovery` is True.
        max_gap: Maximum OFF gap tested for recovery.
        search_depth: Number of binary-search refinements for recovery time.
        habituation_weight: Weight on the habituation-time frequency term when
            `include_recovery` is True.
        recovery_weight: Weight on the recovery-time frequency term when
            `include_recovery` is True.
        eps: Small positive value used to avoid division by zero.
        return_info: If True, return `(loss, info)`. Otherwise return only
            `loss`.

    Returns:
        Frequency-sensitivity term from Equation 1, or `(loss, info)` when
        `return_info` is True.
    """
    def finish(loss, info):
        loss = float(loss)
        return (loss, info) if return_info else loss

    T_values = np.asarray(T_values, dtype=float).reshape(-1)
    if T_values.size < 2 or not np.all(np.isfinite(T_values)):
        return finish(np.inf, {})
    if np.any(T_values <= float(Ton)):
        return finish(np.inf, {})

    order = np.argsort(T_values)
    periods = T_values[order]

    habituation_times = []
    runs = []
    for T in periods:
        u_program, time_segments = rectangular_pulse_program(A, T, Ton, n_pulses)
        time, _, outputs, _ = iocrn.transient_response_piecewise(
            u_nested_list=u_program,
            x0_list=[x0],
            nested_time_horizon=time_segments,
            force=True,
        )
        response = outputs[0][0]
        habituation_time, peaks = compute_habituation_time(
            time,
            response,
            T,
            n_pulses,
            tolerance=tolerance,
            n_dec=n_dec,
            habituation_time_method=habituation_time_method,
        )

        habituation_times.append(habituation_time)
        runs.append({
            "T": T,
            "time": time,
            "response": response,
            "peaks": peaks,
        })

    habituation_times = np.asarray(habituation_times, dtype=float)
    if np.all(np.isfinite(habituation_times)):
        ratio_terms = habituation_times[:-1] / np.maximum(habituation_times[1:], eps)
    else:
        ratio_terms = np.full(max(habituation_times.size - 1, 0), np.nan, dtype=float)
    habituation_loss = float(0.5 * np.sum(ratio_terms))
    info = {
        "T_values": periods,
        "frequencies": 1.0 / periods,
        "habituation_times": habituation_times,
        "ratio_terms": ratio_terms,
        "habituation_ratio_terms": ratio_terms,
        "habituation_loss": habituation_loss,
        "runs": runs,
        "n_dec": int(n_dec),
        "habituation_time_method": str(habituation_time_method),
        "include_recovery": bool(include_recovery),
    }

    if not np.all(np.isfinite(habituation_times)):
        return finish(np.inf, info)

    if not include_recovery:
        return finish(habituation_loss, info)

    recovery_times = []
    recovery_runs = []
    for T, habituation_time in zip(periods, habituation_times):
        recovery_time, recovery_difference, recovery_peak, recovery_info = compute_recovery_time(
            iocrn,
            A,
            T,
            Ton,
            habituation_time,
            x0,
            recovery_tolerance=recovery_tolerance,
            max_gap=max_gap,
            search_depth=search_depth,
            return_info=True,
        )
        recovery_times.append(recovery_time)
        recovery_runs.append({
            "T": T,
            "recovery_time": recovery_time,
            "recovery_difference": recovery_difference,
            "recovery_peak": recovery_peak,
            **recovery_info,
        })

    recovery_times = np.asarray(recovery_times, dtype=float)
    if np.all(np.isfinite(recovery_times)):
        recovery_ratio_terms = recovery_times[:-1] / np.maximum(recovery_times[1:], eps)
        recovery_loss = float(0.5 * np.sum(recovery_ratio_terms))
    else:
        recovery_ratio_terms = np.full(max(recovery_times.size - 1, 0), np.nan, dtype=float)
        recovery_loss = np.inf

    info["recovery_times"] = recovery_times
    info["recovery_ratio_terms"] = recovery_ratio_terms
    info["recovery_loss"] = recovery_loss
    info["recovery_runs"] = recovery_runs
    info["recovery_tolerance"] = float(recovery_tolerance)
    info["max_gap"] = float(max_gap)
    info["search_depth"] = int(search_depth)

    if not np.isfinite(recovery_loss):
        return finish(np.inf, info)

    total_weight = float(habituation_weight) + float(recovery_weight)
    if total_weight <= 0.0:
        return finish(np.inf, info)

    loss = (
        float(habituation_weight) * habituation_loss
        + float(recovery_weight) * recovery_loss
    ) / total_weight
    info["habituation_weight"] = float(habituation_weight)
    info["recovery_weight"] = float(recovery_weight)
    return finish(loss, info)


def hallmark5_loss(
    iocrn,
    A_values,
    T,
    Ton,
    n_pulses,
    x0,
    tolerance=0.01,
    n_dec=1,
    habituation_time_method="window",
    eps=1e-12,
    return_info=False,
):
    """Compute the Equation 1 intensity-sensitivity term.

    Equation 1 uses the intensity term

        ht_A1 / ht_A2 + ht_A2 / ht_A3

    where A1 < A2 < A3. This function sorts `A_values` in increasing order and
    averages the two adjacent habituation-time ratios:

        0.5 * (ht(A_1) / ht(A_2) + ht(A_2) / ht(A_3)).

    Args:
        iocrn: IOCRN-like object implementing `transient_response_piecewise`.
        A_values: Sequence of input amplitudes to compare.
        T: Duration of one complete pulse period.
        Ton: Duration of the ON part of each pulse.
        n_pulses: Number of pulse periods to evaluate for habituation.
        x0: Initial condition for each simulation.
        tolerance: Relative decrement threshold used to detect habituation.
        n_dec: Number of peak-to-peak intervals in the max-min window used to
            detect habituation. For the tube method, this is the minimum number
            of peak-to-peak intervals that must remain after the detected
            pulse.
        habituation_time_method: Either `"window"` for the historical local
            window criterion or `"tube"` for the suffix-tube criterion.
        eps: Small positive value used to avoid division by zero.
        return_info: If True, return `(loss, info)`. Otherwise return only
            `loss`.

    Returns:
        Intensity-sensitivity term from Equation 1, or `(loss, info)` when
        `return_info` is True.
    """
    def finish(loss, info):
        loss = float(loss)
        return (loss, info) if return_info else loss

    A_values = np.asarray(A_values, dtype=float).reshape(-1)
    if A_values.size < 2 or not np.all(np.isfinite(A_values)):
        return finish(np.inf, {})
    if np.any(A_values <= 0.0):
        return finish(np.inf, {})

    amplitudes = A_values[np.argsort(A_values)]

    habituation_times = []
    runs = []
    for A in amplitudes:
        u_program, time_segments = rectangular_pulse_program(A, T, Ton, n_pulses)
        time, _, outputs, _ = iocrn.transient_response_piecewise(
            u_nested_list=u_program,
            x0_list=[x0],
            nested_time_horizon=time_segments,
            force=True,
        )
        response = outputs[0][0]
        habituation_time, peaks = compute_habituation_time(
            time,
            response,
            T,
            n_pulses,
            tolerance=tolerance,
            n_dec=n_dec,
            habituation_time_method=habituation_time_method,
        )

        habituation_times.append(habituation_time)
        runs.append({
            "A": A,
            "time": time,
            "response": response,
            "peaks": peaks,
        })

    habituation_times = np.asarray(habituation_times, dtype=float)
    if np.all(np.isfinite(habituation_times)):
        ratio_terms = habituation_times[:-1] / np.maximum(habituation_times[1:], eps)
    else:
        ratio_terms = np.full(max(habituation_times.size - 1, 0), np.nan, dtype=float)
    info = {
        "A_values": amplitudes,
        "habituation_times": habituation_times,
        "ratio_terms": ratio_terms,
        "runs": runs,
        "n_dec": int(n_dec),
        "habituation_time_method": str(habituation_time_method),
    }

    if not np.all(np.isfinite(habituation_times)):
        return finish(np.inf, info)

    return finish(float(0.5 * np.sum(ratio_terms)), info)


def hallmark6_loss(
    iocrn,
    A,
    T,
    Ton,
    habituation_time,
    recovery_time,
    x0,
    stricter_recovery_tolerance=None,
    stricter_habituation_tolerance=0.005,
    recovery_tolerance=0.05,
    n_dec=1,
    habituation_time_method="window",
    max_continued_pulses=None,
    max_gap=4000.0,
    search_depth=16,
    eps=1e-12,
    LARGE_NUMBER=1e4,
    return_info=False,
):
    """Compute a subliminal-accumulation loss for habituation hallmark 6.

    Hallmark 6 says that repeated stimulation can continue to accumulate after
    the response has reached an asymptotic level. Following the paper, this
    function continues the stimulation protocol after normal habituation until
    the relative peak decrement falls below a stricter habituation tolerance,
    then computes the recovery time in the usual way.

    The loss is

        recovery_time / continued_recovery_time,

    clipped to [0, 1].

    Args:
        iocrn: IOCRN-like object implementing `transient_response_piecewise`.
        A: Input amplitude during each ON segment.
        T: Duration of one complete pulse period.
        Ton: Duration of the ON part of each pulse.
        habituation_time: Number of pulse periods in the normal train.
        recovery_time: Recovery time computed with the normal tolerance.
        x0: Initial condition for each simulation.
        stricter_recovery_tolerance: Deprecated compatibility argument. It is
            accepted but not used by the paper-style Hallmark 6 loss.
        stricter_habituation_tolerance: Lower relative peak-decrement threshold
            used to continue stimulation beyond normal habituation.
        recovery_tolerance: Normal recovery tolerance used after the continued
            stimulation phase.
        n_dec: Number of peak-to-peak intervals in the max-min window used to
            detect the stricter continued habituation time. For the tube
            method, this is the minimum number of peak-to-peak intervals that
            must remain after the detected pulse.
        habituation_time_method: Either `"window"` for the historical local
            window criterion or `"tube"` for the suffix-tube criterion.
        max_continued_pulses: Maximum number of pulses allowed when searching
            for the stricter habituation point. Defaults to twice the normal
            habituation time when called directly; the task wrapper passes the
            main pulse-train length.
        max_gap: Largest OFF gap tested before giving up.
        search_depth: Number of binary-search refinements.
        eps: Small positive value used to avoid division by zero.
        LARGE_NUMBER: If any response reaches this magnitude, treat the
            trajectory as unstable.
        return_info: If True, return `(loss, info)`. Otherwise return only
            `loss`.

    Returns:
        Scalar loss in [0, 1], or `(loss, info)` when `return_info` is True.
    """
    def finish(loss, info):
        loss = float(np.clip(loss, 0.0, 1.0))
        return (loss, info) if return_info else loss

    if not np.isfinite(habituation_time) or not np.isfinite(recovery_time):
        return finish(1.0, {})

    if float(recovery_time) < 0.0:
        return finish(1.0, {})

    n_dec = int(n_dec)
    if n_dec < 1:
        return finish(1.0, {})
    habituation_time_method = str(habituation_time_method).lower()
    if habituation_time_method not in {"window", "tube"}:
        return finish(1.0, {})

    normal_habituation_time = int(habituation_time)
    if normal_habituation_time < 1:
        return finish(1.0, {})

    if max_continued_pulses is None:
        max_continued_pulses = max(normal_habituation_time + 1, 2 * normal_habituation_time)
    max_continued_pulses = int(max_continued_pulses)
    if max_continued_pulses <= normal_habituation_time:
        return finish(1.0, {})

    inputs, segments = rectangular_pulse_program(A, T, Ton, max_continued_pulses)
    time, trajectories, outputs, _ = iocrn.transient_response_piecewise(
        u_nested_list=inputs,
        x0_list=[x0],
        nested_time_horizon=segments,
        force=True,
    )
    response = outputs[0][0]
    input_signal = _sample_piecewise_input_signal(time, inputs[0], segments)
    info = {
        "recovery_time": recovery_time,
        "continued_recovery_time": np.inf,
        "stricter_recovery_time": np.inf,
        "stricter_habituation_time": np.inf,
        "continued_habituation_time": np.inf,
        "stricter_habituation_tolerance": stricter_habituation_tolerance,
        "legacy_stricter_recovery_tolerance": stricter_recovery_tolerance,
        "recovery_tolerance": recovery_tolerance,
        "n_dec": n_dec,
        "habituation_time_method": habituation_time_method,
        "max_continued_pulses": max_continued_pulses,
        "continued_train_time": time,
        "continued_train_input_signal": input_signal,
        "continued_train_response": response,
        "continued_train_trajectory": trajectories[0],
    }

    if (
        response.size == 0
        or not np.all(np.isfinite(response))
        or np.nanmax(np.abs(response)) >= float(LARGE_NUMBER)
    ):
        return finish(1.0, info)

    peaks = extract_period_peaks(time, response, T, max_continued_pulses)
    relative_decrements = np.divide(
        peaks[:-1] - peaks[1:],
        peaks[:-1],
        out=np.full(max(peaks.size - 1, 0), np.inf, dtype=float),
        where=np.abs(peaks[:-1]) > eps,
    )
    info["continued_peaks"] = peaks
    info["continued_relative_decrements"] = relative_decrements

    (
        window_decrements,
        window_maxima,
        window_minima,
        window_is_nonincreasing,
    ) = compute_peak_window_decrements(peaks, n_dec=n_dec, eps=eps)
    info["continued_window_relative_decrements"] = window_decrements
    info["continued_window_peak_maxima"] = window_maxima
    info["continued_window_peak_minima"] = window_minima
    info["continued_window_is_nonincreasing"] = window_is_nonincreasing
    tube_decrements, tube_maxima, tube_minima = compute_peak_tube_decrements(
        peaks,
        min_remaining_intervals=n_dec,
        eps=eps,
    )
    info["continued_tube_relative_decrements"] = tube_decrements
    info["continued_tube_peak_maxima"] = tube_maxima
    info["continued_tube_peak_minima"] = tube_minima

    if (
        peaks.size < 2
        or not np.all(np.isfinite(peaks))
        or np.any(np.abs(peaks) >= float(LARGE_NUMBER))
    ):
        return finish(1.0, info)

    if habituation_time_method == "tube":
        min_index = normal_habituation_time
        qualifying_indices = np.flatnonzero(
            (tube_decrements < float(stricter_habituation_tolerance))
            & (np.arange(tube_decrements.size) >= min_index)
        )
    else:
        min_index = max(normal_habituation_time - n_dec, 0)
        qualifying_indices = np.flatnonzero(
            window_is_nonincreasing
            & (window_decrements < float(stricter_habituation_tolerance))
            & (np.arange(window_decrements.size) >= min_index)
        )
    if len(qualifying_indices) == 0:
        return finish(1.0, info)

    if habituation_time_method == "tube":
        stricter_habituation_time = int(qualifying_indices[0])
    else:
        stricter_habituation_time = int(qualifying_indices[0] + n_dec)
    info["stricter_habituation_time"] = stricter_habituation_time
    info["continued_habituation_time"] = stricter_habituation_time

    continued_recovery_time, recovery_difference, recovery_peak, recovery_info = compute_recovery_time(
        iocrn,
        A,
        T,
        Ton,
        stricter_habituation_time,
        x0,
        recovery_tolerance=recovery_tolerance,
        max_gap=max_gap,
        search_depth=search_depth,
        return_info=True,
    )
    info.update({
        "continued_recovery_time": continued_recovery_time,
        "stricter_recovery_time": continued_recovery_time,
        "recovery_difference": recovery_difference,
        "recovery_peak": recovery_peak,
    })
    info.update(recovery_info)

    if not np.isfinite(continued_recovery_time):
        return finish(1.0, info)

    loss = float(recovery_time) / max(float(continued_recovery_time), eps)
    return finish(loss, info)


def _amplify_component_losses(component_losses, amplification_factors, invalid_loss):
    """Amplify finite component losses at or above one but not invalid."""
    amplification_factors = {} if amplification_factors is None else dict(amplification_factors)
    amplified = {}
    factors = {}
    for name, value in component_losses.items():
        value = float(value)
        factor = float(amplification_factors.get(name, 1.0))
        factors[name] = factor
        if np.isfinite(value) and 1.0 <= value < float(invalid_loss):
            amplified[name] = float(factor * value)
        else:
            amplified[name] = value
    return amplified, factors


def habituation_hallmarks_loss(
    iocrn,
    A,
    T,
    Ton,
    n_pulses,
    x0,
    weights=None,
    T_values=None,
    A_values=None,
    h1_kwargs=None,
    h2_kwargs=None,
    h3_kwargs=None,
    h4_kwargs=None,
    h5_kwargs=None,
    h6_kwargs=None,
    amplification_factors=None,
    invalid_loss=1e4,
    store_info=True,
    return_info=False,
):
    """Compute a weighted sum of the six habituation hallmark losses.

    This is the task-level wrapper for the individual hallmark losses. It calls
    each hallmark loss with `return_info=True`, combines the scalar losses with
    user-provided weights, and stores the per-hallmark info on
    `iocrn.last_task_info` for later rendering.

    Args:
        iocrn: IOCRN-like object implementing `transient_response_piecewise`.
        A: Reference input amplitude.
        T: Reference pulse period.
        Ton: Duration of the ON part of each pulse.
        n_pulses: Number of pulses used for reference, frequency, and intensity
            habituation simulations.
        x0: Initial condition for each simulation.
        weights: Optional dictionary with keys `"hallmark1"`, ...,
            `"hallmark6"`. Missing weights default to 1.
        T_values: Pulse periods used by hallmark 4. Defaults to
            `[T, 4*T/3, 5*T/3]`.
        A_values: Input amplitudes used by hallmark 5. Defaults to
            `[A, 2*A, 3*A]`.
        h1_kwargs: Optional keyword overrides for `hallmark1_loss`.
            If this contains `n_dec` or `habituation_time_method`, those values
            are also used by hallmarks 3, 4, 5, and 6 unless those
            hallmark-specific kwargs override them.
        h2_kwargs: Optional keyword overrides for `hallmark2_loss`.
        h3_kwargs: Optional keyword overrides for `hallmark3_loss`.
        h4_kwargs: Optional keyword overrides for `hallmark4_loss`.
        h5_kwargs: Optional keyword overrides for `hallmark5_loss`.
        h6_kwargs: Optional keyword overrides for `hallmark6_loss`.
        amplification_factors: Optional dictionary with keys `"hallmark1"`,
            ..., `"hallmark6"`. A finite component loss `L_i` is replaced by
            `amplification_factors[i] * L_i` only when
            `1 <= L_i < invalid_loss`.
            Missing factors default to 1.
        invalid_loss: Finite penalty used when a component loss is NaN or inf.
        store_info: If True, write the combined information to
            `iocrn.last_task_info`.
        return_info: If True, return `(loss, info)`. Otherwise return only
            `loss`.

    Returns:
        Weighted scalar loss, or `(loss, info)` when `return_info` is True.
    """
    weights = {} if weights is None else dict(weights)
    h1_kwargs = {} if h1_kwargs is None else dict(h1_kwargs)
    h2_kwargs = {} if h2_kwargs is None else dict(h2_kwargs)
    h3_kwargs = {} if h3_kwargs is None else dict(h3_kwargs)
    h4_kwargs = {} if h4_kwargs is None else dict(h4_kwargs)
    h5_kwargs = {} if h5_kwargs is None else dict(h5_kwargs)
    h6_kwargs = {} if h6_kwargs is None else dict(h6_kwargs)
    shared_n_dec = h1_kwargs.get("n_dec", 1)
    shared_habituation_time_method = h1_kwargs.get("habituation_time_method", "window")
    h3_kwargs.setdefault("n_dec", shared_n_dec)
    h4_kwargs.setdefault("n_dec", shared_n_dec)
    h5_kwargs.setdefault("n_dec", shared_n_dec)
    h6_kwargs.setdefault("n_dec", shared_n_dec)
    h3_kwargs.setdefault("habituation_time_method", shared_habituation_time_method)
    h4_kwargs.setdefault("habituation_time_method", shared_habituation_time_method)
    h5_kwargs.setdefault("habituation_time_method", shared_habituation_time_method)
    h6_kwargs.setdefault("habituation_time_method", shared_habituation_time_method)
    if T_values is None:
        T_values = [float(T), 4.0 * float(T) / 3.0, 5.0 * float(T) / 3.0]
    if A_values is None:
        A_values = [float(A), 2.0 * float(A), 3.0 * float(A)]

    h1_loss, h1_info = hallmark1_loss(
        iocrn, A, T, Ton, n_pulses, x0, return_info=True, **h1_kwargs
    )
    skip_after_hallmark1 = (
        not np.isfinite(h1_info.get("habituation_time", np.inf))
        or bool(h1_info.get("too_early_habituation", False))
    )
    if skip_after_hallmark1:
        raw_component_losses = {
            "hallmark1": h1_loss,
            "hallmark2": invalid_loss,
            "hallmark3": invalid_loss,
            "hallmark4": invalid_loss,
            "hallmark5": invalid_loss,
            "hallmark6": invalid_loss,
        }
        component_losses = {
            name: float(value) if np.isfinite(value) else float(invalid_loss)
            for name, value in raw_component_losses.items()
        }
        raw_finite_component_losses = dict(component_losses)
        component_losses, component_amplification_factors = _amplify_component_losses(
            component_losses,
            amplification_factors,
            invalid_loss,
        )
        component_weights = {
            name: float(weights.get(name, 1.0))
            for name in component_losses
        }
        total_loss = float(sum(
            component_weights[name] * component_losses[name]
            for name in component_losses
        ))
        hallmark_info = {
            "hallmark1": h1_info,
            "hallmark2": {"recovery_time": np.inf, "skipped": True},
            "hallmark3": {"skipped": True},
            "hallmark4": {"habituation_times": np.asarray([], dtype=float), "runs": [], "skipped": True},
            "hallmark5": {"habituation_times": np.asarray([], dtype=float), "runs": [], "skipped": True},
            "hallmark6": {
                "recovery_time": np.inf,
                "continued_recovery_time": np.inf,
                "stricter_recovery_time": np.inf,
                "stricter_habituation_time": np.inf,
                "skipped": True,
            },
        }
        info = {
            "component_losses": component_losses,
            "raw_component_losses": raw_finite_component_losses,
            "component_weights": component_weights,
            "component_amplification_factors": component_amplification_factors,
            "hallmark_info": hallmark_info,
            "A": float(A),
            "T": float(T),
            "Ton": float(Ton),
            "n_pulses": int(n_pulses),
            "T_values": np.asarray(T_values, dtype=float),
            "A_values": np.asarray(A_values, dtype=float),
            "skipped_after_hallmark1": True,
        }

        if store_info:
            if not hasattr(iocrn, "last_task_info") or iocrn.last_task_info is None:
                iocrn.last_task_info = {}
            iocrn.last_task_info["reward"] = total_loss
            iocrn.last_task_info["reward type"] = "habituation_hallmarks_paper"
            iocrn.last_task_info["type"] = "transient response"
            iocrn.last_task_info["hallmark_info"] = info
            iocrn.last_task_info["component_losses"] = component_losses
            iocrn.last_task_info["raw_component_losses"] = raw_finite_component_losses
            iocrn.last_task_info["component_weights"] = component_weights
            iocrn.last_task_info["component_amplification_factors"] = component_amplification_factors

        return (total_loss, info) if return_info else total_loss

    h2_loss, h2_info = hallmark2_loss(
        iocrn,
        A,
        T,
        Ton,
        h1_info.get("habituation_time", np.inf),
        x0,
        return_info=True,
        **h2_kwargs,
    )
    h3_loss, h3_info = hallmark3_loss(
        iocrn,
        A,
        T,
        Ton,
        h1_info.get("habituation_time", np.inf),
        h2_info.get("recovery_time", np.inf),
        x0,
        return_info=True,
        **h3_kwargs,
    )
    h4_loss, h4_info = hallmark4_loss(
        iocrn, A, T_values, Ton, n_pulses, x0, return_info=True, **h4_kwargs
    )
    h5_loss, h5_info = hallmark5_loss(
        iocrn, A_values, T, Ton, n_pulses, x0, return_info=True, **h5_kwargs
    )
    h6_call_kwargs = dict(h6_kwargs)
    h6_call_kwargs.setdefault("max_continued_pulses", n_pulses)
    h6_loss, h6_info = hallmark6_loss(
        iocrn,
        A,
        T,
        Ton,
        h1_info.get("habituation_time", np.inf),
        h2_info.get("recovery_time", np.inf),
        x0,
        return_info=True,
        **h6_call_kwargs,
    )

    raw_component_losses = {
        "hallmark1": h1_loss,
        "hallmark2": h2_loss,
        "hallmark3": h3_loss,
        "hallmark4": h4_loss,
        "hallmark5": h5_loss,
        "hallmark6": h6_loss,
    }
    component_losses = {
        name: float(value) if np.isfinite(value) else float(invalid_loss)
        for name, value in raw_component_losses.items()
    }
    raw_finite_component_losses = dict(component_losses)
    component_losses, component_amplification_factors = _amplify_component_losses(
        component_losses,
        amplification_factors,
        invalid_loss,
    )
    component_weights = {
        name: float(weights.get(name, 1.0))
        for name in component_losses
    }
    total_loss = float(sum(
        component_weights[name] * component_losses[name]
        for name in component_losses
    ))

    hallmark_info = {
        "hallmark1": h1_info,
        "hallmark2": h2_info,
        "hallmark3": h3_info,
        "hallmark4": h4_info,
        "hallmark5": h5_info,
        "hallmark6": h6_info,
    }
    info = {
        "component_losses": component_losses,
        "raw_component_losses": raw_finite_component_losses,
        "component_weights": component_weights,
        "component_amplification_factors": component_amplification_factors,
        "hallmark_info": hallmark_info,
        "A": float(A),
        "T": float(T),
        "Ton": float(Ton),
        "n_pulses": int(n_pulses),
        "T_values": np.asarray(T_values, dtype=float),
        "A_values": np.asarray(A_values, dtype=float),
    }

    if store_info:
        if not hasattr(iocrn, "last_task_info") or iocrn.last_task_info is None:
            iocrn.last_task_info = {}
        iocrn.last_task_info["reward"] = total_loss
        iocrn.last_task_info["reward type"] = "habituation_hallmarks_paper"
        iocrn.last_task_info["type"] = "transient response"
        iocrn.last_task_info["hallmark_info"] = info
        iocrn.last_task_info["component_losses"] = component_losses
        iocrn.last_task_info["raw_component_losses"] = raw_finite_component_losses
        iocrn.last_task_info["component_weights"] = component_weights
        iocrn.last_task_info["component_amplification_factors"] = component_amplification_factors

    return (total_loss, info) if return_info else total_loss


def _sample_piecewise_input_signal(time, inputs, segments):
    """Sample a one-dimensional piecewise-constant input on a stitched time grid."""
    input_signal = np.zeros_like(time, dtype=float)
    start = 0.0
    for input_value, segment in zip(inputs, segments):
        end = start + float(np.asarray(segment, dtype=float)[-1])
        mask = (time >= start) & (time <= end)
        input_signal[mask] = float(np.asarray(input_value, dtype=float).reshape(-1)[0])
        start = end
    return input_signal
