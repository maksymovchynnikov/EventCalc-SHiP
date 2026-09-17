# mergeResults.py
from contextlib import contextmanager
import hashlib
import json
import os
import stat
import tempfile

import numpy as np

from funcs import coupling_conventions, output_provenance
from funcs.fast_event_io import write_event_rows
from funcs.simulation_config import PROJECT_ROOT

try:
    import fcntl
except ImportError:  # pragma: no cover - EventCalc's supported hosts are POSIX.
    fcntl = None


EVENT_WRITE_CHUNK_SIZE = 10000
MAX_READABLE_CHANNEL_TAG_LENGTH = 96


def _coordinate_token(value):
    """Return a compact filename token without losing coordinate identity."""
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError("output coordinates must be finite")
    compact = f"{numeric:.3e}"
    if float(compact) == numeric:
        return compact
    return f"{numeric:.17e}"


def _positive_int_from_env(name, default):
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _mixing_string(MixingPatternArray):
    return '_'.join(_coordinate_token(mp) for mp in MixingPatternArray)


def _alp_mixed_label(xi, interference):
    if xi is None or interference not in {"constructive", "destructive"}:
        raise ValueError("ALP-mixed output requires xi and an interference sign")
    xi_text = format(float(xi), ".8g").replace("-", "m").replace(".", "p")
    return f"xi{xi_text}_{interference}"


def _generator_suffix(generator_tag):
    """Return a safe optional filename suffix for batch-run provenance.

    ``None`` deliberately preserves the historical filenames used by the
    interactive driver and by third-party callers of this module.
    """
    if generator_tag is None:
        return ""
    tag = str(generator_tag).strip()
    if not tag:
        return ""
    if any(not (char.isalnum() or char in "-_.") for char in tag):
        raise ValueError(
            "generator_tag may contain only letters, digits, '-', '_', and '.'"
        )
    return "_" + tag


def channel_selection_tag(decay_channels, selected_decay_indices):
    """Return a canonical, filename-safe identity for a channel selection.

    The identity is based on resolved decay-table rows, not on the spelling of
    the CLI tokens, so equivalent selections share a filename.  Short subsets
    remain human-readable.  Very long subsets use a collision-resistant digest
    and retain their cardinality; the event file's process headers remain the
    authoritative human-readable channel list.
    """
    channel_count = len(decay_channels)
    try:
        indices = sorted({int(index) for index in selected_decay_indices})
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "selected decay-channel indices must be integers") from exc
    if not indices:
        raise ValueError("at least one decay channel must be selected")
    if indices[0] < 0 or indices[-1] >= channel_count:
        raise ValueError("selected decay-channel index is out of range")
    if indices == list(range(channel_count)):
        return "channels-all"

    readable = "channels-i" + ".".join(str(index + 1) for index in indices)
    if len(readable) <= MAX_READABLE_CHANNEL_TAG_LENGTH:
        return readable

    canonical = ",".join(str(index) for index in indices).encode("ascii")
    digest = hashlib.sha256(canonical).hexdigest()[:16]
    return f"channels-n{len(indices)}-{digest}"


def _channel_suffix(channel_tag):
    if channel_tag is None:
        return ""
    tag = str(channel_tag).strip()
    if not tag:
        return ""
    if not tag.startswith("channels-"):
        raise ValueError("channel_tag must start with 'channels-'")
    if any(not (char.isalnum() or char in "-.") for char in tag):
        raise ValueError(
            "channel_tag may contain only letters, digits, '-', and '.'")
    return "_" + tag


def _physics_label(LLP_name, MixingPatternArray, uncertainty, alp_production_mode,
                   alp_mixing_xi, alp_interference):
    """The part of an output filename that names the physics configuration."""
    if LLP_name == "HNL" and MixingPatternArray is not None and isinstance(
            MixingPatternArray, np.ndarray):
        return _mixing_string(MixingPatternArray)
    if LLP_name == "Dark-photons" and uncertainty is not None:
        return str(uncertainty)
    if LLP_name == "ALP-photon" and alp_production_mode is not None:
        return str(alp_production_mode)
    if LLP_name == "ALP-mixed":
        return _alp_mixed_label(alp_mixing_xi, alp_interference)
    if "Scalar" in LLP_name and uncertainty is not None:
        return str(uncertainty)
    return None


def _event_output_path(eventData_dir, LLP_name, mass, c_tau,
                       MixingPatternArray, uncertainty, alp_production_mode,
                       generator_tag=None, channel_tag=None,
                       alp_mixing_xi=None, alp_interference=None):
    generator_suffix = _generator_suffix(generator_tag)
    channel_suffix = _channel_suffix(channel_tag)
    mass_token = _coordinate_token(mass)
    ctau_token = _coordinate_token(c_tau)
    label = _physics_label(LLP_name, MixingPatternArray, uncertainty,
                           alp_production_mode, alp_mixing_xi, alp_interference)
    stem = f'{LLP_name}_{mass_token}_{ctau_token}'
    if label is not None:
        stem = f'{stem}_{label}'
    return os.path.join(
        eventData_dir, f'{stem}{generator_suffix}{channel_suffix}_data.dat')


def _total_filename(LLP_name, MixingPatternArray, uncertainty,
                    alp_production_mode, generator_tag=None, channel_tag=None,
                    alp_mixing_xi=None, alp_interference=None):
    generator_suffix = _generator_suffix(generator_tag)
    channel_suffix = _channel_suffix(channel_tag)
    label = _physics_label(LLP_name, MixingPatternArray, uncertainty,
                           alp_production_mode, alp_mixing_xi, alp_interference)
    stem = LLP_name if label is None else f"{LLP_name}_{label}"
    return f"{stem}{generator_suffix}{channel_suffix}_total.txt"


def _output_directories(output_root, LLP_name):
    """Create and return the event and totals directories of one run."""
    base = os.path.join(
        str(PROJECT_ROOT if output_root is None else output_root),
        'outputs', LLP_name)
    eventData_dir = os.path.join(base, 'eventData')
    total_dir = os.path.join(base, 'total')
    os.makedirs(eventData_dir, exist_ok=True)
    os.makedirs(total_dir, exist_ok=True)
    return eventData_dir, total_dir


@contextmanager
def _exclusive_path_lock(target_path):
    """Serialize writers that target the same final output path."""
    directory = os.path.dirname(target_path) or "."
    lock_path = os.path.join(
        directory, "." + os.path.basename(target_path) + ".lock")
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _atomic_text_output(target_path):
    """Write a same-directory temporary file and atomically publish it."""
    directory = os.path.dirname(target_path) or "."
    basename = os.path.basename(target_path)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=directory, prefix=f".{basename}.", suffix=".tmp", text=True)
    stream = os.fdopen(descriptor, "w", encoding="utf-8")
    try:
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        try:
            published_mode = stat.S_IMODE(os.stat(target_path).st_mode)
        except FileNotFoundError:
            published_mode = 0o644
        os.chmod(temporary_path, published_mode)
        os.replace(temporary_path, target_path)
    except BaseException:
        if not stream.closed:
            stream.close()
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_lines(target_path, lines):
    with _atomic_text_output(target_path) as stream:
        stream.writelines(lines)


def _write_hadronization_sidecar(output_path, hadronization_metadata):
    """Record the release version and mother PDG code beside a matched sample."""
    if not hadronization_metadata or hadronization_metadata.get('backend') != 'exhad':
        return None
    sidecar_path = output_path + '.hadronization.json'
    _atomic_write_lines(
        sidecar_path,
        [json.dumps(hadronization_metadata, indent=2, sort_keys=True), "\n"])
    return sidecar_path


def _append_total_row(
        total_dir, total_filename, data_values, replace_coordinate=False):
    total_file_path = os.path.join(total_dir, total_filename)
    header = (
        'mass coupling_squared c_tau N_LLP_tot epsilon_polar '
        'epsilon_azimuthal P_decay_averaged Br_visible N_ev_tot\n'
    )
    data_string = ' '.join("{:.17e}".format(x) for x in data_values) + "\n"
    coordinate = (
        _coordinate_token(data_values[0]),
        _coordinate_token(data_values[2]),
    )

    # Tagged outputs are current snapshots: their per-coordinate event file is
    # replaced on a deterministic rerun, so the corresponding summary row is
    # replaced as well.  The lock covers the complete read/modify/publish
    # transaction and os.replace prevents partial summaries after interruption.
    with _exclusive_path_lock(total_file_path):
        try:
            with open(total_file_path, encoding="utf-8") as total_file:
                existing_lines = total_file.readlines()
        except FileNotFoundError:
            existing_lines = []

        if not existing_lines:
            _atomic_write_lines(total_file_path, [header, data_string])
            return True

        if not replace_coordinate:
            if data_string in existing_lines[1:]:
                return False
            _atomic_write_lines(
                total_file_path, existing_lines + [data_string])
            return True

        matching_indices = [
            index for index, line in enumerate(existing_lines[1:], start=1)
            if len(line.split()) >= 3
            and (
                _coordinate_token(float(line.split()[0])),
                _coordinate_token(float(line.split()[2])),
            ) == coordinate
        ]
        if (len(matching_indices) == 1
                and existing_lines[matching_indices[0]] == data_string):
            return False

        if not matching_indices:
            _atomic_write_lines(
                total_file_path, existing_lines + [data_string])
            return True

        first_match = matching_indices[0]
        matching_set = set(matching_indices)
        rewritten = [
            line for index, line in enumerate(existing_lines)
            if index not in matching_set or index == first_match
        ]
        rewritten[first_match] = data_string
        _atomic_write_lines(total_file_path, rewritten)
    return True


def _write_event_rows(file_obj, mother_rows, decay_rows, chunk_size):
    if len(mother_rows) != len(decay_rows):
        raise ValueError("Mother-particle and decay-product rows have different lengths.")

    for start in range(0, len(mother_rows), chunk_size):
        end = min(start + chunk_size, len(mother_rows))
        rows = np.hstack((mother_rows[start:end], decay_rows[start:end]))
        write_event_rows(file_obj, rows, "%.18e", chunk_size=chunk_size)


def _output_process_blocks(decay_channels, selected_decay_indices,
                           size_per_channel, process_labels=None):
    """Return ordered output blocks with adjacent equal labels coalesced.

    ``process_labels`` contains optional per-selected-row overrides produced
    by the hadronic backend.  A ``None`` entry retains the corresponding
    physical decay-channel name.  Coalescing changes headers only: row order
    and the allocation of every generated event remain unchanged.
    """
    n_selected = len(selected_decay_indices)
    if len(size_per_channel) != n_selected:
        raise ValueError(
            "size_per_channel must have one entry per selected decay channel")
    if process_labels is None:
        process_labels = [None] * n_selected
    elif len(process_labels) != n_selected:
        raise ValueError(
            "process_labels must have one entry per selected decay channel")

    blocks = []
    start_row = 0
    for position, table_index in enumerate(selected_decay_indices):
        channel_size = int(size_per_channel[position])
        if channel_size < 0:
            raise ValueError("channel event counts must be non-negative")
        label = process_labels[position]
        if label is None:
            label = decay_channels[table_index]
        label = str(label)
        if channel_size:
            if blocks and blocks[-1][0] == label:
                previous_label, previous_start, previous_size = blocks[-1]
                blocks[-1] = (
                    previous_label, previous_start,
                    previous_size + channel_size)
            else:
                blocks.append((label, start_row, channel_size))
        start_row += channel_size
    return blocks


def save(
    motherParticleResults,
    decayProductsResults,
    LLP_name,
    mass,
    MixingPatternArray,
    c_tau,
    decayChannels,
    size_per_channel,
    finalEvents,
    epsilon_polar,
    epsilon_azimuthal,
    N_LLP_tot,
    coupling_squared,
    P_decay_averaged,
    N_ev_tot,
    br_visible_val,
    selected_decay_indices,
    uncertainty,
    ifExportEvents,
    alp_production_mode=None,
    *,
    generator_tag=None,
    channel_tag=None,
    process_labels=None,
    coupling_metadata=None,
    generator_provenance=None,
    alp_mixing_xi=None,
    alp_interference=None,
    hadronization_metadata=None,
    output_root=None,
):
    """
    Saves simulation results to data files.

    NOTE: The redundant check for N_ev_tot >= 2 is removed.
    The threshold is fully handled in simulate.py.
    """
    if len(motherParticleResults) != len(decayProductsResults):
        raise ValueError("Mother-particle and decay-product result arrays must have the same number of rows.")
    if sum(int(value) for value in size_per_channel) != len(motherParticleResults):
        raise ValueError(
            "Per-channel event counts must sum to the number of output rows.")

    eventData_dir, total_dir = _output_directories(output_root, LLP_name)
    outputfileName = _event_output_path(
        eventData_dir, LLP_name, mass, c_tau,
        MixingPatternArray, uncertainty, alp_production_mode, generator_tag,
        channel_tag, alp_mixing_xi, alp_interference
    )

    # Since we already checked N_ev_tot < min_events_threshold in simulate.py,
    # here we only check ifExportEvents.
    if ifExportEvents:
        write_chunk_size = _positive_int_from_env("EVENT_WRITE_CHUNK_SIZE", EVENT_WRITE_CHUNK_SIZE)
        coupling_annotation = coupling_conventions.event_header_annotation(
            LLP_name, coupling_metadata)
        mixture_annotation = (
            " This is |g_agammagamma,total|^2. "
            f"Operator fraction xi: {float(alp_mixing_xi):.8g}. "
            f"Interference: {alp_interference}."
            if LLP_name == "ALP-mixed"
            else ""
        )
        with _exclusive_path_lock(outputfileName):
            with _atomic_text_output(outputfileName) as f:
                header = (
                    f"Sampled {finalEvents:.6e} events inside SHiP volume. "
                    f"Squared coupling: {coupling_squared:.6e}. "
                    f"Total number of produced LLPs: {N_LLP_tot:.6e}. "
                    f"Polar acceptance: {epsilon_polar:.6e}. "
                    f"Azimuthal acceptance: {epsilon_azimuthal:.6e}. "
                    f"Averaged decay probability: {P_decay_averaged:.6e}. "
                    f"Visible Br Ratio: {br_visible_val:.6e}. "
                    f"Total number of events: {N_ev_tot:.6e} "
                    f"Channel selection: {channel_tag or 'legacy-unspecified'}"
                    f"{mixture_annotation}{coupling_annotation}\n\n"
                )
                f.write(header)

                process_blocks = _output_process_blocks(
                    decayChannels, selected_decay_indices, size_per_channel,
                    process_labels=process_labels)
                for channel, start_row, channel_size in process_blocks:
                    end_row = start_row + channel_size
                    channel_header = (
                        f"#<process={channel}; "
                        f"sample_points={channel_size}>\n\n")
                    f.write(channel_header)

                    _write_event_rows(
                        f,
                        motherParticleResults[start_row:end_row],
                        decayProductsResults[start_row:end_row],
                        write_chunk_size
                    )
                    f.write("\n\n")
        coupling_conventions.write_output_sidecar(
            outputfileName, LLP_name, coupling_metadata)
        output_provenance.write_generator_sidecar(
            outputfileName, generator_provenance)
        _write_hadronization_sidecar(outputfileName, hadronization_metadata)
    else:
        print("Exporting events is disabled. The events file has not been recorded.")

    data_values = [
        mass,
        coupling_squared,
        c_tau,
        N_LLP_tot,
        epsilon_polar,
        epsilon_azimuthal,
        P_decay_averaged,
        br_visible_val,
        N_ev_tot
    ]
    total_filename = _total_filename(
        LLP_name, MixingPatternArray, uncertainty, alp_production_mode,
        generator_tag, channel_tag, alp_mixing_xi, alp_interference
    )
    _append_total_row(
        total_dir, total_filename, data_values,
        replace_coordinate=generator_tag is not None)
    coupling_conventions.write_output_sidecar(
        os.path.join(total_dir, total_filename), LLP_name, coupling_metadata)
    output_provenance.write_generator_sidecar(
        os.path.join(total_dir, total_filename), generator_provenance)


def save_total_only(
    LLP_name,
    mass,
    coupling_squared,
    c_tau,
    N_LLP_tot,
    epsilon_polar,
    epsilon_azimuthal,
    P_decay_averaged,
    br_visible_val,
    N_ev_tot,
    uncertainty,
    MixingPatternArray,
    decayChannels,
    alp_production_mode=None,
    *,
    generator_tag=None,
    channel_tag=None,
    coupling_metadata=None,
    generator_provenance=None,
    alp_mixing_xi=None,
    alp_interference=None,
    output_root=None,
):
    _, total_dir = _output_directories(output_root, LLP_name)
    data_values = [
        mass,
        coupling_squared,
        c_tau,
        N_LLP_tot,
        epsilon_polar,
        epsilon_azimuthal,
        P_decay_averaged,
        br_visible_val,
        N_ev_tot
    ]
    total_filename = _total_filename(
        LLP_name, MixingPatternArray, uncertainty, alp_production_mode,
        generator_tag, channel_tag, alp_mixing_xi, alp_interference
    )
    _append_total_row(
        total_dir, total_filename, data_values,
        replace_coordinate=generator_tag is not None)
    coupling_conventions.write_output_sidecar(
        os.path.join(total_dir, total_filename), LLP_name, coupling_metadata)
    output_provenance.write_generator_sidecar(
        os.path.join(total_dir, total_filename), generator_provenance)
