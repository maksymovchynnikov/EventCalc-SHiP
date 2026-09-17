# selecting_processing.py
import os
import numpy as np
import re
import sys
from dataclasses import dataclass, replace


_UNCERTAINTIES = frozenset(("lower", "central", "upper"))
_CHANNEL_TAG_RE = re.compile(
    r"^channels-(?:"
    r"(?P<all>all)|"
    r"i(?P<indices>\d+(?:\.\d+)*)|"
    r"n(?P<count>\d+)-(?P<digest>[0-9a-fA-F]{16})"
    r")$"
)
_GENERATOR_TAG_RE = re.compile(
    r"^(?P<kind>"
    r"eventcalc-direct|"
    r"raw-pythia|"
    r"stock-pythia(?:-[A-Za-z0-9.-]+)?|"
    r"exhad-[A-Za-z0-9.-]+"
    r")-seed(?P<seed>\d+)"
    r"(?:-src(?P<source_digest>[0-9a-fA-F]+))?$"
)
_PORTABLE_EXHAD_ID_RE = re.compile(
    r"^(?P<backend>[A-Za-z0-9.-]+)"
    r"-m\.(?P<model>[a-z][a-z0-9-]*)"
    r"-v\.(?P<variant>[a-z][a-z0-9-]*)"
    r"-u\.(?P<variation>[a-z][a-z0-9-]*)"
    r"-c\.(?P<contract_digest>[0-9a-fA-F]{12})$"
)
_SAFE_PHYSICS_LABEL_RE = re.compile(r"^[A-Za-z0-9.+-]+$")


@dataclass(frozen=True)
class OutputFileMetadata:
    """Structured identity parsed from an EventCalc output filename.

    ``physics_label`` is the exact label present in the filename.  Canonical
    fields such as ``production_mode`` are supplied in addition, so aliases
    remain distinguishable without forcing every consumer to understand them.
    """

    path: str
    filename: str
    kind: str
    llp_name: str
    mass: float | None = None
    c_tau: float | None = None
    physics_label: str | None = None
    xi: float | None = None
    interference: str | None = None
    mixing_pattern: tuple | None = None
    production_mode: str | None = None
    uncertainty: str | None = None
    generator_tag: str | None = None
    generator: str | None = None
    backend: str | None = None
    seed: int | None = None
    source_digest: str | None = None
    hadronization_model: str | None = None
    hadronization_variant: str | None = None
    hadronization_variation: str | None = None
    contract_digest: str | None = None
    channel_tag: str | None = None
    channel_indices: tuple = ()
    channel_count: int | None = None
    channel_digest: str | None = None
    extra_tags: tuple = ()

    @property
    def is_legacy(self):
        return (
            self.generator_tag is None
            and self.channel_tag is None
            and not self.extra_tags
        )

    @property
    def selector(self):
        """Return the historical physics selector where one exists."""
        if self.llp_name == "HNL":
            return self.mixing_pattern
        if self.llp_name == "ALP-mixed":
            return (self.xi, self.interference)
        if (
            self.llp_name in ("Dark-photons", "ALP-photon")
            or "Scalar" in self.llp_name
        ):
            return self.physics_label
        return None

    def display_identifier(self):
        """Return a concise, lossless label for an interactive file choice."""
        fields = []
        if self.llp_name == "HNL" and self.mixing_pattern is not None:
            fields.append(
                "mixing=[%s]"
                % ", ".join(f"{value:.3f}" for value in self.mixing_pattern)
            )
        elif self.llp_name == "Dark-photons":
            if self.production_mode not in (None, "primary"):
                fields.append(f"production={self.production_mode}")
            if self.uncertainty is not None:
                fields.append(f"uncertainty={self.uncertainty}")
        elif self.llp_name == "ALP-photon":
            fields.append(f"production={self.production_mode}")
            if self.physics_label != self.production_mode:
                fields.append(f"label={self.physics_label}")
        elif self.llp_name == "ALP-mixed" and self.xi is not None:
            fields.append(f"xi={self.xi:g}, {self.interference}")
        elif "Scalar" in self.llp_name and self.physics_label is not None:
            fields.append(f"prescription={self.physics_label}")

        if self.generator_tag is not None:
            fields.append(f"generator={self.generator_tag}")
        if self.channel_tag is not None:
            fields.append(f"channels={self.channel_tag[len('channels-'):]}")
        fields.extend(f"tag={tag}" for tag in self.extra_tags)
        return "; ".join(fields) if fields else "legacy/default"


def _parse_generator_tag(token):
    match = _GENERATOR_TAG_RE.fullmatch(token)
    if match is None:
        return None

    kind = match.group("kind")
    if kind == "eventcalc-direct":
        generator, backend = "eventcalc-direct", "direct"
    elif kind == "raw-pythia":
        generator, backend = "raw-pythia", "pythia"
    elif kind.startswith("stock-pythia"):
        generator = "stock-pythia"
        backend = kind[len("stock-pythia-"):] if kind != "stock-pythia" else "pythia"
    else:
        generator = "exhad"
        backend = kind[len("exhad-"):]
    model = variant = variation = contract_digest = None
    if generator == "exhad":
        portable = _PORTABLE_EXHAD_ID_RE.fullmatch(backend)
        if portable is not None:
            backend = portable.group("backend")
            model = portable.group("model")
            variant = portable.group("variant")
            variation = portable.group("variation")
            contract_digest = portable.group("contract_digest").lower()
    return {
        "generator_tag": token,
        "generator": generator,
        "backend": backend,
        "seed": int(match.group("seed")),
        "source_digest": match.group("source_digest"),
        "hadronization_model": model,
        "hadronization_variant": variant,
        "hadronization_variation": variation,
        "contract_digest": contract_digest,
    }


def _parse_channel_tag(token):
    match = _CHANNEL_TAG_RE.fullmatch(token)
    if match is None:
        return None

    indices = ()
    if match.group("indices") is not None:
        indices = tuple(
            int(value) for value in match.group("indices").split(".")
        )
    return {
        "channel_tag": token,
        "channel_indices": indices,
        "channel_count": (
            int(match.group("count"))
            if match.group("count") is not None
            else None
        ),
        "channel_digest": match.group("digest"),
    }


def _is_metadata_token(token):
    return (
        _parse_generator_tag(token) is not None
        or _parse_channel_tag(token) is not None
    )


def _parse_qualifiers(tokens):
    metadata = {
        "generator_tag": None,
        "generator": None,
        "backend": None,
        "seed": None,
        "source_digest": None,
        "channel_tag": None,
        "channel_indices": (),
        "channel_count": None,
        "channel_digest": None,
        "extra_tags": (),
    }
    extras = []
    for token in tokens:
        generator = _parse_generator_tag(token)
        if generator is not None:
            if metadata["generator_tag"] is not None:
                raise ValueError("filename contains multiple generator tags")
            metadata.update(generator)
            continue

        channels = _parse_channel_tag(token)
        if channels is not None:
            if metadata["channel_tag"] is not None:
                raise ValueError("filename contains multiple channel tags")
            metadata.update(channels)
            continue

        if token.startswith("channels-"):
            raise ValueError(f"invalid channel tag '{token}'")
        if token.startswith((
                "eventcalc-direct", "raw-pythia",
                "stock-pythia", "exhad-")):
            raise ValueError(f"invalid generator tag '{token}'")
        extras.append(token)
    metadata["extra_tags"] = tuple(extras)
    return metadata


def _parse_dp_label(label):
    if label in _UNCERTAINTIES:
        return "primary", label
    try:
        production_mode, uncertainty = label.rsplit("-", 1)
    except ValueError as exc:
        raise ValueError(
            f"invalid dark-photon physics label '{label}'"
        ) from exc
    if (
        not production_mode
        or uncertainty not in _UNCERTAINTIES
        or _SAFE_PHYSICS_LABEL_RE.fullmatch(production_mode) is None
    ):
        raise ValueError(f"invalid dark-photon physics label '{label}'")
    return production_mode, uncertainty


def _parse_alp_mixed_label(label_tokens):
    """Read the operator fraction and interference sign of an ALP-mixed file.

    The writer encodes them as two tokens, ``xi<value>`` with the decimal point
    written ``p`` and a leading minus written ``m``, followed by the sign word.
    """
    if len(label_tokens) < 2 or not label_tokens[0].startswith("xi"):
        raise ValueError("ALP-mixed filename lacks an xi/interference label")
    interference = label_tokens[1]
    if interference not in {"constructive", "destructive"}:
        raise ValueError(
            f"invalid ALP-mixed interference label '{interference}'")
    try:
        xi = float(label_tokens[0][2:].replace("m", "-").replace("p", "."))
    except ValueError as exc:
        raise ValueError(
            f"invalid ALP-mixed operator fraction '{label_tokens[0]}'") from exc
    return xi, interference


def _canonical_alp_mode(label):
    aliases = {
        "primary": "primary",
        "cascade": "cascade",
        "cascades": "cascade",
        "combined": "combined",
        "primary+cascade": "combined",
        "cascade+primary": "combined",
    }
    try:
        return aliases[label]
    except KeyError as exc:
        raise ValueError(
            f"invalid ALP-photon production mode '{label}'"
        ) from exc


def _parse_output_filename(path, kind, llp_name=None):
    filename = os.path.basename(os.fspath(path))
    suffix = "_data.dat" if kind == "event" else "_total.txt"
    if not filename.endswith(suffix):
        raise ValueError(f"not an EventCalc {kind} filename: {filename}")

    stem = filename[:-len(suffix)]
    tokens = stem.split("_")
    minimum = 3 if kind == "event" else 1
    if len(tokens) < minimum:
        raise ValueError(f"filename has too few fields: {filename}")

    encoded_llp = tokens[0]
    if llp_name is not None and encoded_llp != llp_name:
        raise ValueError(
            f"filename LLP '{encoded_llp}' does not match '{llp_name}'"
        )

    mass = c_tau = None
    offset = 1
    if kind == "event":
        try:
            mass = float(tokens[1])
            c_tau = float(tokens[2])
        except ValueError as exc:
            raise ValueError(
                f"invalid mass or lifetime in filename: {filename}"
            ) from exc
        offset = 3

    remaining = list(tokens[offset:])
    mixing_pattern = None
    physics_label = None
    production_mode = None
    uncertainty = None
    xi = None
    interference = None

    if encoded_llp == "HNL":
        if remaining and not _is_metadata_token(remaining[0]):
            if len(remaining) < 3:
                raise ValueError(
                    f"HNL filename has incomplete mixing pattern: {filename}"
                )
            try:
                mixing_pattern = tuple(float(value) for value in remaining[:3])
            except ValueError as exc:
                raise ValueError(
                    f"invalid HNL mixing pattern in filename: {filename}"
                ) from exc
            remaining = remaining[3:]
    elif encoded_llp == "Dark-photons":
        if not remaining or _is_metadata_token(remaining[0]):
            raise ValueError(
                f"dark-photon filename lacks a physics label: {filename}"
            )
        physics_label = remaining.pop(0)
        production_mode, uncertainty = _parse_dp_label(physics_label)
    elif encoded_llp == "ALP-photon":
        if not remaining or _is_metadata_token(remaining[0]):
            raise ValueError(
                f"ALP-photon filename lacks a production mode: {filename}"
            )
        physics_label = remaining.pop(0)
        production_mode = _canonical_alp_mode(physics_label)
    elif encoded_llp == "ALP-mixed":
        if not remaining or _is_metadata_token(remaining[0]):
            raise ValueError(
                f"ALP-mixed filename lacks an xi/interference label: {filename}"
            )
        xi, interference = _parse_alp_mixed_label(remaining[:2])
        physics_label = "_".join(remaining[:2])
        remaining = remaining[2:]
    elif "Scalar" in encoded_llp:
        if remaining and not _is_metadata_token(remaining[0]):
            physics_label = remaining.pop(0)

    metadata = _parse_qualifiers(remaining)
    return OutputFileMetadata(
        path=os.fspath(path),
        filename=filename,
        kind=kind,
        llp_name=encoded_llp,
        mass=mass,
        c_tau=c_tau,
        physics_label=physics_label,
        xi=xi,
        interference=interference,
        mixing_pattern=mixing_pattern,
        production_mode=production_mode,
        uncertainty=uncertainty,
        **metadata,
    )


def parse_event_filename(path, llp_name=None):
    """Parse one legacy or tagged ``*_data.dat`` filename."""
    return _parse_output_filename(path, "event", llp_name=llp_name)


def parse_total_filename(path, llp_name=None):
    """Parse one legacy or tagged ``*_total.txt`` filename."""
    return _parse_output_filename(path, "total", llp_name=llp_name)


def discover_output_files(directory, kind, llp_name=None):
    """Discover and parse EventCalc outputs in a directory tree."""
    suffix = "_data.dat" if kind == "event" else "_total.txt"
    parser = parse_event_filename if kind == "event" else parse_total_filename
    records = []
    if not os.path.isdir(directory):
        return records
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        for filename in sorted(files):
            if not filename.endswith(suffix):
                continue
            path = os.path.join(root, filename)
            try:
                records.append(parser(path, llp_name=llp_name))
            except ValueError:
                continue
    return records


def _physics_label_matches(record, requested):
    if requested is None:
        return True
    if record.llp_name == "Dark-photons":
        try:
            production_mode, uncertainty = _parse_dp_label(requested)
        except ValueError:
            return False
        return (
            record.production_mode == production_mode
            and record.uncertainty == uncertainty
        )
    if record.llp_name == "ALP-photon":
        try:
            return record.production_mode == _canonical_alp_mode(requested)
        except ValueError:
            return False
    return record.physics_label == requested


def filter_output_files(
        records, *, physics_label=None, mass=None, c_tau=None,
        generator_tag=None, channel_tag=None):
    """Filter parsed records without discarding provenance dimensions."""
    if channel_tag is not None and not str(channel_tag).startswith("channels-"):
        channel_tag = f"channels-{channel_tag}"

    selected = []
    for record in records:
        if not _physics_label_matches(record, physics_label):
            continue
        if mass is not None and (
            record.mass is None
            or not np.isclose(record.mass, mass, rtol=1e-12, atol=1e-15)
        ):
            continue
        if c_tau is not None and (
            record.c_tau is None
            or not np.isclose(record.c_tau, c_tau, rtol=1e-12, atol=1e-15)
        ):
            continue
        if generator_tag is not None and record.generator_tag != generator_tag:
            continue
        if channel_tag is not None and record.channel_tag != channel_tag:
            continue
        selected.append(record)
    return selected


def require_unique_output(records, description="EventCalc output"):
    """Return one record or raise a diagnostic error instead of guessing."""
    records = list(records)
    if not records:
        raise FileNotFoundError(f"no matching {description}")
    if len(records) != 1:
        choices = ", ".join(sorted(record.filename for record in records))
        raise RuntimeError(
            f"multiple {description} files match; select generator/channel "
            f"metadata explicitly. Candidates: {choices}"
        )
    return records[0]

def parse_filenames(directory):
    """
    Parse legacy and tagged event filenames below ``directory``.

    The historical nested dictionary and relative filepath values are retained.
    Legacy files retain their historical selector keys. Tagged files use their
    immutable :class:`OutputFileMetadata` as the key, so generator, backend,
    seed, source, and channel variants at one coordinate cannot overwrite one
    another in the index.
    """
    llp_dict = {}
    for record in discover_output_files(directory, "event"):
        filepath = os.path.relpath(record.path, directory)
        record = replace(record, path=filepath)
        mass_lifetime = (record.mass, record.c_tau)
        sub_dict = llp_dict.setdefault(
            record.llp_name, {}
        ).setdefault(mass_lifetime, {})

        key = record.selector if record.is_legacy else record
        if key in sub_dict and sub_dict[key] != filepath:
            # Preserve the historical key for its first path, and retain any
            # duplicate physical file under its full metadata identity.
            key = record
        sub_dict[key] = filepath
    return llp_dict


def _option_label(option, llp_name):
    if isinstance(option, OutputFileMetadata):
        return option.display_identifier()
    if llp_name == "HNL" and option is not None:
        return "mixing=[%s]" % ", ".join(
            f"{value:.3f}" for value in option
        )
    if llp_name == "Dark-photons":
        try:
            production, uncertainty = _parse_dp_label(option)
        except (TypeError, ValueError):
            return str(option)
        fields = []
        if production != "primary":
            fields.append(f"production={production}")
        fields.append(f"uncertainty={uncertainty}")
        return "; ".join(fields)
    if llp_name == "ALP-photon":
        return f"production={option}"
    if llp_name == "ALP-mixed" and isinstance(option, tuple) and len(option) == 2:
        return f"xi={option[0]:g}, {option[1]}"
    if "Scalar" in llp_name and option is not None:
        return f"prescription={option}"
    return "legacy/default" if option is None else str(option)


def _option_sort_key(option, llp_name):
    return _option_label(option, llp_name), repr(option)

def user_selection(llp_dict):
    """
    Allows the user to select an LLP, mass-lifetime combination, and mixing patterns, uncertainty choices, or ALP-photon production modes.
    Returns the selected filepath, the selected LLP name, mass, lifetime, and the selected extra identifier.
    """
    print("Available LLPs:")
    llp_names_list = sorted(llp_dict.keys())
    for i, llp_name in enumerate(llp_names_list):
        print(f"{i+1}. {llp_name}")

    # Ask user to choose an LLP
    while True:
        try:
            choice = int(input("Choose an LLP by typing the number: "))
            if 1 <= choice <= len(llp_names_list):
                break
            else:
                print(f"Please enter a number between 1 and {len(llp_names_list)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    selected_llp = llp_names_list[choice - 1]
    print(f"Selected LLP: {selected_llp}")

    # Get available mass-lifetime combinations
    mass_lifetime_list = sorted(llp_dict[selected_llp].keys())
    print(f"Available mass-lifetime combinations for {selected_llp}:")
    for i, (mass, lifetime) in enumerate(mass_lifetime_list):
        print(f"{i+1}. mass={mass:.2e} GeV, lifetime={lifetime:.2e} s")

    # Ask user to choose a mass-lifetime
    while True:
        try:
            mass_lifetime_choice = int(input("Choose a mass-lifetime combination by typing the number: "))
            if 1 <= mass_lifetime_choice <= len(mass_lifetime_list):
                break
            else:
                print(f"Please enter a number between 1 and {len(mass_lifetime_list)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    selected_mass_lifetime = mass_lifetime_list[mass_lifetime_choice - 1]
    selected_mass, selected_lifetime = selected_mass_lifetime
    print(f"Selected mass: {selected_mass:.2e} GeV, lifetime: {selected_lifetime:.2e} s")

    # Select the exact physics/provenance variant. Legacy keys remain valid;
    # tagged keys carry OutputFileMetadata and therefore cannot collapse.
    sub_dict = llp_dict[selected_llp][selected_mass_lifetime]
    options_list = sorted(
        sub_dict.keys(), key=lambda option: _option_sort_key(option, selected_llp)
    )
    if not options_list:
        raise ValueError("No output variants are available for this coordinate.")

    if len(options_list) == 1:
        selected_variant = options_list[0]
    else:
        print(
            f"Available output variants for {selected_llp} with mass "
            f"{selected_mass:.2e} GeV and c*tau {selected_lifetime:.2e} m:"
        )
        for index, option in enumerate(options_list, start=1):
            print(f"{index}. {_option_label(option, selected_llp)}")
        while True:
            try:
                variant_choice = int(
                    input("Choose an output variant by typing the number: ")
                )
                if 1 <= variant_choice <= len(options_list):
                    break
                print(
                    f"Please enter a number between 1 and {len(options_list)}."
                )
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        selected_variant = options_list[variant_choice - 1]

    # Find the file matching the selection
    selected_filepath = sub_dict[selected_variant]
    print(f"Selected variant: {_option_label(selected_variant, selected_llp)}")
    print(f"Selected file: {selected_filepath}")

    return (
        selected_filepath,
        selected_llp,
        selected_mass,
        selected_lifetime,
        selected_variant,
    )

def read_file(filepath):
    """
    Reads the file at the given filepath.
    Returns finalEvents, epsilon_polar, epsilon_azimuthal, br_visible_val, coupling_squared, channels
    """
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        # Updated pattern to include Squared coupling
        pattern = (
            r'Sampled\s+(?P<finalEvents>[\d\.\+\-eE]+)\s+events inside SHiP volume\. '
            r'Squared coupling:\s+(?P<coupling_squared>[\d\.\+\-eE]+)\. '
            r'Total number of produced LLPs:\s+(?P<N_LLP_tot>[\d\.\+\-eE]+)\. '
            r'Polar acceptance:\s+(?P<epsilon_polar>[\d\.\+\-eE]+)\. '
            r'Azimuthal acceptance:\s+(?P<epsilon_azimuthal>[\d\.\+\-eE]+)\. '
            r'Averaged decay probability:\s+(?P<P_decay_averaged>[\d\.\+\-eE]+)\. '
            r'Visible Br Ratio:\s+(?P<br_visible_val>[\d\.\+\-eE]+)\. '
            r'Total number of events:\s+(?P<N_ev_tot>[\d\.\+\-eE]+)'
        )
        match = re.match(pattern, first_line)
        if match:
            finalEvents = float(match.group('finalEvents'))
            coupling_squared = float(match.group('coupling_squared'))
            N_LLP_tot = float(match.group('N_LLP_tot'))
            epsilon_polar = float(match.group('epsilon_polar'))
            epsilon_azimuthal = float(match.group('epsilon_azimuthal'))
            P_decay_averaged = float(match.group('P_decay_averaged'))
            br_visible_val = float(match.group('br_visible_val'))
            N_ev_tot = float(match.group('N_ev_tot'))
            print(f"finalEvents: {finalEvents}, coupling_squared: {coupling_squared}, N_LLP_tot: {N_LLP_tot}, epsilon_polar: {epsilon_polar}, "
                  f"epsilon_azimuthal: {epsilon_azimuthal}, P_decay_averaged: {P_decay_averaged}, "
                  f"br_visible_val: {br_visible_val}, N_ev_tot: {N_ev_tot}")
        else:
            print("Error: First line does not match expected format.")
            sys.exit(1)

        # Skip any empty lines
        while True:
            line = f.readline()
            if not line:
                break
            if line.strip() != '':
                break
        # Now process the rest of the file
        # Extract channels and sample_points
        channels = {}
        current_channel = None
        current_channel_size = 0
        current_data = []
        # If the line we just read is a channel header, process it
        if line.strip().startswith('#<process='):
            match = re.match(
                r'#<process=(?P<channel>.*?);\s*sample_points=(?P<channel_size>[\d\.\+\-eE]+)>', line.strip())
            if match:
                current_channel = match.group('channel')
                current_channel_size = int(float(match.group('channel_size')))
                current_data = []
            else:
                print(f"Error: Could not parse channel line: {line}")
        else:
            print("Error: Expected channel header after first line.")
            sys.exit(1)

        # Continue reading the file
        for line in f:
            line = line.strip()
            if line.startswith('#<process='):
                # This is a new channel
                match = re.match(
                    r'#<process=(?P<channel>.*?);\s*sample_points=(?P<channel_size>[\d\.\+\-eE]+)>', line)
                if match:
                    if current_channel is not None:
                        # Save the data of the previous channel
                        channels[current_channel] = {
                            'size': current_channel_size, 'data': current_data}
                    current_channel = match.group('channel')
                    current_channel_size = int(float(match.group('channel_size')))
                    current_data = []
                else:
                    print(f"Error: Could not parse channel line: {line}")
            elif line == '':
                # Empty line, skip
                continue
            else:
                # This is data
                current_data.append(line)

        # After the loop, save the last channel's data
        if current_channel is not None:
            channels[current_channel] = {
                'size': current_channel_size, 'data': current_data}

    return finalEvents, epsilon_polar, epsilon_azimuthal, br_visible_val, coupling_squared, channels
