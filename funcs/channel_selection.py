"""Shared user-facing decay-channel grouping and selection.

Decay tables retain separate ``Jets-*`` rows as model inputs, but users select
them only as one aggregate group.  This policy is independent of whether raw
Pythia or exhad handles the resulting partonic events.
"""


ALL_JETS_LABEL = "All Jets-* channels"
ALL_JETS_TOKEN = "jets"
ALL_JETS_ALIASES = frozenset({ALL_JETS_TOKEN, "all-jets"})


def is_jet_channel(channel):
    """Return whether a decay-table label belongs to the jet group."""
    return str(channel).strip().startswith("Jets-")


def channel_choices(decay_channels):
    """Return displayed ``(label, raw_indices)`` channel choices.

    Non-jet rows remain individual choices.  All ``Jets-*`` rows become one
    choice inserted where the first jet row occurred.
    """
    names = [str(channel) for channel in decay_channels]
    jet_indices = tuple(i for i, name in enumerate(names)
                        if is_jet_channel(name))
    first_jet = jet_indices[0] if jet_indices else None

    choices = []
    for i, name in enumerate(names):
        if is_jet_channel(name):
            if i == first_jet:
                choices.append((ALL_JETS_LABEL, jet_indices))
            continue
        choices.append((name, (i,)))
    return choices


def _unique_sorted(indices):
    return sorted(set(int(index) for index in indices))


def resolve_channel_tokens(tokens, decay_channels):
    """Resolve batch tokens to raw decay-table indices.

    ``jets`` is canonical and ``all-jets`` is an alias.  Numeric indices refer
    to the collapsed displayed choices, never to hidden individual jet rows.
    """
    names = [str(channel) for channel in decay_channels]
    choices = channel_choices(names)
    raw_tokens = [str(token) for token in (tokens or [])]
    lowered = [token.casefold() for token in raw_tokens]

    if not raw_tokens or lowered == ["all"]:
        return list(range(len(names)))
    if "all" in lowered:
        raise ValueError("'all' cannot be combined with other decay channels")

    jet_indices = [i for i, name in enumerate(names)
                   if is_jet_channel(name)]
    selected = []
    for token, token_lower in zip(raw_tokens, lowered):
        if token_lower in ALL_JETS_ALIASES:
            if not jet_indices:
                raise ValueError(
                    "this decay table has no Jets-* channels to select")
            selected.extend(jet_indices)
            continue

        if token.lstrip("+-").isdigit():
            displayed_index = int(token) - 1
            if not 0 <= displayed_index < len(choices):
                raise ValueError(
                    "channel index %s out of range 1..%d after grouping "
                    "Jets-* rows" % (token, len(choices)))
            selected.extend(choices[displayed_index][1])
            continue

        if token in names:
            table_index = names.index(token)
            if is_jet_channel(names[table_index]):
                raise ValueError(
                    "individual jet channel %r cannot be selected; use "
                    "--channels %s to select every Jets-* row"
                    % (token, ALL_JETS_TOKEN))
            selected.append(table_index)
            continue

        available = [name for name, _ in choices
                     if name != ALL_JETS_LABEL]
        if jet_indices:
            available.append(ALL_JETS_TOKEN)
        raise ValueError(
            "unknown channel %r; available: %s"
            % (token, ", ".join(available)))

    return _unique_sorted(selected)


def selected_channel_labels(selected_indices, decay_channels):
    """Return collapsed labels for an already-resolved selection."""
    selected = set(int(index) for index in selected_indices)
    labels = []
    for label, raw_indices in channel_choices(decay_channels):
        overlap = selected.intersection(raw_indices)
        if not overlap:
            continue
        if label == ALL_JETS_LABEL and overlap != set(raw_indices):
            raise ValueError(
                "internal selection contains only part of the Jets-* group")
        labels.append(label)
    return labels


def prompt_decay_channels(decay_channels, input_fn=None, output_fn=None):
    """Prompt for collapsed interactive choices and expand the result."""
    if input_fn is None:
        input_fn = input
    if output_fn is None:
        output_fn = print

    choices = channel_choices(decay_channels)
    output_fn("\nSelect the decay modes:")
    output_fn("0. All")
    for i, (label, _) in enumerate(choices, 1):
        output_fn(f"{i}. {label}")

    user_input = input_fn(
        "Enter the numbers of the decay channels to select "
        "(separated by spaces): ")
    try:
        displayed_indices = [int(token) for token in user_input.strip().split()]
        if not displayed_indices:
            raise ValueError("No selection made.")
        if 0 in displayed_indices:
            return list(range(len(decay_channels)))

        selected = []
        for displayed_index in displayed_indices:
            if not 1 <= displayed_index <= len(choices):
                raise ValueError(f"Invalid index {displayed_index}.")
            selected.extend(choices[displayed_index - 1][1])
        return _unique_sorted(selected)
    except ValueError as exc:
        raise ValueError(
            f"Invalid input for decay channel selection: {exc}") from exc
