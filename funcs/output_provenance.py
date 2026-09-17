"""Shared rate-aware output provenance for batch and interactive EventCalc."""

import json
import os
import re
import tempfile


GENERATOR_PROVENANCE_SCHEMA = "eventcalc-generator-provenance-v1"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _identity(value, label):
    value = str(value).strip()
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(
            "%s must be a lowercase hyphenated identifier" % label
        )
    return value


def validate_generator_provenance(provenance):
    """Validate one complete hadronization-model identity.

    ``None`` intentionally remains valid: old interactive and legacy batch
    outputs keep their byte-for-byte historical filenames and do not acquire
    a new sidecar.  A portable-model output, on the other hand, must carry all
    of the coordinates needed to distinguish physics variants and contracts.
    """
    if provenance is None:
        return None
    if not isinstance(provenance, dict):
        raise ValueError("generator provenance must be a dictionary")
    required = {
        "schema", "backend", "model_id", "variant_id", "variation_id",
        "contract_path", "contract_sha256", "runtime_card_path",
        "runtime_card_sha256", "route_ownership",
    }
    missing = required - set(provenance)
    unknown = set(provenance) - required
    if missing:
        raise ValueError(
            "generator provenance misses: %s" % ", ".join(sorted(missing))
        )
    if unknown:
        raise ValueError(
            "generator provenance has unknown fields: %s"
            % ", ".join(sorted(unknown))
        )
    if provenance["schema"] != GENERATOR_PROVENANCE_SCHEMA:
        raise ValueError("unsupported generator provenance schema")
    result = dict(provenance)
    for field in ("backend", "model_id", "variant_id", "variation_id"):
        result[field] = _identity(result[field], field)
    for field in ("contract_sha256", "runtime_card_sha256"):
        digest = str(result[field]).strip().lower()
        if _SHA256.fullmatch(digest) is None:
            raise ValueError("%s must be a full lowercase SHA-256" % field)
        result[field] = digest
    for field in ("contract_path", "runtime_card_path"):
        path = str(result[field]).strip()
        if not path:
            raise ValueError("%s must be non-empty" % field)
        result[field] = path
    ownership = result["route_ownership"]
    if not isinstance(ownership, dict) or not ownership:
        raise ValueError("route_ownership must be a non-empty dictionary")
    # Prove now that the sidecar payload is JSON serializable rather than
    # failing only after an expensive event run has completed.
    json.dumps(ownership, allow_nan=False, sort_keys=True)
    return result


def build_generator_tag(
        exhad_bench, exhad_seed, stock_pythia_pool=False,
        eventcalc_direct=False, generator_provenance=None):
    """Build a filename-safe tag for one effective mass-level route."""
    provenance = validate_generator_provenance(generator_provenance)
    if stock_pythia_pool:
        if exhad_bench or eventcalc_direct or provenance is not None:
            raise ValueError(
                "the ALP stock-Pythia pool cannot be tagged as exhad/direct"
            )
        return "stock-pythia-alp-seed%d" % int(exhad_seed)
    if eventcalc_direct:
        if exhad_bench or provenance is not None:
            raise ValueError(
                "an effective exhad sample cannot be tagged as direct"
            )
        return "eventcalc-direct-seed%d" % int(exhad_seed)
    if not exhad_bench:
        if provenance is not None:
            raise ValueError(
                "generator provenance requires an effective exhad benchmark"
            )
        return "raw-pythia-seed%d" % int(exhad_seed)
    bench = str(exhad_bench).strip()
    if any(not (char.isalnum() or char in "-.") for char in bench):
        raise ValueError("resolved exhad benchmark is not filename-safe")
    if provenance is not None:
        bench = (
            "%s-m.%s-v.%s-u.%s-c.%s"
            % (
                bench,
                provenance["model_id"],
                provenance["variant_id"],
                provenance["variation_id"],
                provenance["contract_sha256"][:12],
            )
        )
    return f"exhad-{bench}-seed{int(exhad_seed)}"


def add_hnl_source_tag(generator_tag, llp):
    """Bind either HNL generator mode to the signed rate-table source."""
    digest = getattr(llp, "hnl_decay_source_sha256", None)
    if getattr(llp, "LLP_name", None) != "HNL" or not digest:
        return generator_tag
    return "%s-src%s" % (generator_tag, str(digest)[:12])


def effective_generator_tag(
        decay_products, llp, selected_decay_indices, exhad_seed,
        stock_pythia_pool=False):
    """Resolve the rate-aware per-mass output provenance tag."""
    effective_exhad_bench = decay_products.effective_exhad_bench(
        llp.mass,
        llp.PDGs,
        selected_decay_indices,
        BrRatio=llp.BrRatios_distr,
        llp_name=llp.LLP_name,
        particle_path=llp.particle_path,
        exhad_variant=llp.scalar_lifetime,
    )
    direct = (
        effective_exhad_bench is None
        and not stock_pythia_pool
        and not decay_products.selected_sample_requires_pythia(
            llp.PDGs, llp.BrRatios_distr, selected_decay_indices
        )
    )
    tag = build_generator_tag(
        effective_exhad_bench,
        exhad_seed,
        stock_pythia_pool=stock_pythia_pool,
        eventcalc_direct=direct,
    )
    return effective_exhad_bench, add_hnl_source_tag(tag, llp)


def effective_generator_identity(
        decay_products, llp, selected_decay_indices, exhad_seed,
        stock_pythia_pool=False, provenance_resolver=None):
    """Return ``(bench, tag, provenance)`` for one mass-level route.

    The separate resolver keeps this module independent of the optional exhad
    bridge.  It is invoked only after a positive effective exhad route has
    been established, so a below-boundary/direct point never receives a
    portable-model label accidentally.
    """
    effective_exhad_bench = decay_products.effective_exhad_bench(
        llp.mass,
        llp.PDGs,
        selected_decay_indices,
        BrRatio=llp.BrRatios_distr,
        llp_name=llp.LLP_name,
        particle_path=llp.particle_path,
        exhad_variant=llp.scalar_lifetime,
    )
    provenance = None
    if effective_exhad_bench is not None and provenance_resolver is not None:
        provenance = provenance_resolver(effective_exhad_bench)
    direct = (
        effective_exhad_bench is None
        and not stock_pythia_pool
        and not decay_products.selected_sample_requires_pythia(
            llp.PDGs, llp.BrRatios_distr, selected_decay_indices
        )
    )
    tag = build_generator_tag(
        effective_exhad_bench,
        exhad_seed,
        stock_pythia_pool=stock_pythia_pool,
        eventcalc_direct=direct,
        generator_provenance=provenance,
    )
    return (
        effective_exhad_bench,
        add_hnl_source_tag(tag, llp),
        provenance,
    )


def write_generator_sidecar(output_path, provenance):
    """Atomically write ``<output>.generator.json`` for a modelled sample."""
    resolved = validate_generator_provenance(provenance)
    if resolved is None:
        return None
    output_path = os.path.abspath(str(output_path))
    sidecar_path = output_path + ".generator.json"
    document = {
        "schema": GENERATOR_PROVENANCE_SCHEMA,
        "output_file": os.path.basename(output_path),
        "generator": resolved,
    }
    payload = (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    directory = os.path.dirname(sidecar_path)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix="." + os.path.basename(sidecar_path) + ".",
        dir=directory,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, sidecar_path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
    return sidecar_path
