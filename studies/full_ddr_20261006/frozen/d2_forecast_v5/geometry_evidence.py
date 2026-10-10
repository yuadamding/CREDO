"""Pure receipt validation shared by Python-3.6 controller and CUDA worker."""


def validate_admission(record):
    expected = dict(status="all_fitting_geometry_support_admitted", particles=512,
        fitting_laws=46809, paired_laws=45014, maximum_geometry_elements=16*1024**2,
        dropped_cells=0, solver="unchanged_geomloss_tensorized", observation_weights_changed=False)
    if not isinstance(record, dict) or any(record.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Full fitting-support admission differs")
    maximum = record.get("maximum_singleton_elements")
    support = record.get("maximum_support")
    if (type(maximum) is not int or not 0 < maximum <= expected["maximum_geometry_elements"]
            or type(support) is not int or support < 1 or (512+support)**2 > maximum):
        raise RuntimeError("Support admission workspace arithmetic differs")
    rows = record.get("exact_width_laws")
    if not isinstance(rows, list):
        raise RuntimeError("Exact-width support inventory missing")
    seen = set()
    for row in rows:
        name, n = row.get("measure_id"), row.get("support")
        if not isinstance(name, str) or not name or name in seen or type(n) is not int or n < 1:
            raise RuntimeError("Invalid or duplicate exact-width identity")
        seen.add(name)
        padded = 1 << (n-1).bit_length()
        if (row.get("width") != n or row.get("pairwise_elements") != (512+n)**2
                or (512+n)**2 > maximum or (512+padded)**2 <= expected["maximum_geometry_elements"]):
            raise RuntimeError("Exact-width workspace policy differs")
    return [row["measure_id"] for row in rows]


def validate_geometry_qualification(complete):
    required = validate_admission(complete.get("geometry_admission"))
    if complete.get("full_support_challenge_qualified") is not True or len(complete.get("profiles", [])) != 18:
        raise RuntimeError("Full-support qualification incomplete")
    for row in complete["profiles"]:
        challenge = row.get("full_support_challenge", {})
        if (challenge.get("exact_width_measure_ids") != required or challenge.get("full_batch_guides") != 1086
                or challenge.get("all_required_included") is not True):
            raise RuntimeError("Qualified shape omitted an exact-width support law")
