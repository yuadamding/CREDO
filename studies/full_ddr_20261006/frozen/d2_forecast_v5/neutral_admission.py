"""Strict neutral-probe admission mapping, not GPU numerical qualification."""
import copy

NEUTRAL = ["/opt/credo/bin/python", "-c", "print('CREDO_NEUTRAL_IMAGE_STARTED')"]
ZERO = {
    key: {"cpu": "0", "memory": "0", "nvidia.com/gpu": "0"}
    for key in ("requests", "limits")
}
DECLARED = {
    key: {"cpu": "1", "memory": "4Gi", "nvidia.com/gpu": "0"}
    for key in ("requests", "limits")
}


LIVE = {key: {"cpu": "0", "memory": "4Gi", "nvidia.com/gpu": "0"} for key in ("requests", "limits")}

def same(expected, observed, path=""):
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            raise RuntimeError("Admission type changed: " + path)
        for key, value in expected.items():
            if key not in observed:
                raise RuntimeError("Admission field removed: " + path + "/" + key)
            same(value, observed[key], path + "/" + key)
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            raise RuntimeError("Admission list changed: " + path)
        for i, value in enumerate(expected):
            same(value, observed[i], path + "/" + str(i))
    elif expected != observed:
        raise RuntimeError("Protected field changed: " + path)


def audit_neutral(job, manifest):
    """Bound neutral admission: dry-run 0/0, created 0/4Gi; actual either known form."""
    expected = copy.deepcopy(manifest)
    source = expected["spec"]["template"]["spec"]
    actual = job["spec"]["template"]["spec"]
    if len(source["containers"]) != 1 or len(actual["containers"]) != 1:
        raise RuntimeError("Neutral probe must have one container")
    wanted, observed = source["containers"][0], actual["containers"][0]
    if source.get("securityContext") != actual.get("securityContext") or wanted.get(
        "securityContext"
    ) != observed.get("securityContext"):
        raise RuntimeError("Unbound static security context mutation")
    if source.get("volumes") != actual.get("volumes") or wanted.get(
        "volumeMounts"
    ) != observed.get("volumeMounts"):
        raise RuntimeError("Unbound static mount mutation")
    if wanted["command"] != NEUTRAL or observed["command"] != NEUTRAL:
        raise RuntimeError("Neutral command changed")
    mask = [dict(name="NVIDIA_VISIBLE_DEVICES", value="none")]
    if wanted["env"] != mask or observed["env"] != mask:
        raise RuntimeError("Static visibility mask changed")
    if wanted["resources"] not in (DECLARED, ZERO, LIVE) or observed["resources"] not in (ZERO, LIVE):
        raise RuntimeError(
            "Static resource mapping differs from observed zero contract"
        )
    wanted["resources"] = copy.deepcopy(observed["resources"])
    # The neutral, zero-GPU site default is explicitly bound, not a GPU request.
    if actual.get("nodeSelector") != {"gpu-type": "A100"}:
        raise RuntimeError("Static admission node selector changed")
    source["nodeSelector"] = {"gpu-type": "A100"}
    same(expected, job)
    allowed = set(source) | {
        "dnsPolicy",
        "schedulerName",
        "priorityClassName",
        "priority",
        "preemptionPolicy",
        "tolerations",
        "serviceAccount",
        "serviceAccountName",
        "nodeSelector",
        "affinity",
    }
    if set(actual) - allowed:
        raise RuntimeError("Unexpected admitted Pod field")
    if (
        set(observed)
        - set(wanted)
        - {"terminationMessagePath", "terminationMessagePolicy"}
    ):
        raise RuntimeError("Unexpected admitted container field")
    if any(
        k in observed
        for k in ("lifecycle", "livenessProbe", "startupProbe", "readinessProbe")
    ):
        raise RuntimeError("Unexpected executable probe mutation")


