"""Local synthetic packing/control checks; no remote or scheduler calls."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_launch as module
import preparation_control as control


def fixture_payload():
    reader = module.read_regular

    def with_synthetic_worker(path):
        if Path(path) == module.BASE / "prepare_worker.py":
            return b'"""Synthetic no-op fixture, never launched."""\n'
        return reader(path)

    with patch.object(module, "read_regular", side_effect=with_synthetic_worker):
        return module.assemble()


class PackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = fixture_payload()

    def test_exact_resources_mount_and_no_runtime_deadline(self):
        plan = json.loads(self.payload["PLAN.json"])
        job = json.loads(self.payload["gpu-job.json"])
        self.assertEqual(plan["runtime_deadline_policy"], module.POLICY)
        self.assertEqual(plan["automatic_retries"], 0)
        self.assertEqual(plan["parameter_updates"], 0)
        self.assertFalse(plan["D1_endpoint_access"])
        self.assertNotIn("activeDeadlineSeconds", job["spec"])
        pod = job["spec"]["template"]["spec"]
        self.assertNotIn("activeDeadlineSeconds", pod)
        self.assertFalse(pod["automountServiceAccountToken"])
        self.assertEqual(pod["restartPolicy"], "Never")
        self.assertEqual(job["spec"]["backoffLimit"], 0)
        self.assertTrue(job["spec"]["suspend"])
        self.assertEqual(pod["nodeSelector"]["gpu-type"], "H100")
        worker = pod["containers"][0]
        self.assertEqual(worker["image"], module.IMAGE)
        for side in ("requests", "limits"):
            self.assertEqual(worker["resources"][side]["cpu"], "24")
            self.assertEqual(worker["resources"][side]["memory"], "200Gi")
            self.assertEqual(worker["resources"][side]["nvidia.com/gpu"], "1")
        env = {row["name"]: row for row in worker["env"]}
        self.assertEqual(len(env), len(worker["env"]))
        self.assertEqual(
            env["CREDO_PLAN_SHA256"]["value"], module.sha(self.payload["PLAN.json"])
        )
        self.assertEqual(env["OMP_NUM_THREADS"]["value"], "8")
        self.assertNotIn("NVIDIA_VISIBLE_DEVICES", env)
        self.assertEqual(
            pod["volumes"][0]["persistentVolumeClaim"]["claimName"], "yding4-gpu-home"
        )

    def test_reused_lifecycle_and_static_sources_unchanged(self):
        for name in ("lifecycle_core.py", "runtime_policy.py", "deployment.py"):
            self.assertEqual(module.sha(self.payload[name]), module.PINNED[name])
        authority = json.loads(self.payload["STATIC_REUSE.json"])
        for name, digest in authority["files"].items():
            self.assertEqual(module.sha(self.payload[name]), digest)

    def test_archive_deterministic_safe_and_inventory_complete(self):
        first = module.pack(self.payload)
        second = module.pack(dict(reversed(list(self.payload.items()))))
        self.assertEqual(first, second)
        spec = importlib.util.spec_from_file_location(
            "fixture_deployment", module.HISTORICAL / "deployment.py"
        )
        deployment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(deployment)
        members, archive, build = first
        restored, inventory = deployment.archive_payload(
            archive, build["archive_sha256"], build["inventory_sha256"]
        )
        self.assertEqual(restored, members)
        self.assertEqual(inventory, build["files"])

    def test_bundle_local_publication_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as parent:
            path = Path(parent) / "attempt"
            with patch.object(module, "assemble", return_value=self.payload):
                receipt = module.build(path)
                self.assertEqual(
                    module.sha((path / "bundle.tar.gz").read_bytes()),
                    receipt["archive_sha256"],
                )
                static_spec = importlib.util.spec_from_file_location(
                    "fixture_static_reuse", path / "bundle/gpu_v2_static.py"
                )
                static = importlib.util.module_from_spec(static_spec)
                static_spec.loader.exec_module(static)
                self.assertTrue(
                    static.validate(
                        path / "bundle", json.loads(self.payload["PLAN.json"])
                    )
                )
                with self.assertRaises(FileExistsError):
                    module.build(path)
            # Restore test-only directory permissions so TemporaryDirectory owns cleanup.
            for directory in path.rglob("*"):
                if directory.is_dir():
                    directory.chmod(0o700)

    def test_protected_runtime_deadline_in_template_rejected(self):
        plan = json.loads(self.payload["PLAN.json"])
        template = json.loads(self.payload["gpu-job.json"])
        template["spec"]["activeDeadlineSeconds"] = 600
        with self.assertRaisesRegex(ValueError, "deadlines"):
            module.render_manifest(template, plan)

    def test_archive_rejects_unsafe_member_names(self):
        changed = dict(self.payload)
        changed["../escape.py"] = b"bad"
        with self.assertRaisesRegex(ValueError, "Noncanonical"):
            module.pack(changed)

    def test_control_and_publisher_are_host_python36_syntax(self):
        import ast

        for name in (
            "control.py",
            "publication.py",
            "lifecycle_core.py",
            "runtime_policy.py",
        ):
            ast.parse(self.payload[name], feature_version=(3, 6))

    def test_publisher_rejects_any_wrong_root(self):
        from types import SimpleNamespace

        namespace = {}
        with patch.dict(sys.modules, {"deployment": SimpleNamespace()}):
            exec(self.payload["publication.py"], namespace)
        plan = json.loads(self.payload["PLAN.json"])
        self.assertEqual(
            [str(p) for p in namespace["validated_roots"](plan)],
            [module.CONTROL_ROOT, module.HOME_RUN],
        )
        for key in ("control_root", "bundle_root", "worker_output", "run_id"):
            altered = copy.deepcopy(plan)
            altered[key] += "-wrong"
            with self.assertRaises(ValueError):
                namespace["validated_roots"](altered)

    def test_read_rejects_symlink_and_symlink_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actual = root / "file"
            actual.write_bytes(b"fixture")
            (root / "link").symlink_to(actual)
            with self.assertRaises(ValueError):
                module.read_regular(root / "link")
            (root / "directory").symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                module.read_regular(root / "directory/file")

    def test_monitor_terminal_requires_application_and_exact_cleanup(self):
        from types import SimpleNamespace

        calls = []
        fake_core = SimpleNamespace(
            get_job=lambda name: {"job": name},
            verify=lambda *args: None,
            complete=lambda value: (True, False),
            capture=lambda *args: [{"metadata": {"uid": "pod"}}],
            cleanup=lambda binding, root: calls.append("cleanup"),
            sha=lambda path: "hash",
            publish=lambda path, value: calls.append((path.name, value)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "startup-validated.json").write_text("{}")
            with (
                patch.object(control, "core", fake_core, create=True),
                patch.object(control, "validate_result", return_value=True),
            ):
                control.monitor(
                    {"run_id": "run", "worker_output": tmp},
                    {"job": "job", "uid": "uid"},
                    root,
                )
            self.assertEqual(calls[-1], "cleanup")
            self.assertEqual(calls[0][0], "terminal.json")
            self.assertTrue(calls[0][1]["application_pass"])
            calls.clear()
            with (
                patch.object(control, "core", fake_core, create=True),
                patch.object(
                    control, "validate_result", side_effect=ValueError("bad result")
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "Preparation failed"):
                    control.monitor(
                        {"run_id": "run", "worker_output": tmp},
                        {"job": "job", "uid": "uid"},
                        root,
                    )
            self.assertFalse(calls[0][1]["application_pass"])
            self.assertEqual(calls[-1], "cleanup")


if __name__ == "__main__":
    unittest.main()
