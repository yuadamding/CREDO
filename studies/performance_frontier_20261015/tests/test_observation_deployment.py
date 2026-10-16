"""Metadata/fabricated deployment checks: no optimizer, GPU, remote or data I/O."""
import ast
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from deployment import build
from deployment import common
from frontier.coordination import WorkBinding

STUDY = Path(__file__).absolute().parents[1]
WORKSPACE = STUDY.parent
RETAINED = WORKSPACE / 'credo_full_ddr_gse314342_20261006_v1/comparator_replay_assessment_launch_v1/package_v1/payload/comparator_replay_assessment_v1/bundle'
PUBLISHER = WORKSPACE / 'credo_full_ddr_gse314342_20261006_v1/final_refit_v1/bundle/deployment.py'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fabricated_hardware():
    return dict(device='cuda:0', name='NVIDIA H100 80GB HBM3', total_memory_bytes=85028372480,
                capability=[9, 0], visible_devices=1, precision='FP32', tf32=False,
                deterministic_algorithms=True, required_headroom_bytes=8 * common.GIB)


def admission(mode='qualify'):
    result = dict(schema='credo.frontier.observation-admission-export.v1', run_id='credo-frontier-test-qualification',
                status='reserved', storage_reserved_bytes=(16 if mode == 'qualify' else 4) * common.GIB,
                scientific_update_reservation=0 if mode == 'qualify' else 20000,
                aggregate_reserved_output_bytes=16 * common.GIB, live_h100_reservations=1,
                journal_event_sha256='a' * 64, fixture=True)
    if mode == 'fit':
        result['work_binding'] = WorkBinding.representation(('D1', 'D2')).to_dict()
        result['qualification_sha256'] = 'd' * 64
        result['qualified_hardware'] = fabricated_hardware()
        result['qualified_peak_reserved_bytes'] = 4 * common.GIB
    return result


def bundle(mode='qualify', extra=None):
    return build.construct(STUDY, RETAINED, 'observation_qualification_v1',
                           'credo-frontier-test-qualification', admission(mode), mode, extra)


class ObservationDeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload, cls.archive, cls.report = bundle()
        cls.plan = json.loads(cls.payload['PLAN.json'])

    def test_deterministic_archive_and_exact_retained_bytes(self):
        second_payload, second_archive, second_report = bundle()
        self.assertEqual((self.payload, self.archive, self.report), (second_payload, second_archive, second_report))
        for name, digest in build.PINNED.items():
            if name != 'gpu-job.json':
                self.assertEqual(common.identity(self.payload[name])['sha256'], digest)

    def test_allowlisted_source_separate_from_output(self):
        source = json.loads(self.payload['SOURCE_MANIFEST.json'])
        self.assertEqual(set(source['files']), set(build.SOURCE_MEMBERS))
        self.assertEqual({name for name in self.payload if name.startswith('source/')},
                         {'source/' + name for name in build.SOURCE_MEMBERS})
        worker = json.loads(self.payload['WORKER_CONTRACT.json'])
        self.assertEqual(worker['source_root'], self.plan['bundle_root'] + '/source')
        self.assertNotEqual(self.plan['worker_output'], worker['output_root'])
        self.assertNotIn('dataset_root', worker)
        self.assertEqual(worker['output_limit_bytes'], 16 * common.GIB)

    def test_scalar_manifest_preserves_institutional_surface(self):
        actual = json.loads(self.payload['gpu-job.json'])
        old = json.loads((RETAINED / 'gpu-job.json').read_bytes())
        self.assertEqual(actual['spec']['completions'], 1)
        self.assertEqual(actual['spec']['parallelism'], 1)
        self.assertTrue(actual['spec']['suspend'])
        self.assertEqual(actual['spec']['backoffLimit'], 0)
        pod, prior = actual['spec']['template']['spec'], old['spec']['template']['spec']
        for key in ('volumes', 'securityContext', 'automountServiceAccountToken', 'enableServiceLinks', 'nodeSelector'):
            self.assertEqual(pod[key], prior[key])
        for key in ('image', 'resources', 'securityContext', 'volumeMounts'):
            self.assertEqual(pod['containers'][0][key], prior['containers'][0][key])
        self.assertEqual(pod['containers'][0]['command'][-1], self.report['plan_sha256'])
        self.assertNotIn('activeDeadlineSeconds', actual['spec'])
        self.assertNotIn('activeDeadlineSeconds', pod)

    def test_retained_archive_validator_accepts_new_exact_layout(self):
        publisher = module(PUBLISHER, 'frontier_test_retained_publisher')
        payload, inventory = publisher.archive_payload(self.archive, self.report['archive_sha256'], self.report['inventory_sha256'])
        self.assertEqual(payload, self.payload)
        self.assertEqual(set(inventory), set(payload) - {'INVENTORY.json'})
        with tarfile.open(fileobj=io.BytesIO(self.archive), mode='r:gz') as packed:
            self.assertTrue(all(item.isfile() and item.name.startswith('bundle/') for item in packed.getmembers()))
            self.assertTrue(all(item.uid == item.gid == item.mtime == 0 for item in packed.getmembers()))

    def test_static_reuse_checks_without_scheduler(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, raw in self.payload.items():
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
            static = module(root / 'gpu_v2_static.py', 'frontier_test_static_reuse')
            self.assertTrue(static.validate(root, self.plan))

    def test_admission_rejects_missing_or_excessive_capacity_and_slots(self):
        for key, value in (('live_h100_reservations', 5), ('live_h100_reservations', True),
                           ('storage_reserved_bytes', 0), ('aggregate_reserved_output_bytes', 385 * common.GIB),
                           ('scientific_update_reservation', 1), ('status', 'complete'), ('journal_event_sha256', '')):
            changed = admission()
            changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                common.validate_admission(self.plan, changed)

    def test_authority_and_deadline_resource_bindings_reject_changes(self):
        authority = json.loads(self.payload['OBSERVATION_PHASE_CONTRACT.json'])
        self.assertEqual(common.validate_authority(authority), authority)
        authority['maximum_concurrent_h100'] = 5
        with self.assertRaises(ValueError):
            common.validate_authority(authority)
        for key, value in (('automatic_retries', 1), ('elapsed_time_deadline_seconds', 86400),
                           ('mode', 'forecast'), ('scientific_output', common.PROJECT),
                           ('maximum_new_scientific_updates', 1)):
            changed = dict(self.plan, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                common.validate_plan(changed)

    def test_fit_requires_exact_native_and_qualification_bindings(self):
        with self.assertRaises(ValueError):
            bundle('fit', {})
        fit = dict(metadata_root=build.NATIVE_ROOT,
                   reader_source_root=self.plan['bundle_root'] + '/retained_reader', dataset_root=build.NATIVE_ROOT,
                   fold_id='subset-D1-D2', training_donors=['D1', 'D2'], metadata_index_sha256='a' * 64,
                   rna_axis_sha256='b' * 64, input_access_sha256='c' * 64,
                   qualification_receipt='/bound/QUALIFICATION_COMPLETE.json', qualification_sha256='d' * 64)
        payload, _, report = bundle('fit', fit)
        self.assertEqual(json.loads(payload['WORKER_CONTRACT.json'])['training_donors'], ['D1', 'D2'])
        self.assertEqual(json.loads(payload['PLAN.json'])['maximum_new_scientific_updates'], 20000)
        self.assertFalse(report['runtime_qualification_established'])
        self.assertEqual({key for key in payload if key.startswith('retained_reader/')},
                         {'retained_reader/' + name for name in build.READER_MEMBERS})
        for name, digest in build.READER_MEMBERS.items():
            self.assertEqual(common.identity(payload['retained_reader/' + name])['sha256'], digest)
        self.assertFalse(any(key.endswith(('.parquet', '.npz', '.pt')) for key in payload))
        self.assertEqual(payload['SOURCE_MANIFEST.json'], self.payload['SOURCE_MANIFEST.json'])
        plan = json.loads(payload['PLAN.json'])
        manifest = json.loads(payload['gpu-job.json'])
        qualified_manifest = json.loads(self.payload['gpu-job.json'])
        self.assertEqual(plan['qualified_hardware'], fabricated_hardware())
        self.assertEqual(plan['qualification_sha256'], fit['qualification_sha256'])
        self.assertEqual(plan['qualified_peak_reserved_bytes'], 4 * common.GIB)
        self.assertEqual(plan['qualified_gpu_product'], 'NVIDIA-H100-80GB-HBM3')
        terms = manifest['spec']['template']['spec']['affinity']['nodeAffinity'][
            'requiredDuringSchedulingIgnoredDuringExecution']['nodeSelectorTerms']
        self.assertEqual(terms[0]['matchExpressions'][0]['values'], ['NVIDIA-H100-80GB-HBM3'])
        self.assertGreater(len(qualified_manifest['spec']['template']['spec']['affinity']['nodeAffinity'][
            'requiredDuringSchedulingIgnoredDuringExecution']['nodeSelectorTerms'][0]['matchExpressions'][0]['values']), 1)
        self.assertNotIn('qualified_hardware', self.plan)
        invalid = dict(fit, reader_source_root='/unverified/historical/reader')
        with self.assertRaises(ValueError):
            bundle('fit', invalid)
        fit['training_donors'] = ['D1', 'D2', 'D3', 'D4']
        with self.assertRaises(ValueError):
            bundle('fit', fit)

    def test_fit_cannot_recombine_another_subsets_valid_input_and_reservation(self):
        fit = dict(metadata_root=build.NATIVE_ROOT,
                   reader_source_root=self.plan['bundle_root'] + '/retained_reader', dataset_root=build.NATIVE_ROOT,
                   fold_id='subset-D1-D3', training_donors=['D1', 'D3'], metadata_index_sha256='a' * 64,
                   rna_axis_sha256='b' * 64, input_access_sha256='c' * 64,
                   qualification_receipt='/bound/QUALIFICATION_COMPLETE.json', qualification_sha256='d' * 64)
        # Both subsets are registered and the path belongs to this phase. Only
        # their cross-file scientific identity is inconsistent.
        with self.assertRaisesRegex(ValueError, 'admission work binding'):
            bundle('fit', fit)

    def test_fit_binding_requires_exact_full_interval_and_canonical_identity(self):
        worker = dict(training_donors=['D1', 'D2'], fold_id='subset-D1-D2',
                      trajectory_id='observation-D1-D2-seed0', qualification_sha256='d' * 64)
        registered = json.loads(self.payload['OBSERVATION_PROTOCOL.json'])['training_subsets']
        self.assertEqual(build.validate_fit_binding(worker, admission('fit'), registered),
                         WorkBinding.representation(('D1', 'D2')))
        changed_bindings = [None, WorkBinding.representation(('D1', 'D3')).to_dict(),
                            WorkBinding.representation(('D1', 'D2'), target_updates=19999).to_dict(),
                            WorkBinding.representation(('D1', 'D2'), start_updates=1).to_dict(),
                            WorkBinding.other('qualification', 'e' * 64).to_dict()]
        boolean_interval = WorkBinding.representation(('D1', 'D2')).to_dict()
        boolean_interval['start_updates'] = False
        changed_bindings.append(boolean_interval)
        for binding in changed_bindings:
            changed = dict(admission('fit'), work_binding=binding)
            with self.subTest(binding=binding), self.assertRaises(ValueError):
                build.validate_fit_binding(worker, changed, registered)
        for key, value in [('fold_id', 'inner-D1-D2'), ('fold_id', 'subset-D1-D3'),
                           ('trajectory_id', 'observation-D1-D3-seed0'),
                           ('trajectory_id', 'observation-D1-D2-seed1'),
                           ('training_donors', ['D2', 'D1'])]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                build.validate_fit_binding(dict(worker, **{key: value}), admission('fit'), registered)

    def test_fit_hardware_must_be_receipt_bound_and_already_allowed_by_template(self):
        worker = dict(training_donors=['D1', 'D2'], fold_id='subset-D1-D2',
                      trajectory_id='observation-D1-D2-seed0', qualification_sha256='d' * 64)
        registered = json.loads(self.payload['OBSERVATION_PROTOCOL.json'])['training_subsets']
        for key, value in [('name', 'NVIDIA H100 PCIe'), ('total_memory_bytes', 75 * common.GIB - 1),
                           ('total_memory_bytes', 82 * common.GIB + 1), ('total_memory_bytes', True),
                           ('capability', [8, 0]), ('visible_devices', True), ('required_headroom_bytes', 0)]:
            changed = admission('fit')
            changed['qualified_hardware'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'qualification hardware'):
                build.validate_fit_binding(worker, changed, registered)
        with self.assertRaisesRegex(ValueError, 'qualification SHA'):
            build.validate_fit_binding(dict(worker, qualification_sha256='e' * 64), admission('fit'), registered)
        changed = admission('fit')
        changed.pop('qualified_hardware')
        with self.assertRaises(ValueError):
            build.validate_fit_binding(worker, changed, registered)
        for total in (75 * common.GIB, 85028372479, 82 * common.GIB):
            changed = admission('fit')
            changed['qualified_hardware']['total_memory_bytes'] = total
            self.assertEqual(build.validate_qualified_hardware(changed['qualified_hardware'])['total_memory_bytes'], total)
            self.assertEqual(build.validate_fit_binding(worker, changed, registered),
                             WorkBinding.representation(('D1', 'D2')))
        for peak in (None, True, 0, 85028372480 - 8 * common.GIB + 1):
            changed = dict(admission('fit'), qualified_peak_reserved_bytes=peak)
            with self.subTest(peak=peak), self.assertRaisesRegex(ValueError, 'qualification peak'):
                build.validate_fit_binding(worker, changed, registered)
        plan = dict(self.plan, mode='fit', qualified_hardware=fabricated_hardware(),
                    qualified_gpu_product=build.QUALIFIED_GPU_PRODUCT)
        template = json.loads((RETAINED / 'gpu-job.json').read_bytes())
        terms = template['spec']['template']['spec']['affinity']['nodeAffinity'][
            'requiredDuringSchedulingIgnoredDuringExecution']['nodeSelectorTerms']
        terms[0]['matchExpressions'][0]['values'] = ['NVIDIA-H100-PCIe']
        with self.assertRaisesRegex(ValueError, 'already be allowed'):
            build.manifest_from_template(template, plan, common.encoded(plan))
        terms[0]['matchExpressions'][0]['values'] = [build.QUALIFIED_GPU_PRODUCT]
        terms.append({'matchExpressions': [{'key': 'gpu-type', 'operator': 'In', 'values': ['H100']}]})
        with self.assertRaisesRegex(ValueError, 'every affinity branch'):
            build.manifest_from_template(template, plan, common.encoded(plan))

    def test_exclusive_local_package_and_receipt_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'package'
            build.write_package(output, self.payload, self.archive, self.report)
            self.assertEqual((output / 'bundle.tar.gz').read_bytes(), self.archive)
            with self.assertRaises(ValueError):
                build.write_package(output, self.payload, self.archive, self.report)
            receipt = Path(temporary) / 'receipt.json'
            common.publish(receipt, {'fixture': True})
            with self.assertRaises(FileExistsError):
                common.publish(receipt, {'replacement': True})
            self.assertEqual(common.read_json(receipt), {'fixture': True})

    def test_host_adapters_are_python36_parseable(self):
        for name in build.ADAPTER_MEMBERS:
            with self.subTest(name=name):
                ast.parse(self.payload[name].decode(), feature_version=(3, 6))

    def test_capacity_accounts_all_reservations_not_only_one_worker(self):
        with patch.dict(sys.modules, {'common': common}):
            capacity = module(STUDY / 'deployment/capacity.py', 'frontier_test_capacity')
        parser = module(RETAINED / 'capacity_metadata.py', 'frontier_test_capacity_metadata')
        snapshot = dict(project_root=common.PROJECT, fileset='h_bcb-yding4', quota_block_size_bytes=1024,
                        quota_output='rsrch8 FILESET 0 104857600 104857600 0 none | 0 50000 50000 0 none',
                        filesystem_available_bytes=100 * common.GIB, filesystem_available_inodes=50000,
                        remote_writes=False, numerical_payloads_opened=False)
        with patch.object(capacity, 'helpers', return_value=parser):
            result = capacity.assess(snapshot, self.plan, 'a' * 64, {'uid': 'fixture'}, admission())
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['required_free_bytes'], 32 * common.GIB)
            increased = dict(admission(), aggregate_reserved_output_bytes=90 * common.GIB)
            self.assertEqual(capacity.assess(snapshot, self.plan, 'a' * 64, {}, increased)['status'], 'held')

    def test_worker_command_is_contract_bound_not_shell(self):
        with patch.dict(sys.modules, {'common': common}):
            entry = module(STUDY / 'deployment/worker_entry.py', 'frontier_test_entry')
        command = entry.command_for(self.plan, json.loads(self.payload['WORKER_CONTRACT.json']))
        self.assertEqual(command[:4], ['/opt/credo/bin/python', '-m', 'frontier.observation_worker', 'qualify'])
        self.assertEqual(command[-1], self.report['worker_contract_sha256'])


if __name__ == '__main__':
    unittest.main()
