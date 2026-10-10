"""C1 application validator; stdlib/host Python 3.6, no automatic retry."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
import xml.etree.ElementTree as ET
from runtime_policy import validate_plan as validate_runtime_policy, validate_job as validate_runtime_job
import c1_contract

BASE = Path(__file__).resolve().parent
CORE_SHA = '3da1a0a5b0d0d26019918ee48e2ce24cd2e616272de7854f4c82ba2e37e7ebd2'
WRITER_UID = 307469
STAGE_FILES = c1_contract.STAGES


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_uid)


def checked_file(path, retain=False):
    path = Path(path).absolute()
    for parent in path.parents:
        require(stat.S_ISDIR(parent.lstat().st_mode), 'Non-directory or symlink artifact ancestor')
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_uid == WRITER_UID, 'Owned regular result artifact required')
    require(not retain or before.st_size <= 64*1024*1024, 'JSON metadata exceeds parser budget')
    digest, pieces = hashlib.sha256(), []
    with os.fdopen(os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        require(_signature(os.fstat(stream.fileno())) == _signature(before), 'Artifact changed at open')
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
            if retain:
                pieces.append(block)
        require(_signature(os.fstat(stream.fileno())) == _signature(before)
                and _signature(path.lstat()) == _signature(before), 'Artifact changed during verification')
    return dict(sha256=digest.hexdigest(), size_bytes=before.st_size), b''.join(pieces) if retain else None


def read_json(path):
    binding, raw = checked_file(path, retain=True)
    return json.loads(raw.decode('utf-8')), binding


def _sha(path):
    return checked_file(path)[0]['sha256']


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def load_core():
    path = BASE/'lifecycle_core.py'
    require(hashlib.sha256(path.read_bytes()).hexdigest() == CORE_SHA, 'Immutable lifecycle source changed')
    spec = importlib.util.spec_from_file_location('c1_lifecycle_core', str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = load_core() if (BASE/'lifecycle_core.py').is_file() else None


def validate_static(plan):
    root = Path(plan['control_root'])/'static'
    terminal, _ = read_json(root/'terminal.json')
    cleanup, _ = read_json(root/'cleanup.json')
    reads = cleanup.get('reads', [])
    require(terminal.get('status') == 'passed' and cleanup.get('status') == 'complete'
        and terminal.get('uid') and terminal.get('uid') == cleanup.get('uid')
        and terminal.get('job') == cleanup.get('job')
        and terminal.get('plan_sha256') == _sha(BASE/'PLAN.json')
        and terminal.get('manifest_sha256') == _sha(BASE/'static-job.json')
        and terminal.get('gpu_requests') == 0 and terminal.get('gpu_limits') == 0
        and terminal.get('visibility_mask') == 'none' and terminal.get('unmasked_visibility_claimed') is False
        and len(reads) >= 2 and all(row == dict(job_absent=True, owned_pods=0) for row in reads[-2:]),
        'Exact completed/cleaned masked zero-GPU static gate required')
    manifest, _ = read_json(BASE/'static-job.json')
    require(terminal.get('job') == manifest['metadata']['name'], 'Static Job name differs')


def _relative(value):
    require(isinstance(value, str), 'Result path must be a string')
    relative = Path(value)
    require(not relative.is_absolute() and '..' not in relative.parts and relative.parts
        and relative.as_posix() == value and value != 'COMPLETE.json', 'Unsafe result path')
    return relative


def validate_inventory(output, result, maximum):
    output = Path(output)
    require(not output.is_symlink() and output.is_dir() and output.stat().st_uid == WRITER_UID,
            'Owned regular output directory required')
    require(isinstance(result.get('files'), list), 'Complete result inventory list required')
    inventory, total = {}, 0
    for item in result['files']:
        relative = _relative(item['path'])
        require(item['path'] not in inventory, 'Duplicate result path')
        observed, _ = checked_file(output/relative)
        require(observed == dict(sha256=item.get('sha256'), size_bytes=item.get('size_bytes')), 'Result bytes differ')
        inventory[item['path']] = observed
        total += observed['size_bytes']
    actual = set()
    for path in output.rglob('*'):
        mode = path.lstat().st_mode
        require(stat.S_ISREG(mode) or stat.S_ISDIR(mode), 'Unexpected special output entry')
        if stat.S_ISREG(mode):
            actual.add(path.relative_to(output).as_posix())
    require(actual == set(inventory) | {'COMPLETE.json'} and result.get('output_bytes') == total
            and total+(output/'COMPLETE.json').stat().st_size <= maximum, 'Whole output inventory/ceiling differs')
    return inventory


def _nested_inventory(stage, prefix, inventory):
    require(isinstance(stage.get('files'), dict), 'Stage complete artifact inventory required')
    expected = set()
    for relative, binding in stage['files'].items():
        name = (Path(prefix)/_relative(relative)).as_posix()
        require(inventory.get(name) == binding, 'Nested stage artifact binding differs')
        expected.add(name)
    require({name for name in inventory if name.startswith(prefix+'/')} == expected | {prefix+'/COMPLETE.json'},
            'Nested stage inventory coverage differs')


def validate_tests(output, inventory, expected_tests, expected_cuda):
    qualification, binding = read_json(output/'TEST_QUALIFICATION.json')
    xml_binding, raw = checked_file(output/'tests.xml', retain=True)
    require(inventory.get('TEST_QUALIFICATION.json') == binding and inventory.get('tests.xml') == xml_binding
        and qualification.get('status') == 'passed' and qualification.get('totals') == expected_tests
        and qualification.get('real_data_opened') is False and qualification.get('junit_sha256') == xml_binding['sha256']
        and qualification.get('actual_cuda_tests') == expected_cuda and expected_cuda >= 1,
        'Bound actual-CUDA test qualification differs')
    document = ET.fromstring(raw)
    suites = [document] if document.tag == 'testsuite' else document.findall('testsuite')
    totals = {key: sum(int(suite.get(key, 0)) for suite in suites) for key in expected_tests}
    cases = list(document.iter('testcase'))
    identities = [(case.get('classname','').rsplit('.',1)[-1],case.get('name','')) for case in cases]
    actual_cuda = [item for item in identities if item in c1_contract.DECLARED_CUDA_TESTS]
    require(totals == expected_tests and len(cases) == expected_tests['tests'] and len(actual_cuda) == expected_cuda
        and len(identities) == len(set(identities)) and set(actual_cuda) == c1_contract.DECLARED_CUDA_TESTS
        and not any(any(case.find(tag) is not None for tag in ('skipped', 'failure', 'error')) for case in cases),
        'JUnit must prove every declared test and actual-CUDA check')


def validate_scientific_gate(gate, selected=True):
    require(isinstance(gate, dict) and gate.get('biological_promotion') is False
        and gate.get('statistical_test') is False and type(gate.get('passed')) is bool
        and isinstance(gate.get('failures'), list) and isinstance(gate.get('comparisons'), list),
        'C1 scientific gate absent or overclaimed')
    if not selected:
        require(gate.get('status') == 'no_selected_anchored_repair' and not gate['passed']
                and gate['failures'] == ['candidate_absent'] and gate['comparisons'] == [],
                'Absent model is not six measured failures')
        return
    expected = {(block, weight) for block, weights in
        [('composition_ce', ['equal_cell', 'equal_source']), ('intended', ['equal_record', 'equal_source']),
         ('supported_trans', ['equal_record', 'equal_source'])] for weight in weights}
    seen, failures = set(), []
    for item in gate['comparisons']:
        key = (item.get('block'), item.get('weighting'))
        require(key in expected and key not in seen and type(item.get('available')) is bool
                and type(item.get('passed')) is bool, 'C1 six-comparison inventory differs')
        seen.add(key)
        if item['available']:
            a, b = item.get('candidate'), item.get('reference')
            require(_finite(a) and _finite(b) and min(a, b) >= 0 and item.get('absolute_delta') == a-b,
                    'C1 observed comparison delta differs')
            passed = a < b if key[0] == 'intended' else a <= b
            reason = 'passed' if passed else 'criterion_failed'
        else:
            passed, reason = False, 'unavailable_or_population_mismatch'
            require(item.get('absolute_delta') is None, 'Unavailable C1 comparison cannot report a delta')
        require(item['passed'] == passed and item.get('reason') == reason, 'C1 strict criterion differs')
        if not passed:
            failures.append(':'.join(key)+':'+reason)
    allowed_extra = {'source_coverage_or_identity_mismatch', 'candidate_not_selection_eligible',
                     'unchanged_zero_correction_is_not_a_repair'}
    require(seen == expected and gate.get('scientific_tolerance') == 0.
        and gate['passed'] == (not gate['failures'])
        and gate.get('status') == ('strict_anchored_preservation_pass' if gate['passed'] else 'strict_anchored_preservation_not_passed')
        and set(failures).issubset(gate['failures']) and set(gate['failures'])-set(failures) <= allowed_extra,
        'C1 strict gate status/failure reconciliation differs')


def validate_comparison_sources(gate, candidate, original):
    """Availability itself is recomputed, not trusted from the gate receipt."""
    if candidate is None:
        validate_scientific_gate(gate, selected=False)
        return
    source_match = (isinstance(original,dict) and candidate.get('source_ids') == original.get('source_ids')
        and candidate.get('source_ids') == c1_contract.SOURCES
        and candidate.get('source_coverage') == original.get('source_coverage')
        and all(candidate.get('source_coverage',{}).get(source,0)>0 for source in c1_contract.SOURCES))
    for row in gate['comparisons']:
        block,weight = row['block'],row['weighting']
        a,b = candidate.get(block,{}),(original or {}).get(block,{})
        av,bv = a.get(weight),b.get(weight)
        available = bool(source_match and a.get('support_identity') and a.get('support_identity') == b.get('support_identity')
            and _finite(av) and _finite(bv) and av >= 0 and bv >= 0)
        require(row.get('candidate') == av and row.get('reference') == bv and row.get('available') == available,
                'C1 scientific gate population/metric source differs')


def validate_resource_timeseries(resource, inventory, prefix=''):
    series=resource.get('timeseries',{})
    require(isinstance(series,dict) and series.get('storage') == 'exclusive_bounded_jsonl_chunks'
        and series.get('directory') == 'resource_samples' and series.get('complete') is True
        and series.get('maximum_chunk_rows') == 4096 and series.get('maximum_chunk_bytes') == 1024*1024
        and series.get('sample_count') == resource.get('sample_count') and resource.get('error_count') == 0
        and resource.get('samples_scope') == 'last_16_only' and len(resource.get('samples',[])) <= 16
        and isinstance(series.get('files'),dict) and series['files']
        and series.get('chunk_count') == len(series['files']), 'C1 bounded resource timeseries incomplete')
    parent=(prefix+'/' if prefix else '')+'resource_samples/'
    cursor=0
    for index,(name,item) in enumerate(sorted(series['files'].items())):
        require(name == 'samples-%08d.jsonl' % index and type(item.get('samples')) is int
            and 0 < item['samples'] <= 4096 and 0 < item.get('size_bytes',0) <= 1024*1024
            and item.get('first_sample_index') == cursor
            and item.get('last_sample_index') == cursor+item['samples']-1
            and inventory.get(parent+name) == dict(sha256=item.get('sha256'),size_bytes=item.get('size_bytes')),
            'C1 resource chunk byte/row identity differs')
        cursor += item['samples']
    require(cursor == series['sample_count']
        and {name for name in inventory if name.startswith(parent)} == {parent+name for name in series['files']},
        'C1 resource timeseries inventory/count coverage differs')


def validate_parent(plan, bundle):
    _, science_raw = checked_file(bundle/'C1_SPECIFICATION.json', retain=True)
    _, authorization_raw = checked_file(bundle/'C1_AUTHORIZATION.json', retain=True)
    c1_contract.validate_payload(plan, science_raw, authorization_raw)
    authority, binding = read_json(bundle/'C1_PARENT_BINDING.json')
    require(binding['sha256'] == plan.get('parent_binding_sha256') and authority.get('consumer_run_id') == plan['run_id']
        and authority.get('parent_retained_unchanged') is True
        and authority.get('schema_id') == 'credo.c1.readonly_parent_authority'
        and authority.get('C1_role1_and_role2_statistics_recomputed') is True
        and authority.get('B1_training_and_selection_not_repeated') is True
        and isinstance(authority.get('files'), dict) and authority['files'], 'C1 parent authority differs')
    prefix = '/rsrch8/home/bcb/yding4/perturbseq/R48_downstream_v1/full_gene_objective_tradeoff_20260918_b1/'
    for name, expected in authority['files'].items():
        require(name.startswith(prefix) and '..' not in Path(name).parts, 'C1 foreign parent artifact')
        require(checked_file(Path(name))[0] == expected, 'Retained parent bytes changed')
    for name, expected in authority.get('completed_B5_evidence', {}).items():
        allowed = ['/rsrch8/'+area+'/bcb/yding4/perturbseq/R48_downstream_v1/full_gene_objective_tradeoff_20260919_b5/'
                   for area in ['home', 'scratch']]
        require(any(name.startswith(prefix) for prefix in allowed) and '..' not in Path(name).parts
                and checked_file(Path(name))[0] == expected, 'Retained B5 completion evidence changed')
    return authority, binding


def validate_training(stages, inventory):
    stats, profile, exposure, trained, seal = [stages[key] for key in ['statistics','profile','exposure','training','selection']]
    require(stats.get('schema') == 'credo.c1.anchored_role_statistics.v1'
        and stats.get('dense_full_cohort_gene_cache') is False and set(stats.get('roles', {})) == {'1','2'},
        'C1 independent role statistics missing')
    for role in ('1','2'):
        item = stats['roles'][role]
        require(item.get('population', {}).get('role') == int(role) and item.get('statistics', {}).get('role') == int(role)
            and item['statistics'].get('genes') == 18129
            and item['statistics'].get('source_ids') == c1_contract.SOURCES
            and item['population'].get('source_ids') == c1_contract.SOURCES, 'C1 role/source/gene identity differs')
    require(profile.get('schema') == 'credo.c1.discarded_training_profile.v1' and profile.get('role') == 1
        and profile.get('complete_updates') == 1 and profile.get('profile_weights_discarded') is True
        and _finite(profile.get('elapsed_seconds')) and profile['elapsed_seconds'] > 0
        and _finite(profile.get('whole_device_peak_bytes')) and profile['whole_device_peak_bytes'] > 0,
        'C1 discarded complete-update resource profile differs')
    training = trained.get('training_contract', {})
    settings = dict(optimizer='AdamW',learning_rate=.001,weight_decay=.01,gradient_clip_norm=10.,seed=0,
        full_gradient_updates=200,snapshot_updates=[25,50,100,200],anchor_penalty_weight=1.,
        post_selection_refit=False,resume_policy='fail_and_stop',runtime_deadline_seconds=None)
    require(all(training.get(k) == v for k,v in settings.items()) and exposure.get('training') == training
        and exposure.get('status') == 'frozen_before_scientific_fitting' and exposure.get('evaluation_outcomes_accessed') is False
        and exposure.get('profile_receipt_sha256') == inventory[STAGE_FILES['profile']]['sha256']
        and trained.get('schema') == 'credo.c1.anchored_training.v1' and trained.get('status') == 'complete'
        and trained.get('trial_count') == 3 and trained.get('model_training_updates') == 600
        and trained.get('snapshot_count') == 12 and trained.get('selection_performed') is False
        and all(trained.get(k) == 0 for k in ['audit_cells_used','encoder_updates','original_decoder_updates'])
        and trained.get('original_anchor_frozen') is True and trained.get('no_post_selection_refit') is True
        and trained.get('training_contract_sha256') == identity(training)
        and trained.get('input_bindings_sha256') == identity(stats.get('input_bindings')),
        'C1 training/exposure boundary differs')
    snapshots, arms = trained.get('snapshots', {}), {arm['arm_id'] for arm in c1_contract.ARMS}
    require(len(snapshots) == 12 and {(r.get('arm_id'),r.get('update')) for r in snapshots.values()} ==
        {(arm,u) for arm in arms for u in [25,50,100,200]} and len(trained.get('history', [])) == 600
        and {(r.get('arm_id'),r.get('update')) for r in trained['history']} ==
        {(arm,u) for arm in arms for u in range(1,201)}, 'C1 complete matched-exposure family required')
    for name,item in snapshots.items():
        require(name == item.get('snapshot_id')
            and inventory.get('readouts/'+_relative(item['path']).as_posix()) == dict(sha256=item.get('sha256'),size_bytes=item.get('size_bytes'))
            and item.get('parameter_phase') == 'post_optimizer_step' and item.get('inner_selection',{}).get('role') == 2
            and item.get('primary_selection_eligible') == (item.get('arm_id') == 'preservation_aware_intended_repair'),
            'C1 checkpoint byte/phase/eligibility binding differs')
    require(seal.get('schema') == 'credo.c1.original_relative_selection.v1'
        and seal.get('registry_identity') == identity(sorted(snapshots.values(),key=lambda r:r['snapshot_id']))
        and seal.get('seal_identity') == identity({k:v for k,v in seal.items() if k != 'seal_identity'})
        and seal.get('checkpoint_bytes_verified') is True and seal.get('postselection_refit') is False
        and seal.get('candidate_audit_performance_access_before_seal') is False
        and seal.get('prefit_role0_access') == 'frozen_numerical_replay_only'
        and seal.get('maximum_primary_candidates') == 4 and seal.get('reference',{}).get('role') == 2,
        'C1 independent role2 selection seal differs')
    selection, feasible = seal.get('selection',{}), []
    primary = {name:item for name,item in snapshots.items() if item['arm_id'] == 'preservation_aware_intended_repair'}
    require(set(selection.get('gates',{})) == set(primary), 'C1 primary-only selection gate inventory differs')
    for name,item in primary.items():
        gate = selection['gates'][name]
        validate_scientific_gate(gate)
        validate_comparison_sources(gate,item['inner_selection'],seal['reference'])
        unchanged = item.get('residual_state_sha256') == item.get('initial_state_sha256')
        require(('unchanged_zero_correction_is_not_a_repair' in gate['failures']) == unchanged,
                'Unchanged zero correction cannot be selected as a repair')
        if gate['passed']:
            feasible.append(item)
    selected = min(feasible,key=lambda r:(r['inner_selection']['intended']['equal_source'],
        r['inner_selection']['intended']['equal_record'],r['inner_selection']['composition_ce']['equal_source'],
        r['update'],r['sha256'])) if feasible else None
    require(selection.get('candidate') == selected and selection.get('eligible_primary_snapshots') == 4
        and selection.get('feasible_primary_snapshots') == len(feasible)
        and selection.get('status') == ('selected_anchored_repair' if selected else 'no_feasible_anchored_repair')
        and selection.get('promotion_allowed') is False, 'C1 constrained selection differs')
    return selected


def validate_result(plan,binding,root,owned):
    if plan.get('arm_index_map'):
        from c4_lifecycle import validate_peers
        owned = [validate_peers(sys.modules[__name__],plan,binding,root,owned)]
    output = Path(plan['worker_output'])
    result,complete_binding = read_json(output/'COMPLETE.json')
    tests = dict(tests=plan['expected_gpu_tests'],failures=0,errors=0,skipped=0)
    require(result.get('status') == 'completed_original_anchored_experiment' and result.get('run_id') == plan['run_id']
        and result.get('plan_sha256') == _sha(BASE/'PLAN.json') and result.get('implementation_sha256') == plan['implementation_sha256']
        and result.get('optimizer_updates') == 600 and result.get('trajectory_snapshots') == 12
        and result.get('fresh_configurations') == 3 and result.get('evaluation_only') is False
        and result.get('encoder_optimizer_updates') == 0 and result.get('original_decoder_optimizer_updates') == 0
        and all(result.get(k) is False for k in ['D1_endpoint_opened','D2_endpoint_opened','refit_latents_opened','scientific_promotion','post_selection_refit'])
        and len(owned) == 1 and result.get('pod_uid') == owned[0]['metadata']['uid'] and result.get('test_totals') == tests,
        'C1 root completion boundary differs')
    inventory = validate_inventory(output,result,plan['maximum_new_output_bytes'])
    validate_tests(output,inventory,tests,plan['expected_cuda_tests'])
    require(plan.get('completion_stage_files') == STAGE_FILES and set(result.get('stages',{})) == set(STAGE_FILES),
        'Exact C1 stage registry required')
    stages = {}
    for name,relative in STAGE_FILES.items():
        stages[name],observed = read_json(output/relative)
        require(inventory.get(relative) == observed and result['stages'][name] == observed['sha256'],'C1 stage hash differs')
    parent,parent_binding = validate_parent(plan,BASE)
    reuse = stages['parent']
    require(reuse.get('status') == 'verified_read_only_parent' and reuse.get('parent_binding_sha256') == parent_binding['sha256']
        and reuse.get('parent_retained_unchanged') is True
        and reuse.get('original_B1_selection_identity') == parent['historical_B1_selection_identity'],'C1 parent reuse differs')
    startup,observed = read_json(output/'STARTUP.json')
    release,_ = read_json(root/'startup-validated.json')
    gpu = startup.get('gpu',{})
    require(result.get('startup_sha256') == observed['sha256'] and result.get('junit_sha256') == inventory.get('tests.xml',{}).get('sha256')
        and startup.get('status') == 'startup_guard_passed' and startup.get('pod_uid') == result['pod_uid']
        and startup.get('run_id') == plan['run_id'] and startup.get('plan_sha256') == result['plan_sha256']
        and startup.get('implementation_sha256') == plan['implementation_sha256']
        and gpu.get('count') == 1 and 'H100' in gpu.get('name','') and 'MIG' not in gpu.get('name','')
        and gpu.get('total_memory_bytes',0) >= 75e9 and gpu.get('compute_capability') == '9.0'
        and release.get('pod_uid') == result['pod_uid'] and release.get('uid') == binding['uid']
        and release.get('startup_sha256') == result['startup_sha256'] and release.get('plan_sha256') == result['plan_sha256']
        and release.get('actual_pod_audit_passed') is True and release.get('tests_permitted') is True,'Exact C1 H100 startup release differs')
    for stage,prefix in [('qualification','qualification'),('statistics','statistics'),('training','readouts'),('evaluation','evaluation')]:
        _nested_inventory(stages[stage],prefix,inventory)
    qualification = stages['qualification']
    require(qualification.get('status') == 'passed' and 0 < qualification.get('sample_cells',0) <= 2048
        and qualification.get('optimizer_updates') == 0 and qualification.get('biological_acceptance') is False
        and qualification.get('candidate_ranking') is False and qualification.get('checks',0) > 0,'C1 loaded qualification missing')
    replay,_ = read_json(output/'qualification/RAW_TO_METRIC_REPLAY.json')
    zero,_ = read_json(output/'qualification/ZERO_CORRECTION_PARITY.json')
    resource_check,_ = read_json(output/'qualification/RESOURCE_PROFILE.json')
    validate_resource_timeseries(resource_check,inventory,prefix='qualification')
    require(replay.get('passed') is True and replay.get('checks') and all(c.get('passed') is True for c in replay['checks'])
        and zero.get('passed') is True and zero.get('identical_cached_states') is True and resource_check.get('valid') is True
        and resource_check.get('whole_device_sampled_peak_bytes',0) > 0,'C1 replay/zero/resource evidence differs')
    attribution,attribution_binding=read_json(output/'qualification/attribution/COMPLETE.json')
    require(qualification.get('attribution_complete') == 'attribution/COMPLETE.json'
        and qualification.get('attribution_complete_sha256') == attribution_binding['sha256']
        and attribution.get('status') == 'passed_bounded_fixed_fixture_attribution',
        'C1 fixed-fixture attribution evidence differs')
    _nested_inventory(attribution,'qualification/attribution',inventory)
    selected = validate_training(stages,inventory)
    evaluation,seal = stages['evaluation'],stages['selection']
    require(evaluation.get('status') == 'completed_original_anchored_development_evaluation'
        and evaluation.get('selection_identity') == seal['seal_identity'] and evaluation.get('original_cells') == 19598
        and evaluation.get('original_records') == 90 and evaluation.get('encoded_intended_records') == 87
        and evaluation.get('no_audit_reranking') is True and all(evaluation.get(k) is False for k in
        ['scientific_promotion','D1_endpoint_opened','D2_endpoint_opened','post_selection_refit']),'C1 evaluation boundary differs')
    gate = evaluation.get('gate',{})
    validate_scientific_gate(gate,selected=selected is not None)
    reported_gate,_ = read_json(output/'evaluation/DEVELOPMENT_GATE.json')
    registry,_ = read_json(output/'evaluation/PREDICTOR_REGISTRY.json')
    metrics,_ = read_json(output/'evaluation/METRICS.json')
    validate_comparison_sources(gate,metrics[selected['snapshot_id']] if selected is not None else None,
                                metrics['original_autoencoder'])
    require(gate == result.get('scientific_gate') and gate == reported_gate and registry == evaluation.get('predictor_registry')
        and isinstance(registry,list) and len(registry) in (4,5),'C1 gate/predictor registry mismatch')
    expected = {name for name,row in stages['training']['snapshots'].items() if row['update'] == 200}
    if selected is not None:
        expected.add(selected['snapshot_id'])
    require({row.get('predictor') for row in registry} == expected | {'original_autoencoder'},'C1 predictor inventory differs')
    resources = stages['resources']
    validate_resource_timeseries(resources,inventory)
    require(resources.get('valid') is True and resources.get('errors') == [] and resources.get('sample_interval_seconds') == .5
        and resources.get('sample_count',0) >= 2 and resources.get('whole_device_sampled_peak_bytes',0) > 0
        and resources.get('torch_peak_allocated_bytes',0) > 0 and resources.get('own_process_sampled_peak_rss_bytes',0) > 0
        and resources.get('elapsed_time_deadline') is None,'C1 measured resource evidence missing')
    require(checked_file(output/'COMPLETE.json')[0] == complete_binding,'C1 completion changed during validation')
    return True


def monitor(plan,binding,root,static=False):
    require(not static,'Neutral static qualification has a separate controller')
    validate_runtime_policy(plan)
    while True:
        current = core.get_job(binding['job'])
        validate_runtime_job(current)
        core.verify(current,binding,False)
        passed,failed = core.complete(current)
        indexed = bool(plan.get('arm_index_map'))
        owned_now = core.pods(binding['job'],binding['uid']) if indexed else None
        if indexed:
            failed = failed or any(p.get('status',{}).get('phase') == 'Failed' for p in owned_now)
            failed = failed or any((Path(c1_contract.arm_output(plan,i))/'FAILED.json').exists() for i in range(3))
        if (indexed or not (root/'startup-validated.json').exists()) and not failed:
            try:
                if indexed:
                    from c4_lifecycle import release_startups
                    release_startups(core,plan,binding,root,current,owned_now)
                else:
                    core.release_startup(plan,binding,root,current)
            except Exception as error:
                core.publish(root/'startup-failure.json',dict(error=repr(error),uid=binding['uid']))
                core.capture(current,root);core.cleanup(binding,root)
                raise
        if passed or failed:
            owned = core.capture(current,root)
            application,reason = False,None
            try:
                application = bool(passed and not failed and validate_result(plan,binding,root,owned))
            except Exception as error:
                reason = repr(error)
            core.publish(root/'terminal.json',dict(status='passed' if application else 'failed',job=binding['job'],uid=binding['uid'],
                run_id=plan['run_id'],kubernetes_complete=passed,application_pass=application,reason=reason,scientific_promotion=False,
                plan_sha256=core.sha(BASE/'PLAN.json'),pod_uid=owned[0]['metadata']['uid'] if len(owned)==1 else None,
                result_complete_sha256=core.sha(Path(plan['worker_output'])/'COMPLETE.json') if application else None))
            core.cleanup(binding,root)
            require(application,'C1 failed; evidence retained, no automatic retry')
            return
        time.sleep(30)


def main():
    require(core is not None,'Run only the built immutable lifecycle bundle')
    execution,_ = read_json(BASE/'PLAN.json')
    _,science = checked_file(BASE/'C1_SPECIFICATION.json',retain=True)
    _,authority = checked_file(BASE/'C1_AUTHORIZATION.json',retain=True)
    c1_contract.validate_payload(execution,science,authority)
    if len(sys.argv)>1 and sys.argv[1] in ('create','launch','supervise'):
        validate_static(execution)
    if len(sys.argv)>1 and sys.argv[1] in ('launch','supervise'):
        from c4_lifecycle import validate_stop
        validate_stop(execution,checked_file)
    core.monitor = monitor
    core.main()


if __name__ == '__main__':
    main()
