"""Prospective C1 contract; standard-library-only, no inference or I/O on import."""
import hashlib
import json

# Numerical CUDA tests only; mocked resource/device-accounting tests do not
# establish actual GPU execution and must never be counted in this inventory.
DECLARED_CUDA_TESTS = frozenset([('test_s3b2', 'test_cuda_full_gene_shared_batched_sequential'), ('test_s3b2', 'test_cuda_full_gene_shared_gradient_dense')])

RUN_NAME = 'decoder_anchored_correction_20260920_c5'
RUN_ID = 'credo-r48-anchored-20260920-c5'
IMAGE = 'hpcharbor.mdanderson.edu/yding41/credo_env@sha256:138d7191792ce1139708751f109817a976c5b96ba3a00be3dff88173b62eb21a'
WORKER_ROOT = '/rsrch8/home/bcb/yding4/perturbseq/R48_downstream_v1/' + RUN_NAME
CONTROL_ROOT = '/rsrch8/scratch/bcb/yding4/perturbseq/R48_downstream_v1/' + RUN_NAME
SOURCES = ['D3_Rest', 'D3_Stim48hr', 'D4_Rest', 'D4_Stim48hr']
ARMS = [
    dict(arm_id='anchored_composition_control', intended_weight=0.0, trans_weight=0.0, selection_eligible=False),
    dict(arm_id='anchored_intended_diagnostic', intended_weight=0.03, trans_weight=0.0, selection_eligible=False),
    dict(arm_id='preservation_aware_intended_repair', intended_weight=0.03, trans_weight=0.03, selection_eligible=True),
]
STAGES = dict(parent='PARENT_REUSE.json', qualification='qualification/COMPLETE.json',
              statistics='statistics/COMPLETE.json', profile='PROFILE.json', exposure='FROZEN_EXPOSURE.json',
              training='readouts/COMPLETE.json', selection='SELECTION.json',
              evaluation='evaluation/COMPLETE.json', resources='RESOURCES.json')


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False) + '\n').encode('ascii')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def scientific_contract():
    return dict(schema_id='credo.c1.original_anchored_science', schema_version=1,
        source_ids=list(SOURCES), output_gene_count=18129, latent_width=48, condition_features=2,
        residual_parameters=906450, residual_output_bias=False,
        anchor='exact_original_calibration_decoder_B5_scoring_FP32_decoder_FP64_normalization',
        predictor='single_full_gene_base_logits_plus_zero_initialized_linear_residual',
        frozen_base=True, molecular_split_seed=0, role_fit=1, role_selection=2, role_audit=0,
        primary_arm='preservation_aware_intended_repair', arms=ARMS,
        training=dict(optimizer='AdamW', learning_rate=0.001, weight_decay=0.01,
            gradient_clip_norm=10.0, seed=0, full_gradient_updates=200, snapshot_updates=[25,50,100,200],
            anchor_penalty=1.0, anchor_KL='KL_original_to_corrected',
            composition_weighting='equal_cell', anchor_KL_weighting='equal_source_of_equal_cell',
            effect_weighting='equal_source_equal_record',
            minimum_loss_scale=1e-12, chunk_cells=512,
            anchor_batch_order='role_cache_order_512_cells_interleaved_valid_molecular_directions',
            scales='role1_original_CE_and_observed_intended_trans_squared_effect_energy',
            effect_precision='FP64_differentiable_cell_probability_average_no_FP32_rounding',
            all_arms_identical_exposure=True, profiling_state_discarded=True,
            encoder_updates=0, no_dense_full_cohort_probability_cache=True, resume_supported=False),
        support=dict(panel='exact_B1_inner_fit_membership_only',
            role_statistics='independent_role1_role2', minimum_guides=2,
            minimum_guide_half_cells=5, minimum_NTC_half_cells=20,
            half_rule='B1_supported_trans_seed0_source_cell_sha256',
            no_guide_drop_or_outcome_adaptive_resplit=True, unavailable_not_zero=True,
            all_four_sources_required=True),
        selection=dict(reference='original_calibration_decoder_same_population',
            CE_weightings=['equal_cell','equal_source'], intended_weightings=['equal_record','equal_source'],
            trans_weightings=['equal_record','equal_source'], CE_delta_maximum=0.,
            intended_strict_decrease=True, supported_trans_delta_maximum=0.,
            metric_precision='FP32_after_FP64_valid_molecular_average_then_FP64_groups',
            record_first=True, disjoint_cell_half_directions=2,
            rank=['intended_equal_source','intended_equal_record','CE_equal_source','update','sha256'],
            maximum_primary_candidates=4, null_status='no_feasible_anchored_repair',
            no_postselection_refit=True, no_audit_reranking=True, numerical_tolerance_as_scientific_slack=False,
            all_trans_additional_veto=False, individual_source_additional_veto=False),
        qualification=dict(prefit_role0_numerical_replay_exception=True,
            allowed_models=['original_autoencoder','B1_lambda0p03_u200','zero_residual','deterministic_nonzero_fixture'],
            candidate_performance_exposed=False, sample_selection='metadata_only_frozen_before_inference',
            maximum_cells=2048, per_guide_half_cells=32, NTC_per_source_half_cells=64,
            minimum_guide_half_cells=1, minimum_NTC_half_cells=20,
            numerical_replay_only_not_scientific_support=True,
            flag_guide_halves_below_scientific_minimum=5,
            diagnostics=['ENSG00000160551','ENSG00000119004','ENSG00000164305'],
            tolerances=dict(states=dict(atol=2e-5,rtol=2e-5),logits=dict(atol=2e-5,rtol=2e-5),
                probabilities=dict(atol=1e-7,rtol=2e-5),metrics=dict(atol=1e-6,rtol=1e-6)),
            zero_correction_same_cached_input_exact=True, no_tolerance_widening=True,
            attribution=dict(required=True, models='original_and_deterministic_nonzero_fixture_only',
                chunk_cells=32, maximum_cells=2048, maximum_output_bytes=3_500_000_000,
                base_logits_dtype='FP32', residual_dtype='FP64', q0_dtype='FP64', qtheta_dtype='FP64',
                retain_full_gene_molecular_arrays_and_treatment_control_means=True,
                log_probability_identity_absolute_tolerance=1e-12, causal_B1_attribution_claim=False)),
        evaluation=dict(cells=19598, original_records=90, encoded_records=87,
            cell_split_seeds=[0,1,2,3,4],
            predictors='three_terminal_arms_original_and_selected_if_nonterminal_no_reranking',
            same_original_relative_six_constraints=True, known_development_not_independent_test=True,
            mandatory=['intended','supported_trans','all_trans','all_genes','zero_effect',
                'observed_repeatability','per_source','known_failures','error_concentration']),
        resource=dict(gpus=3, gpus_per_worker=1, GPU='H100', CPU=24, host_memory='200Gi',
            torch_memory_ceiling_bytes=60_000_000_000, output_byte_ceiling=15_000_000_000,
            whole_device_sample_seconds=0.5, process_RSS_sample_seconds=0.5,
            timeseries='bounded_exclusive_JSONL_chunks_compact_metadata_no_duration_cap',
            runtime_deadline_policy='no_runtime_deadline_user_20260919', automatic_retries=0),
        protected_D1_D2_access=False, dynamics_training=False, biological_promotion=False,
        old_experiments_retained_unchanged=True,
        governing_attachment_sha256='684e114af8efd130d668fe84443ee591b2fbdd4a7dd6692e80c77fb44b340695')


def authorization():
    return dict(schema_id='credo.c1.user_authorization', schema_version=1,
        request='use several H100 to parallelize training; yes to replace serial run with three H100s',
        scope='C4_disjoint_arm_parallelization_restart_fits_reuse_verified_C3_statistics',
        user_approved=True, approved_image=IMAGE, approved_PVC='yding4-gpu-home', approved_subpath='perturbseq',
        image_and_mount_reuse='previously_explicitly_approved_unchanged',
        GPU_ceiling=3, GPUs_per_worker=1, scheduler='kubernetes', CPU=24, host_memory='200Gi',
        runtime_deadline_policy='no_runtime_deadline_user_20260919', automatic_retries=0,
        parent_retained_unchanged=True, no_D1_D2_access=True, no_encoder_updates=True,
        biological_promotion=False, prefit_role0_numerical_replay_exception=True,
        worker_visibility_exception='existing_approved_home_PVC_perturbseq_C1_subdirectory',
        source_repositories_unchanged=True)


def validate_payload(plan, science_raw, authorization_raw):
    from runtime_policy import validate_plan
    validate_plan(plan)
    science, authority = json.loads(science_raw), json.loads(authorization_raw)
    if canonical(science) != canonical(scientific_contract()) or canonical(authority) != canonical(authorization()):
        raise ValueError('Exact C1 scientific/current-user contract required')
    if (plan.get('run_id') != RUN_ID or plan.get('control_root') != CONTROL_ROOT
            or plan.get('bundle_root') != WORKER_ROOT+'/bundle'
            or plan.get('worker_output') != WORKER_ROOT+'/output' or plan.get('image') != IMAGE
            or plan.get('c1_specification_sha256') != sha(science_raw)
            or plan.get('authorization_sha256') != sha(authorization_raw)
            or plan.get('completion_stage_files') != STAGES
            or plan.get('gpu_ceiling') != 3 or plan.get('expected_optimizer_updates') != 600
            or plan.get('arm_index_map') != [a['arm_id'] for a in ARMS]
            or plan.get('aggregator_index') != 2
            or plan.get('expected_trajectory_snapshots') != 12):
        raise ValueError('C1 execution/science/authority binding mismatch')
    return science, authority


def arm_output(plan, index):
    if type(index) is not int or index not in (0, 1, 2):
        raise ValueError('Exactly one of the three fixed arm indexes required')
    return plan['worker_output'] if index == 2 else plan['worker_output'] + '-arm-' + str(index)
