"""Small fabricated native files -> real preparation/normalizer publications.

No training, optimizer, CUDA, biological input or forecast is performed here.
The observation export is explicitly untrained schema-fixture material, not a
scientific selected model. Its nominal update fields exercise the retained
loader; a separate marker and identity flag preserve that distinction.
"""

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from .cache_compiler import CompilerLimits
from .fitting_inputs import FittingInputContract, SCHEMA as INPUT_SCHEMA
from .fitting_preparation import FittingPopulation, PreparationLimits, prepare_fitting_cache
from .fitting_provider import ProviderLimits
from .geometry import GeometryOptions, RaggedSinkhornGeometry, geomloss_backend
from .main_qualification_worker_plan import cases, worker_schedule
from .native_inputs import FoldInputs, NativeInputIndex, PackageBinding
from .normalizer_publication import NormalizationLimits, publish_normalizers
from .observation import ObservationFold, SparseR48
from .observation_loader import ObservationArtifactAdmission, ObservationFile, _path, _sha
from .observation_training import ObservationTrainingSpec, _publish_tensor
from .planner import DONORS, TrajectoryKey, screen_plan, family_plan, finalist_plan, outer_plan
from .retention import STUDY_ID
from .training_objective import ObjectiveExecution


MIB = 1024**2
STUDY_LABEL = "fabricated-worker-qualification"
STUDY_SEED = 731


def file_reference(path):
    path = _path(path)
    raw = path.read_bytes()
    return ObservationFile(str(path), hashlib.sha256(raw).hexdigest(), len(raw))


def write_json(path, value):
    path = _path(path)
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(raw) > 4*MIB:
        raise ValueError("Fixture metadata exceeds four MiB")
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return file_reference(path)


def stage_task(case):
    if case not in cases():
        raise ValueError("Fixed qualification worker case required")
    first = screen_plan()
    families = family_plan({d: tuple(f"F{i}-V2" for i in range(6)) for d in DONORS})
    finalists = finalist_plan(families, {d: ("F0-V2", "F1-V2", "F5-V2") for d in DONORS})
    plan, prior = {"screen": (first, None), "family": (families, first),
                   "outer": (outer_plan(finalists), finalists)}[case.stage]
    task = next(t for t in plan.training if t.key == TrajectoryKey(case.training_donors, case.configuration_id, 0))
    return plan, prior, task


def _native(root, reader_source_root):
    """Generate exactly three guides/genes and three cells per donor/time."""
    data, metadata = root / "data", root / "metadata"
    (data / "source").mkdir(parents=True)
    (metadata / "source").mkdir(parents=True)
    pd.DataFrame(dict(sgRNA=["g0", "g1", "g2"], primary_target_id=["T0", "NTC", "T2"],
        is_design_ntc=[False, True, False], experimental_library_member=[True]*3)).to_parquet(
            data / "source/guides.parquet", index=False)
    samples, features, shards = [], [], []
    for donor in DONORS:
        for condition in ("Stim8hr", "Stim48hr"):
            source = donor + "_" + condition
            samples.append(dict(source_id=source, donor_id=donor, condition=condition))
            for i, gene in enumerate(("PuroR", "ENSG1", "ENSG2", "ENSG3")):
                features.append(dict(source_id=source, source_position=3-i,
                    canonical_position=i, feature_id=gene, is_RNA=i>0))
            rows = ([[9,0,1,2], [7,2,0,1], [2,2,1,0]] if condition == "Stim48hr"
                    else [[9,1,0,2], [7,2,1,0], [2,0,3,0]])
            counts = sparse.csr_matrix(np.asarray(rows, dtype=np.uint32))
            count_path = data / f"source/{source}/counts/00000.csr.npz"
            cell_path = data / f"source/{source}/cells/00000.parquet"
            count_path.parent.mkdir(parents=True); cell_path.parent.mkdir(parents=True)
            sparse.save_npz(count_path, counts)
            pd.DataFrame(dict(source_id=[source]*3, row_in_shard=np.arange(3), cell_id=["c0","c1","c2"],
                guide_index=[0,0,1], guide_id=["g0","g0","g1"], primary_target_id=["T0","T0","NTC"],
                is_control=[False,False,True], RNA_UMIs=np.array([3,3,3],dtype=np.uint64),
                donor_id=[donor]*3, condition=[condition]*3)).to_parquet(cell_path,index=False)
            count_ref, cell_ref = file_reference(count_path), file_reference(cell_path)
            shards.append(dict(source_id=source, shard=0, rows=3, nnz=counts.nnz,
                path=str(count_path.relative_to(data)), sha256=count_ref.sha256, bytes=count_ref.size_bytes,
                cells_path=str(cell_path.relative_to(data)), cells_sha256=cell_ref.sha256,
                cell_table_bytes=cell_ref.size_bytes, exact_reload=True))
            pd.DataFrame(dict(source_id=[source]*3, guide_index=[0,1,2], guide_id=["g0","g1","g2"],
                observed=[True]*3, n_cells=[2,1,0], captured_cell_denominator=[3]*3, guide_universe_size=[3]*3,
                jeffreys_frequency=np.array([2.5,1.5,0.5])/4.5)).to_parquet(
                    data / f"source/{source}/population_accounting.parquet",index=False)
    for name, rows in (("samples_and_culture_branches",samples),
            ("features_and_source_positions",features),("shard_manifest",shards)):
        pd.DataFrame(rows).to_parquet(data / f"source/{name}.parquet",index=False)
    # Only this freshly created bounded fixture tree is enumerated.
    members = sorted(p for p in data.rglob("*") if p.is_file())
    inventory = "".join(f"{file_reference(p).sha256}  {p.relative_to(data).as_posix()}\n" for p in members)
    with (data / "SHA256SUMS").open("x") as out: out.write(inventory)
    complete = write_json(data / "INPUT_DATA_COMPLETE.json", dict(inventory_sha256=file_reference(data / "SHA256SUMS").sha256,
        RNA_features=3, technical_features=1))
    for name in ("INPUT_DATA_COMPLETE.json", "SHA256SUMS", "source/guides.parquet",
                 "source/samples_and_culture_branches.parquet", "source/features_and_source_positions.parquet",
                 "source/shard_manifest.parquet"):
        with (metadata / name).open("xb") as out: out.write((data / name).read_bytes())
    binding = PackageBinding(complete.sha256, file_reference(data / "SHA256SUMS").sha256, 3, 3, True)
    return data, metadata, binding, NativeInputIndex(metadata, reader_source_root, binding=binding)


def _observation(root, index, fold, protocol_payload, source_sha256):
    root.mkdir()
    spec = replace(ObservationTrainingSpec.fixture(), features=3, latent_dim=48)
    observed = ObservationFold(fold.fold_id, fold.training_donors, index.rna_axis_sha256)
    protocol = json.loads(protocol_payload)
    protocol.update(RNA_feature_axis_identity=observed.feature_axis_sha256, RNA_feature_count=spec.features,
        metadata_index_identity=index.identity, architecture_dimensions=[3,*spec.hidden_dims,48],
        maximum_accepted_updates_each=spec.maximum_updates, logical_batch_positive_training_cells=spec.logical_batch,
        audit_candidates_accepted_updates=list(spec.audit_updates), checkpoint_interval_accepted_updates=spec.checkpoint_interval,
        maximum_dense_working_estimate_bytes=spec.maximum_dense_bytes,
        fixture_only=True, observation_optimizer_calls_performed=0)
    protocol_ref = write_json(root / "OBSERVATION_PROTOCOL.json", protocol)
    input_sha = index.access(fold,"representation_fit").identity()
    contract = dict(schema="credo.frontier.observation-worker-contract.v1",mode="fit",output_root=str(root),
        trajectory_id="fabricated-untrained-observation",fold_id=fold.fold_id,training_donors=list(fold.training_donors),
        rna_axis_sha256=index.rna_axis_sha256,input_access_sha256=input_sha,source_manifest_sha256=source_sha256,
        protocol_path=protocol_ref.path,protocol_sha256=protocol_ref.sha256,metadata_index_sha256=index.identity,
        fixture_only=True,observation_optimizer_calls_performed=0)
    contract_ref = write_json(root / "WORKER_CONTRACT.json",contract)
    identity = dict(trajectory_id=contract["trajectory_id"],fold_id=fold.fold_id,seed=0,input_sha256=input_sha,
        configuration_sha256=spec.identity,source_sha256=source_sha256,worker_contract_sha256=contract_ref.sha256,
        fabricated_observation_without_training=True)
    model = SparseR48(observed,features=3,hidden_dims=spec.hidden_dims,latent_dim=48,
        maximum_rows=spec.logical_batch,maximum_dense_bytes=spec.maximum_dense_bytes,seed=0)
    with torch.no_grad():
        model.center.copy_(torch.arange(48)/10)
        model.scale.copy_(torch.arange(48)/10+0.5)
        model.scaling_ready.fill_(True)
    model.freeze()
    scaling = dict(positive_RNA_cells=12,population_variance_ddof=0,scale_floor=0.01,fold_id=fold.fold_id,
        scale_floor_rule="max(1e-6,fraction*median_positive_fitting_std)")
    candidate = dict(accepted_updates=2,risk=1.,positive_cell_directions=4,sum_normalized_cross_entropy=4.,
        positive_RNA_cells=2,thinning_seed=1,independent_donors=False,
        artifact=dict(relative_uri="fabricated-candidate-not-produced.pt",sha256="d"*64,size_bytes=128))
    payload = dict(schema="credo.frontier.selected-observation.v1",inference_only=True,identity=identity,
        selected_candidate=candidate,scaling=scaling,completed_updates=spec.maximum_updates,
        model_state={name:value.clone() for name,value in model.state_dict().items()})
    publication = _publish_tensor(root,"SELECTED_OBSERVATION.pt",payload)
    complete = write_json(root / "COMPLETE.json",dict(schema="credo.frontier.observation-worker-completion.v1",
        status="complete",identity=identity,scope="synthetic_cpu_only",accepted_updates=spec.maximum_updates,
        attempted_updates=spec.maximum_updates,scientific_updates=0,actual_disposable_optimizer_calls=0,
        fixture_only=True,main_dynamics_fitted=False,held_out_donor_outcomes_opened=False,
        selected_export=dict(publication=publication,scaling=scaling,selected_updates=2,
            completed_updates=spec.maximum_updates,optimizer_resume=False,technical_holdout_excluded_from_scaling=True),
        audit_history=[candidate]))
    write_json(root / "FIXTURE_ONLY.json",dict(fixture_only=True,optimizer_calls=0,
        scientific_selected_model=False,nominal_update_fields_are_fabricated=True,
        scope="Untrained observation-loader schema fixture; never reusable as a scientific model"))
    return ObservationArtifactAdmission(file_reference(root / "SELECTED_OBSERVATION.pt"),
        file_reference(root / "SELECTED_OBSERVATION.pt.json"),complete,contract_ref,protocol_ref,observed,spec,identity)


def prepare_artifacts(root, *, donors, reader_source_root, observation_protocol,
                      normalization_protocol, rules_reference, source_sha256):
    """Create one fresh fixture fold; reuse it for the direct continuation."""
    root = _path(root); _sha(source_sha256)
    if (donors not in (("D3","D4"),("D2","D3","D4")) or STUDY_ID not in root.parts
            or not root.parent.is_dir() or torch.get_num_threads()>2):
        raise ValueError("Bounded two-thread, fixed-fold qualification fixture root required")
    root.mkdir()  # Never adopt or overwrite an existing preparation.
    raw, metadata, binding, index = _native(root / "native",reader_source_root)
    fold = FoldInputs("subset-"+"-".join(donors),donors)
    observation = _observation(root / "observation",index,fold,observation_protocol,source_sha256)
    handles = tuple(index.open(raw,fold,role,max_batch_rows=3,max_uncompressed_bytes=MIB,
        max_cached_bytes=MIB,max_output_bytes=64*1024) for role in ("dynamics_source","dynamics_supervision"))
    provider = ProviderLimits(8*MIB,8*MIB,64)
    owner = prepare_fitting_cache(handles,observation_admission=observation,
        populations=tuple(FittingPopulation(d,c,d+"_"+c,"fabricated-culture-unresolved")
            for d in donors for c in ("Stim8hr","Stim48hr")),
        limits=PreparationLimits(CompilerLimits(8*MIB,3,64*1024,2,64*1024),provider,64*MIB,256*1024,3))
    prep = owner.publish(root / "prepared",output_cap_bytes=96*MIB)
    geometry = RaggedSinkhornGeometry(options=GeometryOptions(100000,1,geomloss_backend().identity["version"]))
    norms = publish_normalizers(prep["root"],root / "normalizers",preparation_complete_sha256=prep["complete_sha256"],
        preparation_output_cap_bytes=96*MIB,limits=NormalizationLimits(provider,2*len(donors),MIB,4*MIB),
        geometry=geometry,study_id=STUDY_LABEL,study_seed=STUDY_SEED,protocol_payload=normalization_protocol,
        source_particles=8,block=4)
    base = dict(schema=INPUT_SCHEMA,study_id=STUDY_LABEL,study_seed=STUDY_SEED,source_sha256=source_sha256,
        native=dict(metadata_root=str(metadata),reader_source_root=str(reader_source_root),binding=asdict(binding),
                    metadata_index_sha256=index.identity),observation=asdict(observation),
        preparation=dict(root=prep["root"],complete_sha256=prep["complete_sha256"],output_cap_bytes=96*MIB,
                         provider_limits=asdict(provider)),
        normalizers=dict(root=norms["root"],complete_sha256=norms["complete_sha256"],maximum_artifact_bytes=MIB,
                         output_cap_bytes=4*MIB),rules=asdict(rules_reference),schedule=asdict(worker_schedule()),
        execution=asdict(ObjectiveExecution(1,1,3,4,4,100000,3e-6,3e-6)),
        geometry=asdict(geometry.options),maximum_checkpoint_bytes=64*MIB)
    refs = {}
    for case in cases():
        if case.training_donors == donors:
            _, _, task = stage_task(case)
            value = dict(base,task=asdict(task))
            FittingInputContract(json.dumps(value,sort_keys=True,allow_nan=False).encode())
            refs[case.stage] = asdict(write_json(root / ("INPUTS-"+case.stage+".json"),value))
    return write_json(root / "FIXTURE_COMPLETE.json",dict(schema="credo.frontier.main-worker-fixture.v1",
        status="fabricated_input_preparation_complete",donors=list(donors),inputs=refs,
        observation_optimizer_calls=0,main_optimizer_calls=0,scientific_updates=0,
        real_biological_payload_access=False,nominal_observation_updates_are_schema_fixture_only=True,
        counts=dict(genes=3,catalog=3,source_cells_per_donor=3,endpoint_cells_per_donor=3),
        producer_sha256=file_reference(Path(__file__).absolute()).sha256,
        scientific_launch_authority=False,automatic_retry=False))
