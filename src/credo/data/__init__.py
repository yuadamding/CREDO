"""Storage-independent study semantics and compatibility codecs."""

from .codecs import (
    StudyCodec,
    StudyCodecRegistry,
    available_study_codecs,
    open_study,
    register_study_codec,
)
from .design import (
    AxisSpec,
    Checkpoint,
    LongitudinalDesign,
    ProgressionAxis,
    StudyDesign,
    Transition,
)
from .legacy import CurrentFiveFileStudyCodec, FiveFileV2Codec, observation_id
from .native import (
    NativeH5SupportStore,
    NativeStudyV3Codec,
    StudyBuilder,
)
from .native import (
    write_study as write_study_v3,
)
from .native_v4 import (
    NativePackedH5SupportStore,
    NativePerturbSeqStudyV4Codec,
    PerturbSeqStudyBuilder,
    write_perturb_seq_study,
)
from .representations import ArtifactRef, RepresentationCatalog, RepresentationSpec
from .splits import SplitPlan
from .study import (
    CompositionPolicy,
    ReplicatePolicy,
    SelectionSpec,
    Study,
    StudyManifest,
    StudyView,
)
from .support import (
    AbundanceValue,
    EmpiricalLaw,
    InMemorySupportStore,
    MeasureSnapshot,
    SupportRef,
    SupportStore,
    SupportStoreRegistry,
)
from .tables import (
    AbundanceChannelSpec,
    AbundanceSemantics,
    AbundanceTable,
    CompositionTable,
    ConditionTable,
    ContextTable,
    EffectBindingTable,
    InterventionEventTable,
    LPSCompositionTable,
    ObservationTable,
    PerturbationComponentTable,
    PerturbationEffectBindingTable,
    PerturbationReferenceBindingTable,
    PerturbationTable,
    PopulationPoolTable,
    PopulationSeriesTable,
    ReferenceBindingTable,
    SeriesTable,
    SnapshotObservationTable,
    SupportIndexTable,
)
from .validation import ValidationIssue, ValidationReport


def write_study(study, destination):
    """Write schema v4 for PerturbSeqStudy and preserve explicit v3 compatibility."""
    from ..lps import PerturbSeqStudy

    if isinstance(study, PerturbSeqStudy):
        return write_perturb_seq_study(study, destination)
    if isinstance(study, Study):
        return write_study_v3(study, destination)
    raise TypeError("write_study requires a PerturbSeqStudy or schema-v3 Study.")


__all__ = [
    "AbundanceChannelSpec",
    "AbundanceSemantics",
    "AbundanceTable",
    "AbundanceValue",
    "ArtifactRef",
    "AxisSpec",
    "Checkpoint",
    "CompositionTable",
    "CompositionPolicy",
    "ConditionTable",
    "ContextTable",
    "CurrentFiveFileStudyCodec",
    "EmpiricalLaw",
    "EffectBindingTable",
    "FiveFileV2Codec",
    "InMemorySupportStore",
    "InterventionEventTable",
    "LPSCompositionTable",
    "LongitudinalDesign",
    "MeasureSnapshot",
    "NativeH5SupportStore",
    "NativePackedH5SupportStore",
    "NativePerturbSeqStudyV4Codec",
    "NativeStudyV3Codec",
    "ObservationTable",
    "PerturbationComponentTable",
    "PerturbationEffectBindingTable",
    "PerturbationReferenceBindingTable",
    "PerturbationTable",
    "PerturbSeqStudyBuilder",
    "ProgressionAxis",
    "PopulationPoolTable",
    "PopulationSeriesTable",
    "ReferenceBindingTable",
    "RepresentationCatalog",
    "RepresentationSpec",
    "ReplicatePolicy",
    "SelectionSpec",
    "SeriesTable",
    "SnapshotObservationTable",
    "SplitPlan",
    "Study",
    "StudyCodec",
    "StudyCodecRegistry",
    "StudyDesign",
    "StudyManifest",
    "StudyBuilder",
    "StudyView",
    "SupportRef",
    "SupportStore",
    "SupportIndexTable",
    "SupportStoreRegistry",
    "Transition",
    "ValidationIssue",
    "ValidationReport",
    "available_study_codecs",
    "observation_id",
    "open_study",
    "register_study_codec",
    "write_study",
    "write_study_v3",
]
