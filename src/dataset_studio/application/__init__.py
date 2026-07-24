"""Módulo de serviços da aplicação do Dataset Studio."""

from dataset_studio.adapters.ultralytics.trainer import UltralyticsCommandTrainer
from dataset_studio.application.archive_service import (
    archive_status,
    attach_archive_to_dataset,
    attach_archive_to_run,
    import_archive_snapshot,
    materialize_archive_snapshot,
    verify_archive_snapshot,
)
from dataset_studio.application.job_service import JobManager
from dataset_studio.application.extraction_service import ExtractionJobManager
from dataset_studio.application.deployment_service import (
    export_deployment_bundle,
    validate_deployment_bundle,
)
from dataset_studio.application.label_studio_service import (
    ensure_label_studio_project,
    label_studio_integration_status,
)
from dataset_studio.application.registry_service import (
    begin_training_record,
    finalize_training_record,
    promote_registered_model,
    registry_status,
    resolve_model_reference,
    snapshot_version_dataset,
)
from dataset_studio.application.source_service import (
    campaign_status,
    inspect_finished_tasks,
    list_available_models,
    source_status,
)
from dataset_studio.application.version_service import (
    preview_combined_split_metrics,
    preview_split_metrics,
    release_status,
    training_recipe,
    version_status,
)
from dataset_studio.ports.trainer import TrainingParams

__all__ = [
    "JobManager",
    "ExtractionJobManager",
    "archive_status",
    "attach_archive_to_dataset",
    "attach_archive_to_run",
    "import_archive_snapshot",
    "materialize_archive_snapshot",
    "verify_archive_snapshot",
    "export_deployment_bundle",
    "validate_deployment_bundle",
    "ensure_label_studio_project",
    "label_studio_integration_status",
    "begin_training_record",
    "finalize_training_record",
    "promote_registered_model",
    "registry_status",
    "resolve_model_reference",
    "snapshot_version_dataset",
    "source_status",
    "campaign_status",
    "inspect_finished_tasks",
    "list_available_models",
    "preview_split_metrics",
    "preview_combined_split_metrics",
    "version_status",
    "release_status",
    "training_recipe",
    "TrainingParams",
    "UltralyticsCommandTrainer",
]
