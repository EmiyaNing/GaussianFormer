"""Configuration identity and checkpoint compatibility for the integrated D model."""

import hashlib
import json


MODEL_NAME = "sparse_opus_gausisan_world"
TYPE_NAMES = {
    "SparseWorldOPUSGaussianBias": MODEL_NAME,
    "SparseWorldBiasRenderer": "SparseOpusGaussianRenderer",
    "SparseWorldBiasGaussianEncoder": "SparseOpusGaussianEncoder",
    "SparseWorldBiasGaussianTransformer": "SparseOpusGaussianTransformer",
    "SparseWorldBiasGaussianPerceptionHead": "SparseOpusGaussianPerceptionHead",
    "SparseWorldOPUSGaussianBiasLoss": "SparseOpusGaussianWorldLoss",
    "BalancedConditionalSemanticOccupancyLoss": "SparseOpusGaussianVoxelLoss",
    "SparseWorldBiasSubsetDataset": "SparseOpusGaussianWorldDataset",
}
OUTPUT_METADATA = {
    "custom_imports",
    "experiment_name",
    "report_directory",
    "route_arm",
    "legacy_checkpoint_config_sha256",
    "legacy_training_contract_sha256",
    "strict_reference_directory",
}


def canonical_training_config(config):
    """Normalize registry renames and output paths, retaining training values.

    This permits an explicitly allowlisted D checkpoint to load under the new
    public name. Editing a loss, horizon, optimizer, seed, data path or schedule
    still changes the identity and is rejected for a stateful resume.
    """
    values = config.to_dict() if hasattr(config, "to_dict") else dict(config)

    def normalize(value):
        if isinstance(value, dict):
            return {
                key: (
                    TYPE_NAMES.get(item, item)
                    if key == "type" and isinstance(item, str)
                    else normalize(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    return normalize(
        {
            key: value
            for key, value in values.items()
            if not key.startswith("__") and key not in OUTPUT_METADATA
        }
    )


def training_contract_sha256(config):
    payload = json.dumps(
        canonical_training_config(config), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def check_resume_identity(saved, config, config_sha256, manifest_sha256):
    """Check data and optimizer semantics before any stateful continuation."""
    if saved["manifest_sha256"] != manifest_sha256:
        raise ValueError("resume subset manifests changed")
    current = training_contract_sha256(config)
    if saved.get("training_contract_sha256") is not None:
        if saved["training_contract_sha256"] != current:
            raise ValueError("resume training configuration changed")
        return "matching_training_contract"
    if saved["config_sha256"] == config_sha256:
        return "matching_configuration_file"
    allowed = config.get("legacy_checkpoint_config_sha256", [])
    expected = config.get("legacy_training_contract_sha256")
    if saved["config_sha256"] in allowed and current == expected:
        return "verified_legacy_D_registry_rename"
    raise ValueError(
        "checkpoint configuration is incompatible with this training protocol"
    )


def checkpoint_epoch_index(saved):
    """Decode both selected-weight checkpoints and resumable epoch boundaries."""
    if "epoch_index" in saved:
        return int(saved["epoch_index"])
    completed_boundary = (
        saved.get("next_iter", 0) == 0 and saved.get("phase", "train") == "train"
    )
    return max(0, int(saved["epoch"]) - int(completed_boundary))
