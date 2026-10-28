# Conditional semantic CE, first epoch under a 15-epoch schedule

This opt-in experiment adds weighted 17-class CE on **all GT-occupied voxels** to B's unchanged full-space 18-class CE/Lovasz/set objective. Free=17 and ignore=255 are excluded only from the auxiliary. Lambda=1 is independent of the original CE coefficient=10. The model, temporal curriculum and baseline files are preserved; the new configuration is a literal standalone snapshot.

```bash
python3.10 -m experiments.sparseworld_opus_gaussian_bias.conditional_first_epoch.prepare_config
python3.10 -m pytest -q experiments/sparseworld_opus_gaussian_bias/conditional_first_epoch/test_contracts.py
python3.10 -m experiments.sparseworld_opus_gaussian_bias.conditional_first_epoch.smoke
python3.10 -m experiments.sparseworld_opus_gaussian_bias.conditional_first_epoch.launch
```

The verified queue uses 15 epochs / 15360 updates for scheduling, runs one full 4096-sample epoch and 1024-sample evaluation, saves a restartable checkpoint, compares to B/A/Strict, and pauses. It never advances into epoch two on its own. Detailed diagnosis runs at epoch one only.

Formal output: `out/sparseworld_opus_gaussian_bias_conditional_ce_C_4096_1024_15e/C_conditional_semantic_ce/`.

Report/status: `analysis/sparseworld_opus_gaussian_bias_conditional_ce_C_4096_1024_15e/`.

The new 17-class conditional confusion covers every current GT-occupied voxel, even when the original 18-class prediction is free. B did not record this metric over the full held-out set. Comparisons to B instead use matching 18-class IoUs, occupancy recall and semantic correctness among GT voxels predicted occupied; these metrics have different denominators.

GPU smoke includes epoch one and the full future curriculum at epoch 15, asserts C/B identical full model initialization, and checks diagnostic RNG and parameter-gradient preservation. CPU contracts verify auxiliary selection, gradients, B identity at lambda=0, configuration independence and the saved epoch-one boundary.
