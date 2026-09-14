"""A3: A2 plus a modest multi-scale pyramid alignment term."""
_base_ = ['./forecasting_sparseworld_experiment_a2_curriculum32.py']

loss = dict(pyramid_alignment_weight=3.)

