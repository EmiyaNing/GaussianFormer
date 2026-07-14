import unittest
from types import SimpleNamespace

import torch

from gaussian_statistic.aggregator import GaussianStatAggregator


def make_gaussian(means, semantic_classes, scale=2.0, num_classes=2):
    means = torch.tensor(means, dtype=torch.float32)
    count = means.shape[0]
    semantics = torch.full((count, num_classes), -4.0, dtype=torch.float32)
    semantics[
        torch.arange(count),
        torch.tensor(semantic_classes, dtype=torch.long),
    ] = 4.0
    return SimpleNamespace(
        means=means,
        scales=torch.full((count, 3), scale, dtype=torch.float32),
        rotations=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * count,
            dtype=torch.float32,
        ),
        semantics=semantics,
    )


def make_metas(voxel_xyz, voxel_labels):
    labels = torch.tensor(voxel_labels, dtype=torch.long)
    return {
        'occ_xyz': torch.tensor(voxel_xyz, dtype=torch.float32),
        'occ_label': labels,
        'occ_cam_mask': torch.ones_like(labels, dtype=torch.bool),
    }


def make_aggregator(rho, exclude_gaussian_classes=None):
    return GaussianStatAggregator(
        num_classes=2,
        cov_threshold=1.0,
        distance_bins=[0, 100],
        scale_range=[0.1, 2.0],
        purity_threshold_rho=rho,
        chunk_size=2,
        max_pair_elements=16,
        histogram_bins=64,
        exclude_gaussian_classes=exclude_gaussian_classes,
    )


class MixedGaussianStatisticTest(unittest.TestCase):
    def test_threshold_is_strictly_less_than_rho(self):
        gaussian = make_gaussian([[0, 0, 0]], [0])
        metas = make_metas(
            [[0.25, 0, 0], [1.0, 0, 0]],
            [0, 1],
        )
        aggregator = make_aggregator(rho=0.5)
        aggregator.add_frame(gaussian, metas)

        stats = aggregator.finalize()
        self.assertEqual(stats['mixed_gaussian']['mixed_count'], 0)
        self.assertEqual(stats['mixed_gaussian']['ratio_all'], 0.0)
        self.assertEqual(stats['sem_sup']['covered_voxel_count'], 0)
        self.assertEqual(stats['sem_sup']['ratio'], 0.0)

    def test_unused_is_not_mixed_and_stays_in_all_denominator(self):
        gaussian = make_gaussian(
            [[0, 0, 0], [50, 50, 50]],
            [0, 0],
        )
        metas = make_metas(
            [[0.25, 0, 0], [1.0, 0, 0]],
            [0, 1],
        )
        aggregator = make_aggregator(rho=0.6)
        aggregator.add_frame(gaussian, metas)

        stats = aggregator.finalize()
        mixed = stats['mixed_gaussian']
        self.assertEqual(mixed['mixed_count'], 1)
        self.assertEqual(mixed['evaluated_gaussian_count'], 2)
        self.assertEqual(mixed['purity_defined_gaussian_count'], 1)
        self.assertAlmostEqual(mixed['ratio_all'], 0.5)
        self.assertAlmostEqual(mixed['ratio_valid'], 1.0)
        self.assertAlmostEqual(stats['sem_sup']['ratio'], 1.0)

    def test_sem_sup_uses_voxel_union(self):
        gaussian = make_gaussian(
            [[0, 0, 0], [0.1, 0, 0]],
            [0, 0],
        )
        metas = make_metas([[0.25, 0, 0]], [1])
        aggregator = make_aggregator(rho=0.5)
        aggregator.add_frame(gaussian, metas)

        stats = aggregator.finalize()
        self.assertEqual(stats['mixed_gaussian']['mixed_count'], 2)
        self.assertEqual(stats['sem_sup']['covered_voxel_count'], 1)
        self.assertEqual(stats['sem_sup']['occupied_voxel_count'], 1)
        self.assertAlmostEqual(stats['sem_sup']['ratio'], 1.0)
        self.assertTrue(all(
            value for key, value in stats['mixed_sem_sup_sanity'].items()
            if key != 'all_scope_geometric_covered_voxel_count'
        ))

    def test_zero_rho_never_marks_a_gaussian_mixed(self):
        gaussian = make_gaussian([[0, 0, 0]], [0])
        metas = make_metas([[0.25, 0, 0]], [1])
        aggregator = make_aggregator(rho=0.0)
        aggregator.add_frame(gaussian, metas)

        stats = aggregator.finalize()
        self.assertEqual(stats['mixed_gaussian']['mixed_count'], 0)
        self.assertEqual(stats['sem_sup']['covered_voxel_count'], 0)

    def test_all_gaussians_excluded_still_count_sem_sup_denominator(self):
        gaussian = make_gaussian([[0, 0, 0]], [0])
        metas = make_metas([[0.25, 0, 0]], [1])
        aggregator = make_aggregator(
            rho=0.5,
            exclude_gaussian_classes=[0],
        )
        aggregator.add_frame(gaussian, metas)

        stats = aggregator.finalize()
        self.assertEqual(
            stats['mixed_gaussian']['evaluated_gaussian_count'], 0)
        self.assertEqual(stats['sem_sup']['occupied_voxel_count'], 1)
        self.assertEqual(stats['sem_sup']['covered_voxel_count'], 0)


if __name__ == '__main__':
    unittest.main()
