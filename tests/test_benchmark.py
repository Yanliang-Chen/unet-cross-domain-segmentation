import math
import unittest
import numpy as np
import torch

from benchmark.validators import image_statistics, validator_scores, auroc


def logits_from(prob):
    return torch.logit(torch.as_tensor(prob, dtype=torch.float32).clamp(1e-6, 1-1e-6))


def stats_for(clean, noisy=None):
    clean = logits_from(clean)
    return image_statistics(clean, clean if noisy is None else logits_from(noisy))


class ValidatorTests(unittest.TestCase):
    def test_identical_predictions_have_perfect_cmr(self):
        p = torch.zeros(2, 1, 8, 8)
        p[:, :, 2:5, 2:5] = 0.9
        s = stats_for(p)
        assert np.allclose(s["cmr_nhd"], 1.0)
        assert np.all(s["cmr_ei"] > 0.85)

    def test_cmr_counts_only_pixels_predicted_in_either_map(self):
        clean = torch.zeros(1, 1, 10, 10)
        clean[..., :2, :2] = 0.9                      # 4 foreground pixels
        noisy = clean.clone()
        noisy[..., 0, 0] = 0.1                        # one flipped
        s = stats_for(clean, noisy)
        fg_iou, bg_iou = 3/4, 96/97
        assert math.isclose(float(s["cmr_nhd"][0]), (fg_iou+bg_iou)/2, rel_tol=1e-5)

    def test_empty_prediction_omits_absent_foreground_class(self):
        s = stats_for(torch.full((1, 1, 4, 4), 0.01))
        assert np.allclose(s["cmr_nhd"], 1.0)

    def test_foreground_only_cmr_scores_empty_prediction_zero(self):
        s = stats_for(torch.full((1, 1, 4, 4), 0.01))
        assert np.allclose(s["cmr_nhd_fg"], 0.0) and np.allclose(s["cmr_ei_fg"], 0.0)

    def test_entropy_and_infomax_orientation(self):
        uncertain = validator_scores(stats_for(torch.full((1, 1, 4, 4), 0.5)))
        confident = torch.zeros(1, 1, 4, 4)
        confident[..., :2, :] = 1.0
        confident = validator_scores(stats_for(confident))
        assert math.isclose(uncertain["entropy"], -math.log(2), rel_tol=1e-4)
        assert confident["entropy"] > uncertain["entropy"]
        assert confident["infomax"] > uncertain["infomax"]

    def test_bnm_prefers_confident_diverse_over_collapsed(self):
        diverse = torch.zeros(1, 1, 4, 4)
        diverse[..., :2, :] = 1.0
        collapsed = torch.zeros(1, 1, 4, 4)
        assert (validator_scores(stats_for(diverse))["bnm"]
                > validator_scores(stats_for(collapsed))["bnm"])

    def test_mcc_penalises_uncertain_pixels(self):
        confident = torch.zeros(1, 1, 4, 4)
        confident[..., :2, :] = 1.0
        uncertain = torch.full((1, 1, 4, 4), 0.5)
        assert (validator_scores(stats_for(confident))["mcc"]
                > validator_scores(stats_for(uncertain))["mcc"])

    def test_auroc_matches_simple_cases(self):
        assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
        assert auroc([0.1], [0.9]) == 0.0
        assert auroc([0.5], [0.5]) == 0.5


if __name__ == "__main__":
    unittest.main()
