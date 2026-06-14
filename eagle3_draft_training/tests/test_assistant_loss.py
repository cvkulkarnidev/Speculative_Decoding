import unittest

try:
    import torch

    from eagle3_draft.assistant import compute_rollout_loss_and_metrics
except ModuleNotFoundError:
    torch = None
    compute_rollout_loss_and_metrics = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class AssistantLossTest(unittest.TestCase):
    def test_first_rollout_step_predicts_two_tokens_ahead(self) -> None:
        assert torch is not None
        assert compute_rollout_loss_and_metrics is not None
        labels = torch.tensor([[-100, 1, 2, 3]])
        logits = torch.full((1, 4, 5), -10.0)
        logits[0, 0, 2] = 10.0
        logits[0, 1, 3] = 10.0

        loss, metrics = compute_rollout_loss_and_metrics(
            [logits],
            None,
            labels,
            temperature=1.0,
            kl_weight=0.0,
            ce_weight=1.0,
            decay=0.8,
        )

        self.assertLess(loss.item(), 1e-6)
        self.assertEqual(metrics["step_1_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
