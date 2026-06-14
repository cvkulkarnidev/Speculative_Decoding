import argparse
import json
import tempfile
import unittest
from pathlib import Path

from eagle3_draft.vllm_inference import (
    build_speculative_config,
    load_batch,
    validate_draft_model,
    write_batch,
)


class VllmInferenceTest(unittest.TestCase):
    def test_builds_native_eagle3_config(self) -> None:
        args = argparse.Namespace(
            draft_model="org/eagle3-draft",
            num_speculative_tokens=4,
            draft_tensor_parallel_size=2,
        )

        self.assertEqual(
            build_speculative_config(args),
            {
                "method": "eagle3",
                "model": "org/eagle3-draft",
                "num_speculative_tokens": 4,
                "draft_tensor_parallel_size": 2,
            },
        )

    def test_rejects_custom_pt_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "draft_model.pt").touch()

            with self.assertRaisesRegex(ValueError, "not a vLLM-compatible"):
                validate_draft_model(tmpdir)

    def test_batch_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir, "input.jsonl")
            output_path = Path(tmpdir, "output.jsonl")
            input_path.write_text(
                json.dumps({"response_text": "Create a chart", "id": 1}) + "\n",
                encoding="utf-8",
            )

            records, prompts = load_batch(str(input_path), "System")
            write_batch(str(output_path), records, ['{"type":"chart"}'])
            output = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertIn("Create a chart", prompts[0])
            self.assertEqual(output["id"], 1)
            self.assertEqual(output["predicted_genui_json"], '{"type":"chart"}')


if __name__ == "__main__":
    unittest.main()
