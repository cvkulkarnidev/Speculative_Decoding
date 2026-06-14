import unittest
from types import SimpleNamespace

from eagle3_draft.compatibility import validate_assistant_config


class CompatibilityTest(unittest.TestCase):
    def test_accepts_gemma4_assistant(self) -> None:
        validate_assistant_config(SimpleNamespace(model_type="gemma4_assistant"))

    def test_rejects_target_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "Gemma 4 \\*-assistant"):
            validate_assistant_config(SimpleNamespace(model_type="gemma4"))


if __name__ == "__main__":
    unittest.main()
