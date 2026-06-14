import unittest
from types import SimpleNamespace

from eagle3_draft.peft_compat import disable_bitsandbytes_dispatch


class PeftCompatibilityTest(unittest.TestCase):
    def test_temporarily_disables_bitsandbytes_dispatch(self) -> None:
        original_bnb = lambda: True
        original_bnb_4bit = lambda: True
        module = SimpleNamespace(
            is_bnb_available=original_bnb,
            is_bnb_4bit_available=original_bnb_4bit,
        )

        with disable_bitsandbytes_dispatch(module):
            self.assertFalse(module.is_bnb_available())
            self.assertFalse(module.is_bnb_4bit_available())

        self.assertIs(module.is_bnb_available, original_bnb)
        self.assertIs(module.is_bnb_4bit_available, original_bnb_4bit)


if __name__ == "__main__":
    unittest.main()
