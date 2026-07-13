"""Tests for vision_unlearning.integrations.huggingface.

Scope: get_hf_token_from_env(), the shared HF_TOKEN normalization helper used
by InterferencePerEntity.compute(), ResultTemplate.compute(), and
GeneratedDataset.compute() to avoid passing an empty-string token (which
builds an illegal 'Authorization: Bearer ' header) instead of None.
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from vision_unlearning.integrations.huggingface import get_hf_token_from_env


class TestGetHfTokenFromEnv(unittest.TestCase):

    def test_returns_none_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('HF_TOKEN', None)
            self.assertIsNone(get_hf_token_from_env())

    def test_returns_none_when_set_to_empty_string(self) -> None:
        with patch.dict(os.environ, {'HF_TOKEN': ''}, clear=False):
            self.assertIsNone(get_hf_token_from_env())

    def test_returns_token_when_set_to_real_value(self) -> None:
        with patch.dict(os.environ, {'HF_TOKEN': 'hf_real_token_value'}, clear=False):
            self.assertEqual(get_hf_token_from_env(), 'hf_real_token_value')


if __name__ == '__main__':
    unittest.main()
