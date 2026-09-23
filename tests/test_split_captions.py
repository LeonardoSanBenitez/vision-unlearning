"""The caption stored beside a training image must be the string the model is prompted with.

Three layers, because each one can be right while the next is wrong:

* the rule itself, over the three tasks;
* the audit and the repair of splits that already exist, including the counted ways a split can
  disagree with itself;
* the precondition an unlearning session runs before conditioning a model on a split.

The middle layer -- that each of the three split builders actually forwards the rule into the file
it writes -- is in ``test_split_builders.py``, which is heavy-tier because the builders' module
imports pandas, scipy and the datasets library at import time.
"""
import json
import os
from typing import Any, Dict, List

import pytest

from vision_unlearning.benchmarks.I_care import split_captions as sc


def _write_rows(folder: str, rows: List[Dict[str, str]]) -> str:
    path = os.path.join(folder, sc.METADATA_FILE)
    with open(path, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    return path


def _read_rows(folder: str) -> List[Dict[str, str]]:
    with open(os.path.join(folder, sc.METADATA_FILE), encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _image(folder: str, name: str, size: Any = (4, 4)) -> str:
    from PIL import Image

    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    Image.new('RGB', size, color=(3, 2, 1)).save(path)
    return path


class TestTheRule:
    def test_the_caption_is_the_canonical_entity(self) -> None:
        assert sc.caption_fn_for('scenes')('abbey') == 'An image of an abbey scene'
        assert sc.caption_fn_for('breeds')('basenji dog') == 'An image of a basenji dog'
        assert sc.caption_fn_for('people')('Colin_Powell') == 'An image of Colin Powell'

    def test_the_class_name_drops_the_index_or_uuid(self) -> None:
        assert sc.class_name_of('Colin_Powell_0022.jpg') == 'Colin_Powell'
        assert sc.class_name_of('abbey_029c5a6d446d4f73b31626a2e4eb9df4.jpg') == 'abbey'
        assert sc.class_name_of('bouvier des flandres dog_143f3cb9.jpg') == 'bouvier des flandres dog'

    def test_the_expected_caption_of_a_filename(self) -> None:
        assert sc.expected_caption('people', 'Colin_Powell_0022.jpg') == 'An image of Colin Powell'
        assert sc.expected_caption('scenes', 'football_field_01e6adf7.jpg') == 'An image of a football field scene'
        assert sc.expected_caption('breeds', 'affenpinscher dog_002e4174.jpg') == 'An image of an affenpinscher dog'


class TestTheAudit:
    def test_a_correct_split_is_clean(self, tmp_path: Any) -> None:
        folder = str(tmp_path / 'train_forget')
        _image(folder, 'abbey_0001.jpg')
        _write_rows(folder, [{'file_name': 'abbey_0001.jpg', 'text': 'An image of an abbey scene'}])

        audit = sc.audit_split(folder, 'scenes')

        assert audit.is_clean
        assert (audit.n_rows, audit.n_images, audit.n_captions_wrong) == (1, 1, 0)

    def test_every_way_a_split_disagrees_with_itself_is_counted(self, tmp_path: Any) -> None:
        folder = str(tmp_path / 'train_retain')
        _image(folder, 'abbey_0001.jpg')
        _image(folder, 'abbey_0002.jpg')      # present, but no row will mention it
        _write_rows(folder, [
            {'file_name': 'abbey_0001.jpg', 'text': 'abbey'},        # wrong caption
            {'file_name': 'abbey_0003.jpg', 'text': 'An image of an abbey scene'},  # row without an image
        ])

        audit = sc.audit_split(folder, 'scenes')

        assert not audit.is_clean
        assert audit.n_rows == 2
        assert audit.n_images == 2
        assert audit.n_captions_wrong == 1
        assert audit.n_rows_without_image == 1
        assert audit.n_images_without_row == 1
        assert audit.first_wrong == {
            'row': '0', 'file_name': 'abbey_0001.jpg', 'found': 'abbey', 'expected': 'An image of an abbey scene',
        }

    def test_a_split_without_metadata_is_reported_not_crashed(self, tmp_path: Any) -> None:
        folder = str(tmp_path / 'train_forget')
        _image(folder, 'abbey_0001.jpg')

        audit = sc.audit_split(folder, 'scenes')

        assert audit.error is not None
        assert not audit.is_clean
        assert audit.n_images == 1


class TestTheRepair:
    def test_only_the_caption_changes(self, tmp_path: Any) -> None:
        folder = str(tmp_path / 'train_retain')
        _image(folder, 'abbey_0001.jpg')
        _image(folder, 'football_field_0002.jpg')
        _write_rows(folder, [
            {'file_name': 'football_field_0002.jpg', 'text': 'football_field'},
            {'file_name': 'abbey_0001.jpg', 'text': 'abbey'},
        ])

        before = sc.repair_split_captions(folder, 'scenes')

        assert before.n_captions_wrong == 2
        assert _read_rows(folder) == [
            {'file_name': 'football_field_0002.jpg', 'text': 'An image of a football field scene'},
            {'file_name': 'abbey_0001.jpg', 'text': 'An image of an abbey scene'},
        ]
        assert sc.audit_split(folder, 'scenes').is_clean

    def test_a_second_pass_changes_nothing(self, tmp_path: Any) -> None:
        folder = str(tmp_path / 'train_forget')
        _image(folder, 'abbey_0001.jpg')
        _write_rows(folder, [{'file_name': 'abbey_0001.jpg', 'text': 'abbey'}])

        sc.repair_split_captions(folder, 'scenes')
        after_first = open(os.path.join(folder, sc.METADATA_FILE), 'rb').read()
        second = sc.repair_split_captions(folder, 'scenes')

        assert second.n_captions_wrong == 0
        assert open(os.path.join(folder, sc.METADATA_FILE), 'rb').read() == after_first

    def test_dry_run_writes_nothing(self, tmp_path: Any) -> None:
        folder = str(tmp_path / 'train_forget')
        _image(folder, 'abbey_0001.jpg')
        path = _write_rows(folder, [{'file_name': 'abbey_0001.jpg', 'text': 'abbey'}])
        before_bytes = open(path, 'rb').read()

        result = sc.repair_split_captions(folder, 'scenes', dry_run=True)

        assert result.n_captions_wrong == 1
        assert open(path, 'rb').read() == before_bytes


class TestThePrecondition:
    def test_a_prompt_form_split_passes(self, tmp_path: Any) -> None:
        folder = str(tmp_path / 'train_forget')
        _image(folder, 'abbey_0001.jpg')
        _write_rows(folder, [{'file_name': 'abbey_0001.jpg', 'text': 'An image of an abbey scene'}])

        sc.assert_captions_prompted(folder, 'scenes')

    def test_a_wrong_caption_names_the_file_and_the_first_row(self, tmp_path: Any) -> None:
        folder = str(tmp_path / 'train_forget')
        _image(folder, 'abbey_0001.jpg')
        _write_rows(folder, [{'file_name': 'abbey_0001.jpg', 'text': 'abbey'}])

        with pytest.raises(ValueError) as caught:
            sc.assert_captions_prompted(folder, 'scenes')

        message = str(caught.value)
        assert sc.METADATA_FILE in message
        assert 'abbey_0001.jpg' in message
        assert "expected 'An image of an abbey scene'" in message

    def test_an_empty_split_is_refused(self, tmp_path: Any) -> None:
        folder = str(tmp_path / 'train_forget')
        os.makedirs(folder, exist_ok=True)
        _write_rows(folder, [])

        with pytest.raises(ValueError):
            sc.assert_captions_prompted(folder, 'scenes')
