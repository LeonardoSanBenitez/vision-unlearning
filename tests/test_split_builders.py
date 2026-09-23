"""Each split builder must forward the caption rule into the ``metadata.jsonl`` it writes.

The rule existing is not the same as the rule arriving: these tests run each builder over a tiny
source folder and read the file that comes out. Heavy tier -- ``vision_unlearning.datasets.others``
imports pandas, scipy, unidecode and the datasets library at module level -- but no network and no
torch: the one builder that downloads its source has that call substituted, and the two that
symlink have ``os.symlink`` substituted for a copy, as the acquisition scripts already do on
Windows.
"""
import json
import os
import shutil
from typing import Any, Dict, List

from vision_unlearning.benchmarks.I_care import split_captions as sc
from vision_unlearning.datasets import others


def _read_rows(folder: str) -> List[Dict[str, str]]:
    with open(os.path.join(folder, sc.METADATA_FILE), encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _image(folder: str, name: str, size: Any = (4, 4)) -> str:
    from PIL import Image

    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    Image.new('RGB', size, color=(3, 2, 1)).save(path)
    return path


class TestTheBuildersForwardTheRule:
    """One file-producing test per builder: the rule must reach the file, not just exist."""

    def test_create_metadata_jsonl_defaults_to_the_class_name(self, tmp_path: Any) -> None:
        """The library default is unchanged, so no existing caller changes behaviour."""
        folder = str(tmp_path / 'plain')
        _image(folder, 'abbey_0001.jpg')
        others.create_metadata_jsonl(tmp_path / 'plain')
        assert _read_rows(folder) == [{'file_name': 'abbey_0001.jpg', 'text': 'abbey'}]

    def test_create_metadata_jsonl_applies_the_caption_rule(self, tmp_path: Any) -> None:
        folder = str(tmp_path / 'scenes')
        _image(folder, 'abbey_0001.jpg')
        others.create_metadata_jsonl(tmp_path / 'scenes', sc.caption_fn_for('scenes'))
        assert _read_rows(folder) == [{'file_name': 'abbey_0001.jpg', 'text': 'An image of an abbey scene'}]

    def test_the_people_builder_writes_canonical_captions(self, tmp_path: Any, monkeypatch: Any) -> None:
        from PIL import Image

        samples = [
            {'filename': 'Colin_Powell_0001.jpg', 'image': Image.new('RGB', (4, 4))},
            {'filename': 'Colin_Powell_0002.jpg', 'image': Image.new('RGB', (4, 4))},
            {'filename': 'Adrien_Brody_0005.jpg', 'image': Image.new('RGB', (4, 4))},
        ]
        monkeypatch.setattr(others, 'load_dataset', lambda *a, **k: samples)

        forget = str(tmp_path / 'lfw' / 'Colin_Powell' / 'train_forget')
        retain = str(tmp_path / 'lfw' / 'Colin_Powell' / 'train_retain')
        others.download_dataset_lfw(
            forget, retain, target='Colin_Powell', caption_fn=sc.caption_fn_for('people'),
        )

        forget_rows = _read_rows(forget)
        assert [row['file_name'] for row in forget_rows] == ['Colin_Powell_0001.jpg', 'Colin_Powell_0002.jpg']
        assert {row['text'] for row in forget_rows} == {'An image of Colin Powell'}
        assert _read_rows(retain) == [{'file_name': 'Adrien_Brody_0005.jpg', 'text': 'An image of Adrien Brody'}]

    def test_the_breeds_builder_writes_canonical_captions(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setattr(others.os, 'symlink', lambda src, dst: shutil.copyfile(src, dst))

        source = tmp_path / 'downloaded'
        _image(str(source / 'basenji dog'), 'a.jpg')
        _image(str(source / 'affenpinscher dog'), 'b.jpg')

        forget = str(tmp_path / 'breeds' / 'target' / 'train_forget')
        retain = str(tmp_path / 'breeds' / 'target' / 'train_retain')
        others.split_dataset_taras_breeds(
            str(source), forget, retain, target='basenji dog', caption_fn=sc.caption_fn_for('breeds'),
        )

        assert {row['text'] for row in _read_rows(forget)} == {'An image of a basenji dog'}
        assert {row['text'] for row in _read_rows(retain)} == {'An image of an affenpinscher dog'}

    def test_the_scenes_builder_writes_canonical_captions(self, tmp_path: Any, monkeypatch: Any) -> None:
        import numpy as np
        import scipy.io as sio

        monkeypatch.setattr(others.os, 'symlink', lambda src, dst: shutil.copyfile(src, dst))

        source = tmp_path / 'SUN'
        names = ['a/abbey/one.jpg', 'f/football_field/two.jpg']
        for name in names:
            _image(str(source / 'images' / os.path.dirname(name)), os.path.basename(name))
        os.makedirs(str(source / 'SUNAttributeDB'), exist_ok=True)
        sio.savemat(
            str(source / 'SUNAttributeDB' / 'images.mat'),
            {'images': np.array([[name] for name in names], dtype=object)},
        )

        forget = str(tmp_path / 'scenes' / 'target' / 'train_forget')
        retain = str(tmp_path / 'scenes' / 'target' / 'train_retain')
        others.split_dataset_sun(
            str(source), forget, retain, target='football_field', caption_fn=sc.caption_fn_for('scenes'),
        )

        assert {row['text'] for row in _read_rows(forget)} == {'An image of a football field scene'}
        assert {row['text'] for row in _read_rows(retain)} == {'An image of an abbey scene'}
