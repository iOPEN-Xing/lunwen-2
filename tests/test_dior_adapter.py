from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reranker_common import synthetic_pair_records


def test_synthetic_pair_reader_filters_ambiguous_and_positive_only(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"image":"a.jpg","target":0,"positive_text":"p","negative_text":"n","unique_status":"unique"}\n'
        '{"image":"a.jpg","target":0,"positive_text":"p2","negative_text":"n2","unique_status":"ambiguous"}\n'
        '{"image":"a.jpg","target":0,"positive_text":"p3","negative_text":null,"unique_status":"unique"}\n',
        encoding="utf-8",
    )
    rows = synthetic_pair_records(path, {"a.jpg": "train"}, {"train"})
    assert len(rows) == 1
    assert rows[0]["negative_text"] == "n"

