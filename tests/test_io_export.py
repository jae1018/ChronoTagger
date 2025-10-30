def test_save_then_load_roundtrip(tmp_path, labeler):
    from chronotagger.core.models import Interval
    a, b = labeler.df.index[0], labeler.df.index[10]
    labeler.intervals = [Interval(a, b, "PS")]

    path = tmp_path / "session.json"
    labeler._save_session(str(path))

    # Nuke and reload
    labeler.intervals = []
    labeler._load_session(str(path))

    assert len(labeler.intervals) == 1
    assert labeler.intervals[0].label == "PS"


def test_export_intervals_csv_snapshot(tmp_path, labeler):
    from chronotagger.core.models import Interval
    a, b = labeler.df.index[0], labeler.df.index[10]
    labeler.intervals = [Interval(a, b, "PS")]

    out = tmp_path / "intervals.csv"
    labeler.export_intervals(str(out), fmt="csv")
    text = out.read_text().strip()
    # Very light snapshot: just check header + a couple of field names are present
    assert "start,end,label,notes" in text.splitlines()[0]
    assert "PS" in text
