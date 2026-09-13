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
    # Pack M1: the programmatic export paths now REFUSE a label that is
    # not in its own track's class set, rather than writing it verbatim
    # (export_intervals) or as a silent -1 (export_per_sample). "PS" was
    # an orphan against this fixture's default schema, so the schema says
    # so now.
    labeler.classes = ["UNKNOWN", "PS"]
    labeler.intervals = [Interval(a, b, "PS")]

    out = tmp_path / "intervals.csv"
    labeler.export_intervals(str(out), fmt="csv")
    text = out.read_text().strip()
    # Very light snapshot: just check header + a couple of field names are present
    # Pack M1: `track` is ALWAYS present, between end and label. This
    # export has no byte-identity floor -- R2 grants one only to the
    # per-sample column and its sidecar -- and a conditional schema would
    # force every reader to branch on the file's own shape.
    assert "start,end,track,label,notes" in text.splitlines()[0]
    assert "PS" in text
    assert "default" in text
