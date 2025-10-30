import pandas as pd
from chronotagger.core.models import Interval


def test_interval_contains_and_overlap():
    a = pd.Timestamp("2020-01-01 00:00:00")
    b = pd.Timestamp("2020-01-01 00:10:00")
    c = pd.Timestamp("2020-01-01 00:05:00")
    d = pd.Timestamp("2020-01-01 00:20:00")

    iv1 = Interval(a, b, "A")
    iv2 = Interval(c, d, "B")
    iv3 = Interval(b, d, "C")      # adjacent to iv1

    assert iv1.contains(a)
    assert not iv1.contains(b)     # half-open [start, end)
    assert iv1.overlaps(iv2)
    assert not iv1.overlaps(iv3)   # adjacency isn't overlap


def test_interval_serde_roundtrip():
    iv = Interval(pd.Timestamp("2020-01-01 00:00:00"),
                  pd.Timestamp("2020-01-01 00:05:00"),
                  "PS", "note")
    back = Interval.from_dict(iv.to_dict())
    assert (iv.start, iv.end, iv.label, iv.notes) == (back.start, back.end, back.label, back.notes)
