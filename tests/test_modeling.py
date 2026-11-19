import numpy as np

from mp135.modeling import build_model, predict


def test_pipeline_imputes_and_excludes_post_event_columns(example_rows, config):
    train = example_rows.copy()
    train.loc[0, "nearby_offices"] = np.nan
    model = build_model(config, {"kind": "ridge", "alpha": 1.0})
    model.fit(train, train["entries_per_hour"])
    first = predict(model, train)
    changed = train.assign(gate_count_end_of_hour=-999999, entries_per_hour=999999)
    np.testing.assert_allclose(first, predict(model, changed))
    assert np.isfinite(first).all()
    assert (first >= 0).all()
    imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    assert imputer.statistics_[1] == train["nearby_offices"].median()


def test_unseen_category_does_not_crash_model(example_rows, config):
    model = build_model(config, {"kind": "ridge", "alpha": 1.0})
    model.fit(example_rows, example_rows["entries_per_hour"])
    incoming = example_rows.iloc[[0]].assign(line="Yellow Line")
    assert np.isfinite(predict(model, incoming)[0])
