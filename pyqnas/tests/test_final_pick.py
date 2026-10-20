"""The run final.deploy exports: by label, by seed when given, else by the best validation score."""
import pandas as pd

from pynas import final

RUNS = pd.DataFrame(dict(label=["winner"] * 3 + ["finalist 4"] * 3, seed=[0, 1, 2, 0, 1, 2], run=list("abcdef"),
                         val_fp16=[0.60, 0.65, 0.62, 0.66, 0.70, 0.64]))


def test_best_validation_seed():
    assert final.deploy_pick(RUNS).run == "b"
    assert final.deploy_pick(RUNS, "finalist 4").run == "e"


def test_given_seed():
    assert final.deploy_pick(RUNS, "finalist 4", seed=2).run == "f"
    assert final.deploy_pick(RUNS, "finalist 4", seed="0").run == "d"
    assert final.deploy_pick(RUNS, "winner", seed=2.0).run == "c"


def test_missing():
    for kw in (dict(label="finalist 9"), dict(label="winner", seed=7)):
        try:
            final.deploy_pick(RUNS, **kw)
        except RuntimeError:
            continue
        raise AssertionError(f"no error for {kw}")


if __name__ == "__main__":
    import _run
    _run.run(dict(globals()))
