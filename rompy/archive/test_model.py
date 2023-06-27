"""Test the base model class."""
from rompy.model import Model


def test_instantiate(tmpdir):
    child0 = Model(id="child0", kind="swan", workdir=tmpdir / "child0", children=[])
    child1 = Model(id="child1", kind="swan", workdir=tmpdir / "child1", children=[])
    model = Model(
        id="parent",
        kind="swan",
        workdir=tmpdir / "parent",
        children=[child0, child1],
    )
    print(model)

