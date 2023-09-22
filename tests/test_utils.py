import pytest

from rompy.utils import import_function


def test_import_function():
    from xarray import open_dataset
    func = import_function("xarray.open_dataset")
    assert open_dataset == func


def test_import_function_already_imported():
    from xarray import open_dataset
    func = import_function(open_dataset)
    assert open_dataset == func


def test_import_function_need_full_import_path():
    with pytest.raises(ValueError):
        import_function("open_dataset")


def test_import_function_exists():
    with pytest.raises(ImportError):
        import_function("aaa.bbb.ccc.dummy")
