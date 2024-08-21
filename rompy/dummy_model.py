# dummy_model.py
from rompy.registry import TypeRegistry


class ModelRunMeta(type):
    def __new__(mcs, name, bases, attrs):
        return ModelRunProxy


class ModelRunProxy:
    def __new__(cls, *args, **kwargs):
        actual_model_run = TypeRegistry.create_model_run()
        return actual_model_run(*args, **kwargs)

    def __getattr__(name):
        actual_model_run = TypeRegistry.create_model_run()
        return getattr(actual_model_run, name)

    @classmethod
    def __class_getitem__(cls, key):
        actual_model_run = TypeRegistry.create_model_run()
        return actual_model_run.__class_getitem__(key)


class ModelRun(metaclass=ModelRunMeta):

    def dummy_method(self):
        print("dummy")

    @property
    def dummy_property(self):
        return "dummy"