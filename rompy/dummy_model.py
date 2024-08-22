# In model_run.py

from rompy.registry import Registry


class ModelRunMeta(type):
    def __new__(mcs, name, bases, attrs):
        return ModelRunProxy


class ModelRunProxy:
    def __new__(cls, *args, **kwargs):
        actual_model_run = Registry.create_model_run()
        return actual_model_run(*args, **kwargs)

    def __getattr__(name):
        actual_model_run = Registry.create_model_run()
        return getattr(actual_model_run, name)

    @classmethod
    def __class_getitem__(cls, key):
        actual_model_run = Registry.create_model_run()
        return actual_model_run.__class_getitem__(key)


class ModelRun(metaclass=ModelRunMeta):
    pass
