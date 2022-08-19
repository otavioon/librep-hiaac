from functools import partial, wraps
from importlib import import_module
from inspect import isclass, isfunction, ismethod
from pkgutil import walk_packages

def load_submodules(basemod, load_all: bool = True, exclude_pattern: str = "(.*[tT]est.*)|(_.*)"):
    """
    Traverse the source of the module structure starting with module `basemod`, loading all packages plus all files if
    `load_all` is True, excluding anything whose name matches `exclude_pattern`.
    """
    submodules = []
    err_mod: List[str] = []
    for importer, name, is_pkg in walk_packages(
        basemod.__path__, prefix=basemod.__name__ + ".", onerror=err_mod.append
    ):
        if (is_pkg or load_all) and name not in sys.modules and match(exclude_pattern, name) is None:
            try:
                mod = import_module(name)
                importer.find_module(name).load_module(name)  # type: ignore
                submodules.append(mod)
            except OptionalImportError:
                pass  # could not import the optional deps., they are ignored

    return submodules,
