"""Utility functions for parsing."""

import importlib
import itertools
import logging
import pkgutil
import types
from typing import Any, Callable, Dict, List, Optional


def list_all_submodules(*packages: types.ModuleType) -> List[types.ModuleType]:
    """List all submodules in `packages` recursively."""
    # Resolve one or more packages
    if len(packages) > 1:
        return list(
            itertools.chain.from_iterable(map(list_all_submodules, packages))
            )
    else:
        assert len(packages) == 1, "No packages specified"
        package = packages[0]

    submodules: List[types.ModuleType] = []
    for _, module_name, is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        module = __import__(module_name, fromlist="dummylist")
        submodules.append(module)
        if is_pkg:
            submodules.extend(list_all_submodules(module))

    return submodules


def is_wafer_ad_module(obj: types.ModuleType) -> bool:
    """Return whether `obj` is a module in wafer-ad."""
    return isinstance(obj, types.ModuleType) and obj.__name__.startswith(
        "wafer_ad."
    )
    
    
def is_wafer_ad_class(obj: type) -> bool:
    """Return whether `obj` is a class in wafer-ad."""
    return isinstance(obj, type) \
        and obj.__module__.startswith("wafer_ad.")


def get_wafer_ad_classes(module: types.ModuleType) -> Dict[str, type]:
    """Return a lookup of all wafer-ad class names in `module`."""
    if not is_wafer_ad_module(module):
        logging.warning("%s is not a wafer-ad module", module)
        return {}
    classes = {
        key: val for key, val in module.__dict__.items()
        if is_wafer_ad_class(val)
    }
    return classes


def get_all_wafer_ad_classes(
        *packages: types.ModuleType
) -> Dict[str, type]:
    """List all wafer-ad classes in `packages`."""
    submodules = list_all_submodules(*packages)
    classes: Dict[str, type] = {}
    for submodule in submodules:
        new_classes = get_wafer_ad_classes(submodule)
        for key in new_classes:
            if key in classes and classes[key] != new_classes[key]:
                logging.warning(
                    "Class %s found in both %s and "
                    "%s. Keeping first instance. "
                    "Consider renaming.",
                    key,
                    classes[key],
                    new_classes[key]
                )
        classes.update(new_classes)

    return classes


def str_to_class(module_name: str, class_name: str) -> Any:
    # load the module, will raise ImportError if module cannot be loaded
    module = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    cls = getattr(module, class_name)
    return cls


def traverse_and_apply(
    obj: Any, fn: Callable, fn_kwargs: Optional[Dict[str, Any]] = None
) -> Any:
    """Apply `fn` to all elements in `obj`, resulting in same structure."""
    if isinstance(obj, (list, tuple)):
        return [traverse_and_apply(elem, fn, fn_kwargs) for elem in obj]
    elif isinstance(obj, dict):
        return {
            key: traverse_and_apply(val, fn, fn_kwargs)
            for key, val in obj.items()
            }
    else:
        if fn_kwargs is None:
            fn_kwargs = {}
        return fn(obj, **fn_kwargs)
