# (C) 2024 Irreducible Inc.

import functools
import pathlib
import types
from typing import Callable, Dict, List, Tuple

import pytest


def pytest_pycollect_makemodule(module_path: pathlib.Path, parent) -> pytest.Module:
    """
    Customizes the pytest collection process for Python modules, by handling
    @pytest.mark.parametrize_hypothesis decorator.

    Args:
        module_path (pathlib.Path):
        parent:

    Returns:
        pytest.Module: Created module.
    """
    # logic copied from pytest: begin
    if module_path.name == "__init__.py":
        pkg: pytest.Package = pytest.Package.from_parent(parent, path=module_path)
        return pkg
    mod: pytest.Module = pytest.Module.from_parent(parent, path=module_path)
    # logic copied from pytest: end

    # new logic: begin
    handle_parametrize_hypothesis(mod)
    # new logic: end
    return mod


def handle_parametrize_hypothesis(mod: pytest.Module) -> None:
    """
    Perform set of actions for every function marked with @pytest.mark.parametrize_hypothesis in
    provided pytest module. The actions:
     - removes original function from the module,
     - for every kwarg in the decorator, creates a new function in the module that:
       - is a copy of the original function,
       - is decorated with provided list of hypothesis decorators (value),
       - is decorated with @pytest.mark.[key name],

    Args:
        mod (pytest.Module): pytest module
    """

    test_func_by_name: Dict[str, Callable] = {}

    # Iterate over all objects in the module's namespace and find functions
    # decorated with @pytest.mark.parametrize_hypothesis
    for obj_name, obj in getattr(mod.obj, "__dict__", {}).items():
        # find all functions decorated with @pytest.mark.parametrize_hypothesis
        if callable(obj) and hasattr(obj, "pytestmark"):
            if any(mark.name == "parametrize_hypothesis" for mark in obj.pytestmark):
                test_func_by_name[obj_name] = obj

    # For every test function
    for test_func_name, test_func in test_func_by_name.items():
        # Remove original test function from the module
        delattr(mod.obj, test_func_name)

        # Get @pytest.mark.parametrize_hypothesis data
        parametrize_hypothesis: pytest.Mark = next(
            (m for m in getattr(test_func, "pytestmark", []) if m.name == "parametrize_hypothesis"),
        )

        # Validate arguments in @pytest.mark.parametrize_hypothesis
        if len(parametrize_hypothesis.args) > 0:
            raise ValueError(
                f"@pytest.mark.parametrize_hypothesis for '{mod.name}.{test_func_name}' is not correct. "
                + "It contains non key value arguments, which are not supported."
            )

        # Go trough every key value argument in @pytest.mark.parametrize_hypothesis
        for mark_name, decorator_list in parametrize_hypothesis.kwargs.items():
            # Validate
            if not isinstance(decorator_list, (list, tuple, set)):
                raise ValueError(
                    f"@pytest.mark.parametrize_hypothesis for '{mod.name}.{test_func_name}' is not correct. "
                    + f"Value for {mark_name} key is not a list: {decorator_list}"
                )
            if not all(callable(decorator) for decorator in decorator_list):
                raise ValueError(
                    f"@pytest.mark.parametrize_hypothesis for '{mod.name}.{test_func_name}' is not correct. "
                    + f"At least one element on a list for {mark_name} key is not a function: {decorator_list}"
                )

            # Create new test function
            new_test_func_name = test_func_name + "_" + mark_name
            new_test_func = copy_test_func_and_apply_decorators(
                test_func,
                new_test_func_name,
                decorator_list,
            )

            # Decorate new test function with @pytest.mark.[kwarg key]
            marker_decorator = getattr(pytest.mark, mark_name)
            new_test_func = marker_decorator(new_test_func)

            # Add new test function to the module
            setattr(mod.obj, new_test_func_name, new_test_func)


def copy_test_func_and_apply_decorators(
    test_func: Callable,
    new_name: str,
    decorator_list: List | Tuple | set,
) -> Callable:
    """
    Copies a test function and decorates it with provided list of decorators.

    Args:
        test_func (Callable): The original test function to be copied.
        new_name (str): The name for the new function
        decorator_list (List[Callable]): list of decorators to apply to new function

    Returns:
        Callable: The new test function.
    """

    # Create a new function from the original one, using the provided new name and retaining
    # the original function's data.
    new_test_func: Callable = types.FunctionType(
        code=test_func.__code__,
        globals=test_func.__globals__,
        name=new_name,
        argdefs=test_func.__defaults__,
        closure=test_func.__closure__,
    )

    # Update the new function to preserve metadata and documentation from the original
    new_test_func = functools.update_wrapper(new_test_func, test_func)
    for decorator in decorator_list:
        if not callable(decorator):
            raise ValueError(f"failed to create test function {new_name}, this is not decorator: {decorator}")
        new_test_func = decorator(new_test_func)

    return new_test_func
