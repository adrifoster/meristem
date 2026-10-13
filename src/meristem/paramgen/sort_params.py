"""Utility function for sorting Parameter list to resolve dependencies"""

from __future__ import annotations

from collections import defaultdict, deque

from .parameter import Parameter, ScaleFromRootParameter


def sort_params(params: list[Parameter]) -> list[Parameter]:
    """Return params reordered so every root_param is written before its dependents.

    So far, only ScaleFromRootParameters introduce ordering constraints. All other
    parameter types are treated as having no dependencies.

    If a root_param is not present in the parameter list (i.e. it is not
    being calibrated), no ordering constraint is added for it. The dataset
    value is assumed to be whatever was last written.

    Args:
        params (list[Parameter]): Unsorted parameter list, as constructed from the input
        file

    Raises:
        ValueError:  If a cycle is detected among ScaleFromRootParameter dependencies
        (e.g. A scales from B and B scales from A).

    Returns:
        list[Parameter]: Topologically sorted parameter list.
    """
    spec_name_to_param: dict[str, Parameter] = {p.spec.name: p for p in params}

    # map from dataset variable name to param that writes it
    var_to_param: dict[str, Parameter] = {}
    for p in params:
        if p.spec.base_params:
            for var in p.spec.base_params:
                var_to_param[var] = p
        else:
            var_to_param[p.spec.name] = p

    # build adjacency list and in-degree count.
    # edge: root_param to dependent (root must come first)
    dependents: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {p.spec.name: 0 for p in params}

    for param in params:
        if isinstance(param, ScaleFromRootParameter):
            root = param.spec.root_param
            if root in var_to_param:
                root_param_name = var_to_param[root].spec.name
                if root_param_name != param.spec.name:
                    dependents[root_param_name].append(param.spec.name)
                    in_degree[param.spec.name] += 1

    # Kahn's algorithm: seed with all zero-in-degree nodes in original order
    queue: deque[str] = deque(
        p.spec.name for p in params if in_degree[p.spec.name] == 0
    )
    sorted_names: list[str] = []

    while queue:
        name = queue.popleft()
        sorted_names.append(name)
        for dependent in dependents[name]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(sorted_names) != len(params):
        cycle_members = [p.spec.name for p in params if p.spec.name not in sorted_names]
        raise ValueError(
            f"Cycle detected among ScaleFromRootParameter dependencies: "
            f"{cycle_members}. Each root_param must not depend on its own "
            "dependents."
        )

    return [spec_name_to_param[name] for name in sorted_names]
