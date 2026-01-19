"""Other usefull utility functions."""
from typing import Any, List, Optional, Union


def nullable_union_list_object_to_list(param: Optional[Union[List[Any], Any]]) -> List[Any]:
    if isinstance(param, list):
        return param
    elif param is None:
        return []
    else:
        return [param]