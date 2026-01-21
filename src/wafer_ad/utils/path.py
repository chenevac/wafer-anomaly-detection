


import os
from wafer_ad.utils.constant import PROJECT_ROOT


def resolve_path(path: str) -> str:
    path_resolved = (path.replace("$waferad", str(PROJECT_ROOT)).
        replace("$WAFERAD", str(PROJECT_ROOT)).
        replace("${waferad}", str(PROJECT_ROOT)).
        replace("${WAFERAD}", str(PROJECT_ROOT)))
    
    path_resolved=path_resolved.replace("/", "\\") if os.name == "nt" else path_resolved.replace("\\", "/")
    return path_resolved