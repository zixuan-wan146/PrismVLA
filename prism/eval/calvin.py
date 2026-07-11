from __future__ import annotations

import sys as _sys

from prism.eval import calvin_action_protocol as action_protocol
from prism.eval import calvin_config as config
from prism.eval import calvin_eval_summary as eval_summary
from prism.eval import calvin_history as history
from prism.eval import calvin_observation as observation
from prism.eval import calvin_request_builder as request_builder
from prism.eval import calvin_runner as runner
from prism.eval import calvin_spec as spec
from prism.eval.calvin_action_protocol import *  # noqa: F403
from prism.eval.calvin_config import *  # noqa: F403
from prism.eval.calvin_eval_summary import *  # noqa: F403
from prism.eval.calvin_history import *  # noqa: F403
from prism.eval.calvin_observation import *  # noqa: F403
from prism.eval.calvin_request_builder import *  # noqa: F403
from prism.eval.calvin_runner import *  # noqa: F403
from prism.eval.calvin_spec import *  # noqa: F403

_ALIASES = {
    "spec": spec,
    "action_protocol": action_protocol,
    "observation": observation,
    "request_builder": request_builder,
    "history": history,
    "config": config,
    "eval_summary": eval_summary,
    "runner": runner,
}
for _name, _module in _ALIASES.items():
    _sys.modules[f"{__name__}.{_name}"] = _module

__all__ = [name for name in globals() if not name.startswith("_")]
