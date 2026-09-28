"""Launcher so the dashboard can be started from the installed package.

``streamlit run`` takes a *file path*, which normally means the container has to
carry a copy of the source tree next to the installed distribution.  Two copies
of the same code in one image is a reproducibility hazard: whichever one is
first on ``sys.path`` wins, and it is not necessarily the version-stamped
installed one.

Resolving the path through ``importlib.util.find_spec`` removes the second copy.
``find_spec`` locates the module without importing it, which matters here
because :mod:`quantlab.dashboard.app` calls ``main()`` at import time (Streamlit
executes a script rather than importing it, so the module has to work that way).
Importing it to find its filename would start a second Streamlit app inside the
launcher.

Usage::

    python -m quantlab.dashboard --server.port=8501
"""

from __future__ import annotations

import importlib.util
import os
import sys


def resolve_app_path() -> str:
    """Filesystem path of the Streamlit entry module, without importing it."""
    spec = importlib.util.find_spec("quantlab.dashboard.app")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "quantlab.dashboard.app could not be located; is quantlab installed correctly?"
        )
    return spec.origin


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        from streamlit.web import cli as stcli
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "streamlit is not installed; install the 'dashboard' extra: pip install '.[dashboard]'"
        ) from exc

    sys.argv = ["streamlit", "run", resolve_app_path(), *args]
    # ``stcli.main`` is a click command: it raises SystemExit rather than
    # returning.  Let it propagate so the exit code is Streamlit's own.
    stcli.main()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    raise SystemExit(main())
