"""Repo-root launcher: ``python infer_api.py``.

Implementation lives in ``tb6r5_policy_infer.infer_api`` so pip install
exposes ``tb6r5-infer-api`` without importing lerobot/torch.
"""

from tb6r5_policy_infer.infer_api import app, main

__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
