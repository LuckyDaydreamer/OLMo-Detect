from __future__ import annotations

from typing import Iterable, TypeVar

T = TypeVar("T")

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


def progress(iterable: Iterable[T], **kwargs) -> Iterable[T]:
    if _tqdm is None:
        return iterable
    return _tqdm(iterable, **kwargs)

