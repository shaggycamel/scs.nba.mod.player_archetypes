"""Loaders for the Parquet snapshot the role model was built against.

Set `PLAYER_ARCHETYPES_DATA` to the directory containing `nba/`, `statyx/` and
`util/`, or pass a path explicitly to any loader. The data is intentionally not
part of this repository.
"""

from player_archetypes.data import ids, live, load

__all__ = ["ids", "live", "load"]
