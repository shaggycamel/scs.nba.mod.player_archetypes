"""Loaders over the public database views.

The connection defaults to `jetson-read`; set `PLAYER_ARCHETYPES_DB_CON` to use
another `scs_hub` credentials section. There is no Parquet snapshot.
"""

from player_archetypes.data import ids, live, load

__all__ = ["ids", "live", "load"]
