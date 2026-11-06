"""Point-in-time player archetypes for basketball player-games.

The model replaces listed position with a soft, data-driven representation of
what a player actually does on the floor. Every assignment is fitted only on
games before a cutoff, so an archetype is knowable before tip-off.

    from player_archetypes import fit
    model = fit(design, feature_names, k=10, cutoff=date(2024, 10, 1))
    assigned = model.assign(design)

`player_archetypes.cluster` is the model; `player_archetypes.features` builds
its inputs. The parquet loaders live in `player_archetypes.data` so the core
model can be used without the source snapshot.
"""

from player_archetypes import cluster, features
from player_archetypes.cluster import ArchetypeModel, RoleModel, fit

__all__ = ["ArchetypeModel", "RoleModel", "cluster", "features", "fit"]
