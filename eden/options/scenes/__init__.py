"""Scene configuration option presets."""

from .empty import BimanualEmptySceneOptions, EmptySceneOptions
from .lbm_eval import LBMEvalSceneOptions
from .tabletop import KitchenCountertopSceneOptions, TabletopSceneOptions

# Short aliases
Empty = EmptySceneOptions
BimanualEmpty = BimanualEmptySceneOptions
Tabletop = TabletopSceneOptions
KitchenCountertop = KitchenCountertopSceneOptions
LBMEval = LBMEvalSceneOptions

__all__ = [
    "EmptySceneOptions",
    "BimanualEmptySceneOptions",
    "TabletopSceneOptions",
    "KitchenCountertopSceneOptions",
    "LBMEvalSceneOptions",
    "Empty",
    "BimanualEmpty",
    "Tabletop",
    "KitchenCountertop",
    "LBMEval",
]
