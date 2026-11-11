"""Objaverse object options."""

from __future__ import annotations

from typing import ClassVar

from eden.options.entities import EntityOptions
from eden.utils.registry import VariantRegistry

OBJAVERSE_OBJECTS_REGISTRY = VariantRegistry("OBJAVERSE_OBJECTS")


class ObjaverseObjectOptions(EntityOptions):
    """Base options for Objaverse mesh assets hosted on HuggingFace.

    Concrete subclasses are just ``pass`` bodies decorated with the registry's
    ``@register(base_name=..., variant_list=[...])`` call. The decorator binds
    the directory name and variant catalog onto the class, and this base's
    ``__init__`` materializes ``file`` from those + the requested variant.
    """

    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)

    registry: str = "Kashu7100"
    dataset: str = "eden_objaverse"
    assets_subdir: ClassVar[str] = "objaverse"

    _file_template: ClassVar[str] = "{base_name}/{base_name}_{variant}/model.xml"
    _base_name: ClassVar[str] = ""
    _default_variant: ClassVar[int] = 0

    metadata: ClassVar[dict] = {}

    def __init__(self, variant: int | None = None, **kwargs):
        if variant is None:
            variant = self._default_variant
        kwargs.setdefault("file", self._file_template.format(base_name=self._base_name, variant=variant))
        super().__init__(**kwargs)


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="alcohol", variant_list=[0, 1, 2, 4, 5, 7, 9, 10, 11, 12, 13, 14])
class ObjaverseAlcohol(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="apple", variant_list=list(range(16)) + [17, 18, 19, 20, 21, 22])
class ObjaverseApple(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="avocado", variant_list=[0, 1, 2, 4, 5, 6, 8])
class ObjaverseAvocado(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="bagel", variant_list=list(range(8)) + [9])
class ObjaverseBagel(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="bagged_food", variant_list=[0, 3, 4, 5, 7, 8, 9, 11, 14])
class ObjaverseBaggedFood(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="baguette", variant_list=[0, 1, 2, 4, 5, 6])
class ObjaverseBaguette(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="banana",
    variant_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24],
)
class ObjaverseBanana(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="bar", variant_list=[0, 2, 3, 4, 6, 7, 8, 9, 10, 11])
class ObjaverseBar(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="bar_soap", variant_list=[0, 1, 3, 4])
class ObjaverseBarSoap(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="beer", variant_list=[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19]
)
class ObjaverseBeer(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="bell_pepper", variant_list=[0, 1, 2, 3])
class ObjaverseBellPepper(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="bottled_drink", variant_list=[0, 1, 2, 4, 5])
class ObjaverseBottledDrink(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="bottled_water", variant_list=[1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17]
)
class ObjaverseBottledWater(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="bowl", variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
)
class ObjaverseBowl(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="boxed_drink", variant_list=[0, 1, 2, 3, 4, 5, 7])
class ObjaverseBoxedDrink(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="boxed_food", variant_list=[0, 1, 2, 4, 6, 7, 8, 9, 10, 11])
class ObjaverseBoxedFood(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="bread", variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21]
)
class ObjaverseBread(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="broccoli", variant_list=[0, 1, 3])
class ObjaverseBroccoli(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="cake", variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
class ObjaverseCake(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="can", variant_list=[0, 2, 3, 4, 6, 8, 9, 11, 12, 13, 15, 16, 17, 18])
class ObjaverseCan(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="candle", variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
)
class ObjaverseCandle(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="canned_food", variant_list=[0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19]
)
class ObjaverseCannedFood(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="carrot", variant_list=[0, 1, 2, 3, 4, 6, 7])
class ObjaverseCarrot(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="cereal", variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
class ObjaverseCereal(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="cheese", variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8])
class ObjaverseCheese(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="chips", variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]
)
class ObjaverseChips(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="chocolate", variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
)
class ObjaverseChocolate(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="coffee_cup", variant_list=[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21]
)
class ObjaverseCoffeeCup(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="condiment", variant_list=[0, 2, 3])
class ObjaverseCondiment(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="corn", variant_list=[0, 1, 3, 4])
class ObjaverseCorn(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="croissant", variant_list=[0, 1, 2, 3, 4, 5, 6, 8, 9, 10])
class ObjaverseCroissant(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="cucumber", variant_list=[0, 1, 2, 3, 4, 5])
class ObjaverseCucumber(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="cup", variant_list=[2, 3, 4, 5])
class ObjaverseCup(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="cupcake", variant_list=[2, 3, 4, 5, 6, 7, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20]
)
class ObjaverseCupcake(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="cutting_board", variant_list=[0, 1, 2, 4, 5, 7, 9, 11, 12, 13, 15, 16])
class ObjaverseCuttingBoard(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="donut", variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
)
class ObjaverseDonut(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="egg", variant_list=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
class ObjaverseEgg(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="eggplant", variant_list=[0, 1, 2, 3, 4])
class ObjaverseEggplant(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="fish", variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
class ObjaverseFish(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="fork", variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
class ObjaverseFork(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="garlic", variant_list=[0, 1, 2, 4, 5])
class ObjaverseGarlic(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="hot_dog", variant_list=[0, 1, 2, 3, 4, 5, 6])
class ObjaverseHotDog(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="jam", variant_list=[0, 1, 2, 3, 4, 5, 6])
class ObjaverseJam(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="jug", variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8])
class ObjaverseJug(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="ketchup", variant_list=[0, 1, 2, 3, 4, 7, 8, 9, 10, 11])
class ObjaverseKetchup(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="kettle",
    variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28],
)
class ObjaverseKettle(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="kiwi", variant_list=[0, 1, 2, 4, 5, 6, 7])
class ObjaverseKiwi(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="knife",
    variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
)
class ObjaverseKnife(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="ladle", variant_list=[0, 1, 2, 3, 4])
class ObjaverseLadle(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="lemon", variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
)
class ObjaverseLemon(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="lime", variant_list=[0, 1, 2, 3])
class ObjaverseLime(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="mango", variant_list=[1, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19]
)
class ObjaverseMango(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="milk", variant_list=[0, 1, 3, 4, 5, 7, 9, 10])
class ObjaverseMilk(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="mug", variant_list=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
)
class ObjaverseMug(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="mushroom", variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
)
class ObjaverseMushroom(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="onion", variant_list=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10])
class ObjaverseOnion(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="orange", variant_list=[0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])
class ObjaverseOrange(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="pan", variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24]
)
class ObjaversePan(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="peach", variant_list=[0, 1, 2, 3, 4, 5, 6])
class ObjaversePeach(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="pear", variant_list=[0, 1, 2, 3, 5, 6, 8, 10, 11, 12, 13, 15, 16, 17, 18, 19]
)
class ObjaversePear(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="plate", variant_list=[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19, 20]
)
class ObjaversePlate(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="potato", variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19]
)
class ObjaversePotato(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="rolling_pin", variant_list=[0, 1, 2, 3, 4, 5])
class ObjaverseRollingPin(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="scissors", variant_list=[0, 1, 3, 4, 6, 8, 9, 10, 11, 12, 13])
class ObjaverseScissors(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="shaker", variant_list=[0, 1, 2, 3])
class ObjaverseShaker(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="soap_dispenser", variant_list=[0, 1, 2, 3, 4, 6])
class ObjaverseSoapDispenser(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="spatula", variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
class ObjaverseSpatula(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="sponge", variant_list=[0, 1, 2, 3, 4, 5, 6, 8])
class ObjaverseSponge(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="spoon",
    variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
)
class ObjaverseSpoon(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="spray", variant_list=[0, 1, 2])
class ObjaverseSpray(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="squash", variant_list=[0, 1, 3, 5, 7, 8, 13, 15, 16])
class ObjaverseSquash(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="steak", variant_list=[2, 3, 4, 5, 6, 8, 9, 10, 11, 14, 15])
class ObjaverseSteak(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="sweet_potato", variant_list=[1, 2, 3])
class ObjaverseSweetPotato(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="tangerine", variant_list=[0, 1, 2, 3, 4, 5, 6])
class ObjaverseTangerine(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="teapot", variant_list=[0, 3, 4, 5, 6, 7, 10, 11, 13, 14, 15])
class ObjaverseTeapot(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="tomato", variant_list=[0, 1, 2, 3, 5, 6, 7, 8])
class ObjaverseTomato(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="tray", variant_list=[0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12])
class ObjaverseTray(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="waffle", variant_list=[0, 1, 3, 4])
class ObjaverseWaffle(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(
    base_name="water_bottle",
    variant_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23],
)
class ObjaverseWaterBottle(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="wine", variant_list=[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13])
class ObjaverseWine(ObjaverseObjectOptions):
    pass


@OBJAVERSE_OBJECTS_REGISTRY.register(base_name="yogurt", variant_list=[1, 2, 3])
class ObjaverseYogurt(ObjaverseObjectOptions):
    pass
