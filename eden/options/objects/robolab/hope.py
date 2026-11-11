"""RoboLab hope object options (auto-generated from object_catalog.json).

Assets converted from the RoboLab USD library to URDF; hosted at
``https://huggingface.co/datasets/Kashu7100/eden_robolab``.
"""

from __future__ import annotations

from typing import ClassVar

from eden.options.entities import MetadataOptions

from eden.options.objects.robolab.base import ROBOLAB_REGISTRY, RoboLabObjectOptions


@ROBOLAB_REGISTRY.register(name="hope/bbq_sauce_bottle")
class RoboLabHopeBbqSauceBottle(RoboLabObjectOptions):
    """The bottle features a curved shape with a brownish-red color, suggesting its contents. It has a label that reads 'Tangy BBQ Sauce' in vibrant colors, with a star graphic, and a matching orange cap."""

    file: str = "objects/hope/bbq_sauce_bottle/bbq_sauce_bottle.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="condiment")


@ROBOLAB_REGISTRY.register(name="hope/butter")
class RoboLabHopeButter(RoboLabObjectOptions):
    """The object is a rectangular package of butter featuring a vibrant orange and red color scheme. It has images of a cow and text that reads 'FARM FRESH BUTTER' and 'UNSALTED'."""

    file: str = "objects/hope/butter/butter.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/canned_mushrooms")
class RoboLabHopeCannedMushrooms(RoboLabObjectOptions):
    """The can features a vivid design with a purple and orange color scheme, prominently displaying the text 'Sliced Mushrooms'. It has a metallic finish and contains nutritional information on the label."""

    file: str = "objects/hope/canned_mushrooms/canned_mushrooms.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="vegetable")


@ROBOLAB_REGISTRY.register(name="hope/canned_peaches")
class RoboLabHopeCannedPeaches(RoboLabObjectOptions):
    """The can features a bright orange and yellow design with images of sliced peaches. It includes nutritional information and text indicating the product is 'Sliced Peaches in Their Own Natural Juices.'"""

    file: str = "objects/hope/canned_peaches/canned_peaches.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="fruit")


@ROBOLAB_REGISTRY.register(name="hope/canned_tuna")
class RoboLabHopeCannedTuna(RoboLabObjectOptions):
    """The can is cylindrical with a colorful label featuring cartoon fish illustrations in shades of blue and green. The label includes text indicating 'Tuna Fish' and nutritional information on the back."""

    file: str = "objects/hope/canned_tuna/canned_tuna.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/chocolate_pudding_mix")
class RoboLabHopeChocolatePuddingMix(RoboLabObjectOptions):
    """The packaging features a rich brown color with the words 'Instant Chocolate Pudding' prominently displayed in white and red font. The boxes have images of creamy chocolate pudding and nutritional information on one side."""

    file: str = "objects/hope/chocolate_pudding_mix/chocolate_pudding_mix.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/corn_can")
class RoboLabHopeCornCan(RoboLabObjectOptions):
    """The can is cylindrical with a vibrant yellow and green design, prominently featuring the word 'CORN' in bold letters. One side displays nutritional facts and ingredients, printed in a clear, easy-to-read format."""

    file: str = "objects/hope/corn_can/corn_can.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/cream_cheese")
class RoboLabHopeCreamCheese(RoboLabObjectOptions):
    """The object is a rectangular block of cream cheese, predominantly white with a smooth texture. It has blue and red text on the packaging, indicating the brand name and descriptions like 'Fresh Taste'."""

    file: str = "objects/hope/cream_cheese/cream_cheese.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/granola_bars")
class RoboLabHopeGranolaBars(RoboLabObjectOptions):
    """The granola bars are packaged in a colorful yellow box decorated with red and white text. The box displays images of oats and raisins, emphasizing the natural ingredients inside."""

    file: str = "objects/hope/granola_bars/granola_bars.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/green_beans_can")
class RoboLabHopeGreenBeansCan(RoboLabObjectOptions):
    """The can features a vibrant green design with illustrations of green beans and leaves. The label includes the text 'Green Beans' prominently, along with nutritional information on one side."""

    file: str = "objects/hope/green_beans_can/green_beans_can.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/ketchup_bottle")
class RoboLabHopeKetchupBottle(RoboLabObjectOptions):
    """The bottle is made of plastic with a smooth, rounded body and a squeeze top. It features a vibrant orange-red color and a label that reads 'Fancy Tomato Ketchup' in bold text."""

    file: str = "objects/hope/ketchup_bottle/ketchup_bottle.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="container")


@ROBOLAB_REGISTRY.register(name="hope/macaroni_and_cheese")
class RoboLabHopeMacaroniAndCheese(RoboLabObjectOptions):
    """The packaging is a bright orange box with bold, playful text declaring 'SO CHEESY!' and 'MACARONI AND CHEESE.' It prominently features images of macaroni and cheese, suggesting a creamy texture and rich flavor."""

    file: str = "objects/hope/macaroni_and_cheese/macaroni_and_cheese.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/mayonnaise_bottle")
class RoboLabHopeMayonnaiseBottle(RoboLabObjectOptions):
    """The bottle is a squeeze-type with a slightly curved shape, finished in a light beige color. It features a blue cap and a label that prominently displays the word 'MAYO' along with graphics of mayonnaise."""

    file: str = "objects/hope/mayonnaise_bottle/mayonnaise_bottle.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="condiment")


@ROBOLAB_REGISTRY.register(name="hope/milk_carton")
class RoboLabHopeMilkCarton(RoboLabObjectOptions):
    """The carton is predominantly orange with green and white accents, displaying the word 'Milk' prominently on all sides. It features a cartoon cow on the front and nutritional information on one side."""

    file: str = "objects/hope/milk_carton/milk_carton.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/mustard_bottle")
class RoboLabHopeMustardBottle(RoboLabObjectOptions):
    """The bottle is a vibrant yellow color with a glossy finish. It features a green label with white text that reads 'Spicy Yellow Mustard'."""

    file: str = "objects/hope/mustard_bottle/mustard_bottle.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="condiment")


@ROBOLAB_REGISTRY.register(name="hope/oatmeal_raisin_cookies")
class RoboLabHopeOatmealRaisinCookies(RoboLabObjectOptions):
    """The packaging features a vibrant red and white color scheme with images of cookies prominently displayed. The front of the box is labeled 'Oatmeal Raisin Cookies' in bold letters, accompanied by graphics of the cookies themselves."""

    file: str = "objects/hope/oatmeal_raisin_cookies/oatmeal_raisin_cookies.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/orange_juice_carton")
class RoboLabHopeOrangeJuiceCarton(RoboLabObjectOptions):
    """The carton is primarily orange and yellow, featuring large images of sliced oranges. It has text that reads 'ORANGE JUICE' and indicates 'NO PULP' on the front."""

    file: str = "objects/hope/orange_juice_carton/orange_juice_carton.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/parmesan_cheese_canister")
class RoboLabHopeParmesanCheeseCanister(RoboLabObjectOptions):
    """The canister is cylindrical with a green and yellow striped design, featuring text that indicates it contains grated Parmesan cheese. It has a lid on top and nutritional facts printed on the side."""

    file: str = "objects/hope/parmesan_cheese_canister/parmesan_cheese_canister.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/peas_and_carrots")
class RoboLabHopePeasAndCarrots(RoboLabObjectOptions):
    """The object is a cylindrical can featuring a colorful label with green polka dots and orange accents. It has text that reads 'Peas & Carrots' on the front and nutritional information on the back."""

    file: str = "objects/hope/peas_and_carrots/peas_and_carrots.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/pineapple_slices_can")
class RoboLabHopePineappleSlicesCan(RoboLabObjectOptions):
    """The can features a bright blue background with yellow pineapple slice illustrations. It has a label with red text indicating 'Pineapple Slices' and nutritional information on the side."""

    file: str = "objects/hope/pineapple_slices_can/pineapple_slices_can.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/pitted_cherries")
class RoboLabHopePittedCherries(RoboLabObjectOptions):
    """The can is cylindrical with a purple label featuring bright red cherries and white text. The lid is metallic, and the nutritional facts are displayed on one side of the can."""

    file: str = "objects/hope/pitted_cherries/pitted_cherries.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/popcorn_box")
class RoboLabHopePopcornBox(RoboLabObjectOptions):
    """The box is bright blue and features a playful design with yellow and white flowers. It prominently displays the word 'POPCORN' in bold red letters on the front."""

    file: str = "objects/hope/popcorn_box/popcorn_box.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/raisin_box")
class RoboLabHopeRaisinBox(RoboLabObjectOptions):
    """The box is brightly colored with an orange background and features a smiling sun graphic. It prominently displays the word 'RAISINS' along with illustrations of grapes."""

    file: str = "objects/hope/raisin_box/raisin_box.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/ranch_dressing")
class RoboLabHopeRanchDressing(RoboLabObjectOptions):
    """The bottle has a smooth, rounded shape with a prominent green cap. It features a label that displays the text 'Creamy Ranch Dressing' alongside images of vegetables."""

    file: str = "objects/hope/ranch_dressing/ranch_dressing.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="condiment")


@ROBOLAB_REGISTRY.register(name="hope/spaghetti")
class RoboLabHopeSpaghetti(RoboLabObjectOptions):
    """The packaging is a rectangular box with a vibrant red and green design. It prominently features the word 'SPAGHETTI' in bold letters along with the label 'Organic Whole Wheat'."""

    file: str = "objects/hope/spaghetti/spaghetti.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/tomato_sauce_can")
class RoboLabHopeTomatoSauceCan(RoboLabObjectOptions):
    """The can is cylindrical with a glossy red and green label featuring the words 'TOMATO SAUCE' prominently displayed. It is adorned with images of tomatoes and includes nutrition facts and serving suggestions on one side."""

    file: str = "objects/hope/tomato_sauce_can/tomato_sauce_can.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="hope/yogurt_cup")
class RoboLabHopeYogurtCup(RoboLabObjectOptions):
    """The yogurt cup is cylindrical and features a vibrant orange color with a label indicating its strawberry flavor. It has a plastic lid on top and nutrition facts printed on the side."""

    file: str = "objects/hope/yogurt_cup/yogurt_cup.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")
