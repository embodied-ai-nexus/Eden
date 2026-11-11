"""RoboLab ycb object options (auto-generated from object_catalog.json).

Assets converted from the RoboLab USD library to URDF; hosted at
``https://huggingface.co/datasets/Kashu7100/eden_robolab``.
"""

from __future__ import annotations

from typing import ClassVar

from eden.options.entities import MetadataOptions

from eden.options.objects.robolab.base import ROBOLAB_REGISTRY, RoboLabObjectOptions


@ROBOLAB_REGISTRY.register(name="ycb/banana")
class RoboLabYcbBanana(RoboLabObjectOptions):
    """These bananas are long and curved, with a smooth yellow skin and slightly green tips. They are commonly known for their sweet taste and are often eaten raw or used in various dishes."""

    file: str = "objects/ycb/banana/banana.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="fruit")


@ROBOLAB_REGISTRY.register(name="ycb/bowl")
class RoboLabYcbBowl(RoboLabObjectOptions):
    """The bowls are round with a smooth, glossy surface and a vibrant red color. They feature a speckled design and a slightly raised rim, adding a decorative touch."""

    file: str = "objects/ycb/bowl/bowl.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="kitchenware")


@ROBOLAB_REGISTRY.register(name="ycb/bowl2")
class RoboLabYcbBowl2(RoboLabObjectOptions):
    """The bowls are round with a smooth, glossy surface and a vibrant red color. They feature a speckled design and a slightly raised rim, adding a decorative touch."""

    file: str = "objects/ycb/bowl2/bowl2.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="kitchenware")


@ROBOLAB_REGISTRY.register(name="ycb/brick")
class RoboLabYcbBrick(RoboLabObjectOptions):
    """The object is a rectangular brick with a deep red-brown color. It features three circular indentations on one side, giving it a textured appearance."""

    file: str = "objects/ycb/brick/brick.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="block")


@ROBOLAB_REGISTRY.register(name="ycb/cheez_it")
class RoboLabYcbCheezIt(RoboLabObjectOptions):
    """The package is predominantly red with white and yellow accents, featuring bold text that reads 'Cheez-It' and 'Original.' It showcases images of the square crackers and includes nutritional information and serving suggestions on the sides."""

    file: str = "objects/ycb/cheez_it/cheez_it.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="ycb/chocolate_pudding")
class RoboLabYcbChocolatePudding(RoboLabObjectOptions):
    """The package is primarily red and brown with the prominent text 'JELL-O' in bold letters. There are nutritional facts and ingredient details printed on the sides, and the overall shape is a rectangular box."""

    file: str = "objects/ycb/chocolate_pudding/chocolate_pudding.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="ycb/clamp")
class RoboLabYcbClamp(RoboLabObjectOptions):
    """The clamp features a sturdy black body with a bright orange tip for added visibility. Its ergonomic design includes grips that provide comfort during use."""

    file: str = "objects/ycb/clamp/clamp.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="tool")


@ROBOLAB_REGISTRY.register(name="ycb/coffee_can")
class RoboLabYcbCoffeeCan(RoboLabObjectOptions):
    """The can is cylindrical with a shiny blue exterior, featuring the text 'Master Chef' prominently displayed on the front. It has white and yellow accents and contains ground coffee, with brewing instructions printed on the label."""

    file: str = "objects/ycb/coffee_can/coffee_can.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="ycb/cordless_drill")
class RoboLabYcbCordlessDrill(RoboLabObjectOptions):
    """The cordless drill features a sleek design with a predominantly orange body and black grip. It includes a chuck at the front for securing drill bits and has a battery pack attached to the base."""

    file: str = "objects/ycb/cordless_drill/cordless_drill.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="tool")


@ROBOLAB_REGISTRY.register(name="ycb/dry_erase_marker")
class RoboLabYcbDryEraseMarker(RoboLabObjectOptions):
    """The marker features a cylindrical body with a black cap and a white barrel, prominently displaying the brand name 'Expo' in bold text. The tip is chiseled, designed for both broad and fine lines, and the overall color scheme emphasizes its dry-erase functionality."""

    file: str = "objects/ycb/dry_erase_marker/dry_erase_marker.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="stationery")


@ROBOLAB_REGISTRY.register(name="ycb/jello")
class RoboLabYcbJello(RoboLabObjectOptions):
    """The object is a rectangular box with a bright red color featuring the brand name 'JELL-O' prominently displayed on the top. The sides contain additional information such as flavors and nutritional facts, with a glossy finish."""

    file: str = "objects/ycb/jello/jello.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="ycb/mug")
class RoboLabYcbMug(RoboLabObjectOptions):
    """The object is a pair of red ceramic mugs with a glossy finish. They feature a curved handle and have subtle vertical stripes along the body."""

    file: str = "objects/ycb/mug/mug.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="kitchenware")


@ROBOLAB_REGISTRY.register(name="ycb/mustard")
class RoboLabYcbMustard(RoboLabObjectOptions):
    """The object features a bright yellow plastic squeeze bottle with a conical cap. It is labeled with bold text and nutritional information, highlighting its flavor and usage."""

    file: str = "objects/ycb/mustard/mustard.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="condiment")


@ROBOLAB_REGISTRY.register(name="ycb/pitcher")
class RoboLabYcbPitcher(RoboLabObjectOptions):
    """This pitcher has a smooth, rounded body and a curved handle for easy pouring. It features a deep blue color with a tapered spout at the top."""

    file: str = "objects/ycb/pitcher/pitcher.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="kitchenware")


@ROBOLAB_REGISTRY.register(name="ycb/scissors")
class RoboLabYcbScissors(RoboLabObjectOptions):
    """These scissors have a grey plastic handle with orange accents. The blades are shiny metal and are designed for cutting through paper and other materials."""

    file: str = "objects/ycb/scissors/scissors.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="tool")


@ROBOLAB_REGISTRY.register(name="ycb/soft_scrub")
class RoboLabYcbSoftScrub(RoboLabObjectOptions):
    """The container is a tall, white plastic bottle with a slightly curved shape. It features a colorful label with blue and green hues, including the text 'Soft Scrub' prominently displayed on the front."""

    file: str = "objects/ycb/soft_scrub/soft_scrub.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="bottle")


@ROBOLAB_REGISTRY.register(name="ycb/spam_can")
class RoboLabYcbSpamCan(RoboLabObjectOptions):
    """The can is metallic and rectangular with a glossy finish, predominantly featuring a vibrant green and gold color scheme. It prominently displays the brand name 'SPAM' in bold, white letters, along with colorful images of food dishes on its label."""

    file: str = "objects/ycb/spam_can/spam_can.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="ycb/spring_clamp")
class RoboLabYcbSpringClamp(RoboLabObjectOptions):
    """This object features a robust black plastic design with a textured grip for easy handling. The clamp has bright orange tips that are used to secure items together firmly."""

    file: str = "objects/ycb/spring_clamp/spring_clamp.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="tool")


@ROBOLAB_REGISTRY.register(name="ycb/sugar_box")
class RoboLabYcbSugarBox(RoboLabObjectOptions):
    """The box is primarily yellow and white, featuring the word 'Domino' prominently on the front. It has nutritional information and usage instructions printed on the sides in black text."""

    file: str = "objects/ycb/sugar_box/sugar_box.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="ycb/tomato_soup_can")
class RoboLabYcbTomatoSoupCan(RoboLabObjectOptions):
    """The can is made of metal and features a vibrant red and white label. It displays the text 'Campbell's Tomato Soup' prominently along with nutritional information on the back."""

    file: str = "objects/ycb/tomato_soup_can/tomato_soup_can.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="ycb/tuna_can")
class RoboLabYcbTunaCan(RoboLabObjectOptions):
    """The can is cylindrical with a metallic surface featuring a vibrant blue label. It has text indicating the contents, nutritional information, and branding, along with a design that includes a cartoon fish."""

    file: str = "objects/ycb/tuna_can/tuna_can.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="food")


@ROBOLAB_REGISTRY.register(name="ycb/wood_block")
class RoboLabYcbWoodBlock(RoboLabObjectOptions):
    """The object is a rectangular block made of light-colored wood, with a smooth surface. It has straight edges and a natural wood grain texture, showcasing a warm, beige hue."""

    file: str = "objects/ycb/wood_block/wood_block.urdf"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(category="block")
