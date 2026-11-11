"""``menagerie_path_for``: which MuJoCo Menagerie model stands in for a robot."""

from __future__ import annotations

from eden.extensions.deployment.utils.menagerie import menagerie_path_for
from eden.options.entities import RobotOptions
from eden.options.robots.limx_oli import Oli
from eden.options.robots.unitree_g1 import G1Dofs29, G1Dofs29NoHands
from eden.options.robots.unitree_go1 import Go1
from eden.options.robots.unitree_go2 import Go2


def test_robots_map_to_their_models_through_the_mro():
    assert menagerie_path_for(Go2()) == "unitree_go2/go2.xml"
    assert menagerie_path_for(Go1()) == "unitree_go1/go1.xml"
    assert menagerie_path_for(G1Dofs29()) == "unitree_g1/g1.xml"
    assert menagerie_path_for(G1Dofs29NoHands()) == "unitree_g1/g1.xml"
    assert menagerie_path_for(Oli()) == "limx_oli/HU_D04_01.xml", "a vendor MJCF, mapped the same way"


def test_an_unmapped_robot_has_no_model():
    assert menagerie_path_for(RobotOptions(file="arm.urdf")) is None
