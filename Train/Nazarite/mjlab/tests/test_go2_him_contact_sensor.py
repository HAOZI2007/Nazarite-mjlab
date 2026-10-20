"""Regression checks for Go2-HIM leg-contact sensor geometry selection."""

import re
import xml.etree.ElementTree as ET

from nazarite.config.robot_config.go2_cfg import GO2_FOOT_GEOMS, GO2_XML
from nazarite.config.train_config.env_cfgs.him_complex_env_cfg import (
  Nazarite_HIM_Complex_Terrain_Go2,
)


def test_leg_collision_sensor_excludes_foot_geoms():
  cfg = Nazarite_HIM_Complex_Terrain_Go2()
  sensor = next(
    sensor
    for sensor in cfg.scene.sensors or ()
    if sensor.name == "leg_ground_contact"
  )

  assert sensor.primary.mode == "geom"
  pattern = re.compile(sensor.primary.pattern)
  root = ET.parse(GO2_XML).getroot()
  selected_names = {
    geom.attrib["name"]
    for geom in root.iter("geom")
    if "name" in geom.attrib and pattern.fullmatch(geom.attrib["name"])
  }
  expected_names = {
    f"{leg}_{part}_collision_{index}"
    for leg in ("FL", "FR", "RL", "RR")
    for part, count in (("hip", 1), ("thigh", 1), ("calf", 2))
    for index in range(count)
  }

  assert selected_names == expected_names
  assert not selected_names.intersection(GO2_FOOT_GEOMS)

  # The foot geom is a child of the calf body in this MJCF. Keep this explicit
  # so a future model-layout change cannot silently reintroduce body matching.
  foot_geom_bodies = {
    body.attrib["name"]: body
    for body in root.iter("body")
    if body.attrib.get("name", "").endswith("_calf")
  }
  for foot_name in GO2_FOOT_GEOMS:
    calf_name = f"{foot_name}_calf"
    assert any(
      geom.attrib.get("name") == foot_name
      for geom in foot_geom_bodies[calf_name].iter("geom")
    )
