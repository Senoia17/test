import yaml
from shapely.geometry import Point, Polygon


def load_zones(path="configs/zones.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    zones = {}
    for zone_name, info in data["zones"].items():
        zones[zone_name] = Polygon(info["polygon_cm"])

    return zones


def find_zone(point_cm, zones):
    point = Point(float(point_cm[0]), float(point_cm[1]))

    for zone_name, polygon in zones.items():
        if polygon.contains(point) or polygon.touches(point):
            return zone_name

    return "UNKNOWN"

