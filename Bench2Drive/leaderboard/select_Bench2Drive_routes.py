import xml.etree.ElementTree as ET
from pathlib import Path

NAME_OF_NEW_FILE = "bench2drive_EmergencyBreak"
KEEP_TYPES = {
    "BlockedIntersection", "DynamicObjectCrossing", "HardBreakRoute", "OppositeVehicleTakingPriority", 
    "OppositeVehicleRunningRedLight", "ParkingCutIn", "PedestrianCrossing", "ParkingCrossingPedestrian", 
    "StaticCutIn", "VehicleTurningRoute", "VehicleTurningRoutePedestrian", "ControlLoss"
}

# Path to the original and the filtered XML
current_file = Path(__file__)
current_dir = current_file.parent

source_path = current_dir / Path("data/bench2drive220.xml")
filtered_path = current_dir / Path(f"data/{NAME_OF_NEW_FILE}.xml")

# Parse the original XML
tree = ET.parse(source_path)
root = tree.getroot()

# Build a new root element
new_root = ET.Element("routes")

kept_route_ids = []

# Iterate through routes and copy those that contain the desired scenario types
for route in root.findall("route"):
    scenarios = route.find("scenarios")
    if scenarios is None:
        continue
    if any(scenario.attrib.get("type") in KEEP_TYPES for scenario in scenarios.findall("scenario")):
        new_root.append(route)
        kept_route_ids.append(route.attrib.get("id"))

# Write the filtered XML to a new file
new_tree = ET.ElementTree(new_root)
new_tree.write(filtered_path, encoding="utf-8", xml_declaration=True)

print(f"File written to {str(filtered_path)}")