import urllib.request
import urllib.parse
import json

product_id = "11a92c32-fde0-48cd-b471-c122795447ff"

def get_nodes(path):
    url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({product_id})/Nodes{path}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("value", [])
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return []

# Navigate GRANULE -> L2A_... -> IMG_DATA -> R20m
root_nodes = get_nodes("")
print("Root nodes:")
for n in root_nodes:
    print(" ", n.get("Name"))

granule = get_nodes("('GRANULE')/Nodes")
for g in granule:
    g_name = n_name = g.get("Name")
    print("Granule:", g_name)
    img_data = get_nodes(f"('GRANULE')/Nodes('{g_name}')/Nodes('IMG_DATA')/Nodes")
    for idir in img_data:
        idir_name = idir.get("Name")
        print("  IMG_DATA dir:", idir_name)
        band_nodes = get_nodes(f"('GRANULE')/Nodes('{g_name}')/Nodes('IMG_DATA')/Nodes('{idir_name}')/Nodes")
        for b in band_nodes:
            b_name = b.get("Name")
            if any(x in b_name for x in ["B05", "B06", "B07", "B8A"]):
                print(f"    FOUND BAND: {b_name} (ID: {b.get('Id')})")
