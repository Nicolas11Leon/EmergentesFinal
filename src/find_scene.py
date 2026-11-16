import urllib.request
import urllib.parse
import json

base = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

# Polígono aproximado de San José del Palmar
# Longitude: -76.23, Latitude: 4.90
# Bbox: [-76.32, 4.86, -76.20, 4.94]

point = "geography'SRID=4326;POINT(-76.233 4.903)'"
filter_str = f"Collection/Name eq 'SENTINEL-2' and contains(Name, 'MSIL2A') and ContentDate/Start ge 2026-08-18T00:00:00.000Z and ContentDate/Start le 2026-08-18T23:59:59.999Z and OData.CSC.Intersects(area={point})"

params = {
    "$filter": filter_str,
    "$top": 5
}
url = f"{base}?{urllib.parse.urlencode(params)}"

req = urllib.request.Request(url, headers={"Accept": "application/json"})
try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        items = data.get("value", [])
        print(f"Found {len(items)} matching Sentinel-2 products:")
        for item in items:
            print("ID:", item.get("Id"))
            print("Name:", item.get("Name"))
            print("S3Path:", item.get("S3Path"))
            print("Footprint:", item.get("Footprint"))
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code)
    print("Body:", e.read().decode())
