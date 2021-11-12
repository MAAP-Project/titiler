import requests
from pprint import pprint
# pip install rio-tiler cogeo-mosaic --pre
from rio_tiler.io import COGReader
from cogeo_mosaic.mosaic import MosaicJSON

stac_endpoint = "https://earth-search.aws.element84.com/v0/search"

stac_json = {
    "collections": ["sentinel-s2-l2a-cogs"],
    # bounding box over washington state
    "bbox": [-124.733643,45.543831,-116.916161,49.002405],
    "datetime": "2020-06-01T00:00:00.000Z/2020-09-30T23:59:59.000Z",
    "limit": 100
}

headers = {
            "Content-Type": "application/json",
            "Accept": "application/geo+json, application/json",
        }

# Read Items

r_stac = requests.post(stac_endpoint, headers=headers, json=stac_json)
items = r_stac.json()
geojson = {'type': 'FeatureCollection', 'features': items['features']}

titiler_endpoint = "https://titiler.maap-project.org"
titiler_endpoint = "https://jnjue1tezi.execute-api.us-east-1.amazonaws.com/"

first_cog = geojson['features'][0]['assets']['overview']['href']
with COGReader(first_cog) as cog:
    info = cog.info()

# We are creating the mosaicJSON using the geojson
# SRTMGL1 CODs have a "browse" asset type, so we can create a mosaic of these type="image/tiff" assets
# accesor function provide access to those
mosaicdata = MosaicJSON.from_features(geojson.get('features'), minzoom=6, maxzoom=info.maxzoom, accessor=lambda feature : feature['assets']['overview']['href'])

r = requests.post(
    url=f"{titiler_endpoint}/mosaicjson/mosaics",
    headers={
        "Content-Type": "application/vnd.titiler.mosaicjson+json",
    },
    json=mosaicdata.dict(exclude_none=True)).json()

pprint(r)

tiles_endpoint = list(filter(lambda x: x.get("rel") == "tiles", dict(r)["links"]))
print(tiles_endpoint)
# tiles link should be like https://h9su0upami.execute-api.us-east-1.amazonaws.com/mosaicjson/mosaics/f53f5571-ee40-494e-a538-c49f56a9ffeb/tiles/{z}/{x}/{y}@1x?
