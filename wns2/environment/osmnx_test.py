import osmium
import shapely.wkb
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

"""
[plot geo](https://max-coding.medium.com/getting-administrative-boundaries-from-open-street-map-osm-using-pyosmium-9f108c34f86)
"""
OSM_FILE = "osm_data/ukraine-latest.osm.pbf"

def merge_two_dicts(x:dict, y:dict):
    z = x.copy()
    z.update(y) # insert y into z
    return z

class AdminAreaHandler(osmium.SimpleHandler):
    def __init__(self):
        osmium.SimpleHandler.__init__(self)

        self.areas = []
        self.wkbfab = osmium.geom.WKBFactory()

    def area(self, a):
        if ("admin_level" in a.tags and 
            (a.tags["admin_level"] == "4" or a.tags["admin_level"] == "2")):
            print(a)
            wkbshape = self.wkbfab.create_multipolygon(a)
            shapely_obj = shapely.wkb.loads(wkbshape, hex=True)

            area_obj = {"id": a.id, "geo": shapely_obj}
            area = merge_two_dicts(area_obj, a.tags)

            self.areas.append(area)


if __name__ == '__main__':
    handler = AdminAreaHandler()
    # start data file processing
    handler.apply_file(OSM_FILE, locations=True, idx='flex_mem')

    df = pd.DataFrame(handler.areas)
    gdf = gpd.GeoDataFrame(df, geometry="geo")

    # save geodf as csv
    gdf.to_csv("ukraine_admin_boundaries.csv")
    
    # plot
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes()

    # country boundary
    # EPSG 4326 CRS
    gdf[(gdf.admin_level == "2")].set_crs(crs=4326).plot(ax=ax, alpha=1, edgecolor="#000", linewidth=2) 

    # admin level 4 boundaries
    admin_level_4_gdf = gdf[((gdf.admin_level=="4") & (~gdf["ISO3166-2"].isna()))].set_crs(crs=4326)
    admin_level_4_gdf.plot(ax=ax, alpha=0.1, facecolor='b', edgecolor="#000", linewidth=1)

    # add labels
    for idx, row in admin_level_4_gdf.iterrows():
        ax.annotate(text=row["name:en"], xy=(row.geo.centroid.x, row.geo.centroid.y), horizontalalignment='center')

    plt.savefig("uk_map.png")
    plt.show()
        