import osmium
from shapely import wkb, wkt
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import csv
import os

"""
[plot geo](https://max-coding.medium.com/getting-administrative-boundaries-from-open-street-map-osm-using-pyosmium-9f108c34f86)
"""
# make sure you put your own pbf data in "osm_data" folder !
# download site: https://download.geofabrik.de/europe/ukraine.html
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
        if (
            "admin_level" in a.tags and
            "name" in a.tags and
            (a.tags["admin_level"] == "7" or a.tags["admin_level"] == "2")
        ):
            try:
                wkbshape = self.wkbfab.create_multipolygon(a)
                shapely_obj = wkb.loads(wkbshape, hex=True)
                shapely_obj.area
                area_obj = {"id": a.id, "geo": shapely_obj}
                area = merge_two_dicts(area_obj, a.tags)

                self.areas.append(area)
            except RuntimeError:
                return

def get_osm_data(filename:str):
    handler = AdminAreaHandler()
    # start data file processing
    handler.apply_file(OSM_FILE, locations=True, idx='flex_mem')

    df = pd.DataFrame(handler.areas)
    gdf = gpd.GeoDataFrame(df, geometry="geo")

    # save geodf as csv
    gdf.to_csv(filename)
    

"""output a map png""" 
def plot_gdf(gdf:gpd.GeoDataFrame, level_num:int):
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes()

    # country boundary
    # EPSG 4326 CRS
    gdf[(gdf.admin_level == 2)].set_crs(crs=4326).plot(ax=ax, alpha=1, edgecolor="#000", linewidth=2) 

    # admin level boundaries
    if ("ISO3166-2" in gdf.columns):
        admin_level_gdf = gdf[((gdf.admin_level==level_num) & (~gdf["ISO3166-2"].isna()))].set_crs(crs=4326)
    else:
        admin_level_gdf = gdf[(gdf.admin_level==level_num) ].set_crs(crs=4326)
    admin_level_gdf.plot(ax=ax, alpha=0.1, facecolor='b', edgecolor="#000", linewidth=1)

    # add labels if provision
    if (level_num == 4):
        for _, row in admin_level_gdf.iterrows():
            ax.annotate(text=row["name:en"], xy=(row.geo.centroid.x, row.geo.centroid.y), horizontalalignment='center')

    if (not os.path.isdir("map")):
        os.makedirs("map")
    plt.savefig(f"map/uk_map_level_{level_num}.png")
    plt.show()
        

if __name__ == '__main__':
    # read csv and convert to GeoDataframe
    # ref: https://stackoverflow.com/questions/61122875/geopandas-how-to-read-a-csv-and-convert-to-a-geopandas-dataframe-with-polygons
    CSV_FILENAME = "out_oms_data_l7.csv"
    
    # obtain osm data
    # get_osm_data(CSV_FILENAME)
    
    # read csv as osm
    df = pd.read_csv(CSV_FILENAME)
    df["geo"] = df["geo"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geo", crs='epsg:4326')

    # plot
    # plot_gdf(gdf, 7)

    # match with "ASCII name" col in `population.py`
    print(gdf["name:en"])

    # now we need "area" and "pop."
    # determine pop. distribution
    # TODO: https://stackoverflow.com/questions/23697374/calculate-polygon-area-in-planar-units-e-g-square-meters-in-shapely