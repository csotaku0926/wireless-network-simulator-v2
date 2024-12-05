import requests
import json

"""
Obtain population density of city via OpenDataSoft API
hardly have any pop. data in OpenStreetMap
"""
def get_city_opendata(city):
    url = f'https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/geonames-all-cities-with-a-population-1000/records?where=cou_name_en%20%3D%20%22{city}%22&offset=0&timezone=UTC&include_links=false&include_app_metas=false'
    res = requests.get(url)
    # parse response
    dct = json.loads(res.content)
    total_count = dct["total_count"]
    print("total count:", total_count)
    
    # save json
    saveFile = f"{city}_data_{total_count}.json"
    out = dct['results']
    with open(saveFile, 'w') as f:
        json.dump(out, f)

    return out

# ASCII Name == Ivanivka
if __name__ == '__main__':
    city_data = get_city_opendata('Ukraine')
    for data in city_data:
        print(data["ascii_name"])