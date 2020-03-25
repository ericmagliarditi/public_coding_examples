"""
#######################################################
Developed by: Eric Magliarditi
Date Updated: January 14, 2020
Version: 2.0.0
University: Massachusetts Institute of Technology
Use: Deep Telemetry Project
#######################################################
"""
from planetpy import ArgsInputs, Filters, Requests
import argparse
import os
from planet import api
import time

# API_KEY = "fdb528fb28ea4cb79ea6e201c6484045"
API_KEY = os.environ['PL_API_KEY']


def execute(args):
    '''
    This script executes the planet API to download images.

    Current capabilities only allow a single geographic region of interest
    to be obvserved, however, multiple different asset types are allowed.

    Date filters are all checked using REGEX as well as cloud cover percentage

    Currently I utilize command line arguments to give user ability to control
    item types (satellite source) and asset types (visual, etc)

    TODOS:

    1. How to handle requests that are invalid
            -especially when the asset type is not available
    2. How to handle long activation times
    3. Finalize input structure for arguments
    '''

    client = api.ClientV1(api_key=API_KEY)

    start_date = ArgsInputs.get_start_date(args.start_year,
                                           args.start_month, args.start_day)

    percent_cloud = ArgsInputs.get_cloud_percentage(args.cloud_percent)

    regional_filters = Filters.create_regional_filter(
        args.geo_json_path, start_date, percent_cloud)

    req = Requests.create_search_request(regional_filters, args.item_types)

    response = client.quick_search(req)

    path = os.path.join(args.output_dir, 'generated_results.json')

    print("Generated Results JSON")
    with open(path, 'w') as f:
        response.json_encode(f, indent=True)

    print("Begin Item Breakdown")
    callback = api.write_to_file(directory=args.output_dir)
    for item in response.items_iter(5):
        print(f"Item ID: {item['id']}")
        assets = client.get_assets(item).get()

        for asset_type in args.asset_type:
            asset = assets.get(asset_type)
            activation_status = asset.get('status')
            while activation_status == 'inactive':
                print("Activating.....")
                client.activate(asset)
                activation_status = asset.get('status')
                time.sleep(25)
            body = client.download(asset, callback=callback)
            body.wait()

        print("Item Downloaded")
        print("\n")


if __name__ == "__main__":
    '''
    Notes on Args Parse:

    1. Choices argument - forces argument to be within values provided and raises an error if not
    2. Purposely setting date to str so we can quickly parse through in remainder of code
    '''

    parser = argparse.ArgumentParser()

    # parser.add_argument('--start-year', type=int, choices=range(1970,2021), help='Start Year for Analysis')
    parser.add_argument('--start-year', type=str,
                        help='Start Year for Analysis')
    parser.add_argument('--start-month', type=str,
                        help='Start Month for Analysis')
    parser.add_argument('--start-day', type=str, help='Start Day for Analysis')
    parser.add_argument('--cloud-percent', type=float,
                        help='Filter out images that are more than X percent cloud covered')
    parser.add_argument('--geo-json-path', type=str, required=True,
                        help='File path for the geo json document')
    parser.add_argument('--item-types', type=list, default=["PSScene3Band"], help='Satellite item types that are available',
                        choices=["PSOrthoTile", "REOrthoTile", "PSScene3Band",
                                 "PSScene4Band", "REScene", "Sentinel2L1C", "SkySatScene",
                                 "SkySatCollect", "Sentinel1", "MOD09GA", "MOD09GQ", "MYD09GA",
                                 "MYD09GQ"])
    parser.add_argument("--asset-type", type=list, default=[
                        'visual'], help="Represents the type of asset such as visual or analytic from the satellite")
    parser.add_argument("--output-dir", type=str, required=True,
                        help='Directory where assets are saved')

    args = parser.parse_args()

    execute(args)
