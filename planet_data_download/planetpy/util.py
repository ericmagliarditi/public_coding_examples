import json
import os
import re
from planet.api import filters
from datetime import datetime


class ArgumentChecks:

    def get_correct_year(year):
        if not year:
            year = input(
                "What is the start YEAR for analysis. \n Please use the following format: YYYY ")
        year_test = False
        year_pattern = re.compile(r'^\d{4}$')
        while not year_test:
            match_year = year_pattern.search(year)
            if match_year:
                good_year = int(match_year.group())
                year_test = True
            else:
                year = input(
                    "Please enter a valid year using the following format: YYYY ")

        return good_year

    def get_correct_month(month):
        if not month:
            month = input(
                "What is the start MONTH for analysis. \n Please use the following format: MM ")
        month_test = False
        month_pattern = re.compile(r'^\d\d?$')
        while not month_test:
            match_month = month_pattern.search(month)
            if not match_month:
                month = input(
                    "Please enter a valid month using the following format: MM ")
            else:
                good_month = int(match_month.group())
                if good_month > 12 or good_month <= 0:
                    month = input(
                        "Please enter a valid month between 1 and 12: ")
                    good_month = False
            if match_month and good_month:
                month_test = True

        return good_month

    def get_correct_day(day):
        if not day:
            day = input(
                "What is the start DAY for analysis. \n Please use the following format: DD ")
        day_test = False
        day_pattern = re.compile(r'^\d\d?$')
        while not day_test:
            match_day = day_pattern.search(day)
            if not match_day:
                day = input(
                    "Please enter a valid day using the following format: DD ")
            else:
                good_day = int(match_day.group())
                if good_day > 31 or good_day <= 0:
                    day = input("Please enter a valid day from 1 to 31: ")
                    good_day = False
            if match_day and good_day:
                day_test = True

        return good_day

    def get_correct_cloud_percentage(percent):
        if not percent:
            percent = input(
                "Filter images which are more than this value '%' of clouds? \n Please put in a value between 0 and 1: ")
        percent = str(percent)
        cloud_test = False
        pattern = re.compile(r'-?\d\.?\d{0,}')
        while not cloud_test:
            matches = pattern.findall(percent)
            if not matches:
                percent = input(
                    "Please enter a valid number between 0 and 1  ")
            elif len(matches) > 1:
                percent = input(
                    "Please enter only one value between 0 and 1  ")
            elif len(matches) == 1:
                percent_cloud = float(matches[0])
                if percent_cloud <= 1 and percent_cloud >= 0:
                    cloud_test = True
                else:
                    percent = input(
                        "Please enter a valid number between 0 and 1  ")

        return percent_cloud


class ArgsInputs:
    def get_start_date(year, month, day):
        good_year = ArgumentChecks.get_correct_year(year)
        good_month = ArgumentChecks.get_correct_month(month)
        good_day = ArgumentChecks.get_correct_day(day)

        return datetime(year=good_year, month=good_month, day=good_day)

    def get_cloud_percentage(percent):
        good_percent = ArgumentChecks.get_correct_cloud_percentage(percent)
        return good_percent


class Filters:

    def read_geographic_json(geo_json_path):
        '''
        :param geo_json_path: Geographic Coordinates
:paramtype geo_json_path: json file path
        '''
        with open(geo_json_path, 'r') as f:
            geo_json_geometry = json.load(f)

        return geo_json_geometry

    def create_geometry_filter(geo_json_path):
        '''
        Creates the geometry filter

        :param geo_json_path: Geographic Coordinates
:paramtype geo_json_path: json file path

:returns: the dictionary containing the geometry filter
:rtype: dict
        '''

        geo_json_geometry = Filters.read_geographic_json(geo_json_path)
        geometry_filter = {
            "type": "GeometryFilter",
            "field_name": "geometry",
            "config": geo_json_geometry
        }

        return geometry_filter

    def create_regional_filter(geo_json_path, start_date, percent_cloud):
        '''
        Creates the regional filter needed for analysis

        :param geo_json_path: Geographic Coordinates
		:paramtype geo_json_path: json file path

		:param date_1: Starting Date for analysis
		:paramtype date_1: datetime

		:param percent_cloud: will allow you to filter images that are more than x% clouds
		:paramtype percent_cloud: float [0,1]

		:returns: The aggregated filter for the region of interest
		:rtype: dict

		..note: Assumes AND Logic Filter
        '''

        geometry_filter = Filters.create_geometry_filter(geo_json_path)

        date_filter = filters.date_range('acquired', gte=start_date)

        cloud_filter = filters.range_filter('cloud_cover', lte=percent_cloud)

        regional_filter = filters.and_filter(
            geometry_filter, date_filter, cloud_filter)

        return regional_filter


class Requests:

    def create_search_request(regional_filter, item_types):
        return filters.build_search_request(regional_filter, item_types)
