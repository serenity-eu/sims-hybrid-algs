import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Iterator

import aiohttp
import pandas as pd
from geojson import FeatureCollection
from geopandas import GeoDataFrame
from .. import geometry

log = logging.getLogger(Path(__file__).stem)


class CollectionName(StrEnum):
    PLEYADES_HIGH_RES = "phr"
    SPOT = "spot"
    PLEIADES_NEO = "pneo"


@dataclass(frozen=True)
class Up42DataProduct:
    collection_name: CollectionName
    collection_title: str
    host: str
    data_product_id: str


@dataclass(frozen=True)
class SearchParameters:
    start_date: datetime = datetime.strptime("2021-01-01", "%Y-%m-%d")
    end_date: datetime = datetime.now()
    collections: list[CollectionName] = field(
        default_factory=lambda: [
            CollectionName.PLEYADES_HIGH_RES,
            CollectionName.PLEIADES_NEO,
            CollectionName.SPOT,
        ]
    )
    max_cloud_cover: int = 100
    max_image_count: int = 50


class CannotCoverAOIError(Exception):
    pass


class CannotReduceToCountsError(Exception):
    pass


async def _get_token_async(session: aiohttp.ClientSession) -> str:
    if "UP42_USERNAME" not in os.environ or "UP42_PASSWORD" not in os.environ:
        raise ValueError(
            "UP42_USERNAME and UP42_PASSWORD environment variables must be set"
        )

    async with session.post(
        "/oauth/token",
        headers={
            "accept": "application/json",
            "content-type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "password",
            "username": os.getenv("UP42_USERNAME"),
            "password": os.getenv("UP42_PASSWORD"),
        },
    ) as response:
        return (await response.json())["access_token"]


async def _get_data_products(
    session: aiohttp.ClientSession, collections: list[CollectionName]
) -> dict[CollectionName, Up42DataProduct]:
    collection_overview = {}
    async with session.get("/data-products", headers={"accept": "application/json"}) as response:
        raw_data_products = (await response.json())["data"]
        for raw_data_product in raw_data_products:
            collection_name = raw_data_product["collection"]["name"]
            if collection_name not in collections:
                continue

            if raw_data_product["productConfiguration"]["title"] != "Display":
                continue

            collection_title = raw_data_product["collection"]["title"]
            host_name = raw_data_product["collection"]["host"]["name"]

            collection_overview[collection_name] = Up42DataProduct(
                collection_name=collection_name,
                collection_title=collection_title,
                host=host_name,
                data_product_id=raw_data_product["id"],
            )

        return collection_overview


async def _get_order_params_schema(session: aiohttp.ClientSession, data_product_id: str) -> dict:
    async with session.get(
        f"/orders/schema/{data_product_id}", headers={"accept": "application/schema+json"}
    ) as response:
        return await response.json()


def _order_params_for_given_images(
    images_ids: list[str],
    collections: list[CollectionName],
    aoi_geometry: dict,
    collection_to_data_product,
) -> list[dict]:
    return [
        {
            "displayName": f"{id} order",
            "dataProduct": collection_to_data_product[collection].data_product_id,
            "params": {
                "id": id,
                "aoi": aoi_geometry,
            },
            "featureCollection": {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "geometry": aoi_geometry}],
            },
        }
        for id, collection in zip(images_ids, collections)
    ]


async def _search_data_async(
    session: aiohttp.ClientSession,
    aoi_gdf: GeoDataFrame,
    search_params: SearchParameters,
    host_name: str,
) -> GeoDataFrame:
    up42_search_params = {
        "datetime": f"{search_params.start_date.strftime('%Y-%m-%d')}T00:00:00Z/{search_params.end_date.strftime('%Y-%m-%d')}T23:59:59Z",
        "collections": [str(collection) for collection in search_params.collections],
        "intersects": aoi_gdf.geometry[0].__geo_interface__,
        "limit": search_params.max_image_count,
    }

    log.info(f"Searching for {search_params.max_image_count} images in Up42 catalog")

    async with session.post(
        f"/catalog/hosts/{host_name}/stac/search",
        headers={"accept": "application/json", "content-type": "application/json"},
        json=up42_search_params,
    ) as response:
        features = (await response.json())["features"]

        if not features:
            images_gdf = GeoDataFrame()
        else:
            images_gdf = GeoDataFrame.from_features(
                FeatureCollection(features=features), crs="EPSG:4326"
            )

        log.info(f"Found {len(images_gdf)} images in Up42 catalog")
        return images_gdf


async def _estimate_cost_async(session: aiohttp.ClientSession, order_parameters: dict) -> int:
    log.info(f"Estimating cost for order {order_parameters['displayName']}")

    async with session.post(
        "/v2/orders/estimate",
        headers={"accept": "application/json", "content-type": "application/json"},
        json=order_parameters,
    ) as response:
        log.info(f"Received response for order {order_parameters['displayName']}")
        return (await response.json())["summary"]["totalCredits"]


async def _estimate_costs_batch_async(
    session: aiohttp.ClientSession,
    images_gdf: GeoDataFrame,
    aoi_gdf: GeoDataFrame,
    collection_to_data_product: dict[CollectionName, Up42DataProduct],
    batch_size: int = 50,
) -> None:
    # Get two columns from the dataframe as lists
    images_ids, collections = zip(
        *images_gdf.loc[:, ["id", "collection"]].itertuples(index=False, name=None)
    )

    aoi_geometry = aoi_gdf.geometry[0].__geo_interface__
    orders_parameters = _order_params_for_given_images(
        images_ids, collections, aoi_geometry, collection_to_data_product
    )

    cost_estimation_tasks = [
        _estimate_cost_async(session, order_parameters) for order_parameters in orders_parameters
    ]

    costs = []
    for i in range(0, len(orders_parameters), batch_size):
        log.info(f"Estimating cost for orders {i} to {i + batch_size}")

        # Let's refresh the token for each batch
        session.headers.pop("Authorization", None)

        token = await _get_token_async(session)

        session.headers.update({"Authorization": f"Bearer {token}"})

        batch_costs = await asyncio.gather(*cost_estimation_tasks[i : i + batch_size])
        # batch_costs = []
        costs.extend(batch_costs)

    log.info(f"Estimated costs for {len(images_gdf)} images: {costs}")

    images_gdf["cost"] = costs


async def fetch_async(
    aoi_gdf: GeoDataFrame, search_params: SearchParameters, with_costs=False
) -> GeoDataFrame:
    async with aiohttp.ClientSession("https://api.up42.com", raise_for_status=True) as session:
        log.info("Authenticating to Up42")
        token = await _get_token_async(session)

        session.headers.update({"Authorization": f"Bearer {token}"})

        log.info("Collecting data product info from Up42")
        collection_to_data_product = await _get_data_products(session, search_params.collections)

        # Get the host name for the collection (all our collections have the same host)
        host_name = collection_to_data_product[CollectionName.PLEYADES_HIGH_RES].host

        images_gdf: GeoDataFrame = await _search_data_async(
            session, aoi_gdf, search_params, host_name
        )

        if with_costs:
            try:
                await _estimate_costs_batch_async(
                    session, images_gdf, aoi_gdf, collection_to_data_product, batch_size=5
                )
            except Exception as e:
                log.error(f"Error estimating costs: {e}")

        return images_gdf


async def fetch_advanced_async(
    aoi_gdf: GeoDataFrame,
    search_params: SearchParameters,
    image_counts: Iterator[int],
    with_costs=False,
) -> list[GeoDataFrame]:
    async with aiohttp.ClientSession("https://api.up42.com", raise_for_status=True) as session:
        log.info("Authenticating to Up42")
        token = await _get_token_async(session)

        session.headers.update({"Authorization": f"Bearer {token}"})

        log.info("Collecting data product info from Up42")
        collection_to_data_product = await _get_data_products(session, search_params.collections)

        # Get the host name for the collection (all our collections have the same host)
        host_name = collection_to_data_product[CollectionName.PLEYADES_HIGH_RES].host

        log.info(f"Fetching initial pool of {search_params.max_image_count} images from Up42.")
        images_gdf: GeoDataFrame = await _search_data_async(
            session, aoi_gdf, search_params, host_name
        )

        aoi_geometry = aoi_gdf.geometry[0]
        projected_crs = str(aoi_gdf.estimate_utm_crs())

        if not geometry.does_image_set_cover_aoi(images_gdf.geometry, aoi_geometry, projected_crs):
            log.info(
                "Initial image set cannot cover AOI, splitting AOI into quadrants and fetching images for each quadrant"
            )
            aoi_guadrants = geometry.split_into_quadrants(aoi_geometry)
            aoi_quadrants_tasks = [
                _search_data_async(session, aoi_quadrant, search_params, host_name)
                for aoi_quadrant in aoi_guadrants
            ]
            quadrants_images_gdfs = await asyncio.gather(*aoi_quadrants_tasks)
            images_gdf = pd.concat(quadrants_images_gdfs).drop_duplicates().reset_index(drop=True)  # type: ignore # geopandas is poorly typed

            if not geometry.does_image_set_cover_aoi(images_gdf.geometry, aoi_geometry):
                raise CannotCoverAOIError("Cannot cover AOI even with quadrants method")

        log.info("Reducing images set to given counts")
        reduced_images_gdfs = list(geometry.reduce_to_counts(images_gdf, aoi_gdf, image_counts, projected_crs))

        if len(reduced_images_gdfs) == 0:
            log.warning("Cannot reduce to given counts")
            if with_costs:
                await _estimate_costs_batch_async(
                    session, images_gdf, aoi_gdf, collection_to_data_product
                )
            return [images_gdf]

        if with_costs:
            # First estimate cost for largest subset of original set, then propagate costs to smaller subsets (which are subsets of largest subset)
            await _estimate_costs_batch_async(
                session, reduced_images_gdfs[0], aoi_gdf, collection_to_data_product
            )

            # Propagate costs to other GeoDataFrames
            for reduced_images_gdf in reduced_images_gdfs[1:]:
                reduced_images_gdf["cost"] = reduced_images_gdf["id"].map(
                    reduced_images_gdfs[0].set_index("id")["cost"]  # type: ignore # geopandas is poorly typed
                )

        return reduced_images_gdfs


async def estimate_missing_costs_async(
    images_gdf: GeoDataFrame, aoi_gdf: GeoDataFrame
) -> GeoDataFrame:
    images_with_missing_costs_gdf = GeoDataFrame(images_gdf[images_gdf["cost"] == 0])
    if len(images_with_missing_costs_gdf) != 0:
        log.info(f"Estimating costs for {len(images_with_missing_costs_gdf)} images")
        async with aiohttp.ClientSession("https://api.up42.com", raise_for_status=True) as session:
            log.info("Authenticating to Up42")
            token = await _get_token_async(session)

            session.headers.update({"Authorization": f"Bearer {token}"})

            log.info("Collecting data product info from Up42")
            collection_to_data_product = await _get_data_products(
                session, SearchParameters().collections
            )

            await _estimate_costs_batch_async(
                session, images_with_missing_costs_gdf, aoi_gdf, collection_to_data_product
            )

            images_gdf.update(images_with_missing_costs_gdf)
    if len(images_gdf[images_gdf["cost"] == 0]) != 0:
        raise ValueError("Some images still have cost 0")

    return images_gdf


def fetch_advanced(
    aoi_gdf: GeoDataFrame,
    search_params: SearchParameters,
    image_counts: Iterator[int],
    with_costs=False,
) -> list[GeoDataFrame]:
    return asyncio.run(
        fetch_advanced_async(aoi_gdf, search_params, image_counts, with_costs=with_costs)
    )


def fetch(
    aoi_gdf: GeoDataFrame,
    search_params: SearchParameters = SearchParameters(),
    with_costs: bool = False,
) -> GeoDataFrame:
    return asyncio.run(fetch_async(aoi_gdf, search_params, with_costs=with_costs))


def estimate_missing_costs(
    images_gdf: GeoDataFrame, aoi_gdf: GeoDataFrame, inplace: bool = False
) -> GeoDataFrame | None:
    costs = asyncio.run(estimate_missing_costs_async(images_gdf, aoi_gdf))
    if inplace:
        images_gdf.update(costs)
        return None
    else:
        images_gdf = images_gdf.copy()
        images_gdf.update(costs)
        return images_gdf


def normalize(images_gdf: GeoDataFrame) -> GeoDataFrame:
    """
    Transforms the default UP42 image data format into a more suitable format for further processing.
    This function takes a GeoDataFrame containing image data in the UP42 format and extracts or transforms
    specific columns to create a new GeoDataFrame with a simplified structure. The resulting GeoDataFrame
    includes geometry, cost, resolution, cloud coverage, and incidence angle.
    Args:
        images_gdf (GeoDataFrame): A GeoDataFrame containing image data in the UP42 format. It must include
                                   the following columns:
                                   - "geometry": The geometry of the image.
                                   - "cost": The cost associated with the image.
                                   - "resolution": The resolution of the image.
                                   - "cloudCoverage": The cloud coverage percentage of the image.
                                   - "providerProperties": A dictionary containing additional properties,
                                     including "incidenceAngle".
    Returns:
        GeoDataFrame: A new GeoDataFrame containing the following columns:
                      - "geometry": The geometry of the image.
                      - "cost": The cost associated with the image.
                      - "resolution": The resolution of the image.
                      - "cloud_coverage": The cloud coverage percentage of the image.
                      - "incidence_angle": The incidence angle extracted from the provider properties.
    Raises:
        KeyError: If any of the required columns are missing from the input GeoDataFrame.
        TypeError: If the "providerProperties" column does not contain dictionaries with the "incidenceAngle" key.
    Notes:
        - The function assumes that the input GeoDataFrame is well-formed and contains the required columns.
        - The "providerProperties" column is expected to contain dictionaries, and the "incidenceAngle" key
          must be present in each dictionary.
    """
    preprocessed_images_gdf = GeoDataFrame(images_gdf[["id", "geometry", "cost", "resolution"]], crs=images_gdf.crs)
    preprocessed_images_gdf["cloud_coverage"] = images_gdf["cloudCoverage"]
    provider_properties = images_gdf["providerProperties"].to_list()  # type: ignore # geopandas is poorly typed
    preprocessed_images_gdf["incidence_angle"] = [
        json.loads(properties)["incidenceAngle"] for properties in provider_properties
    ]
    return preprocessed_images_gdf

def extract_image_ids(images_gdf: GeoDataFrame) -> list[str]:
    """
    Extracts image IDs from a GeoDataFrame containing image data. ID column is removed from the input GeoDataFrame.
    Args:
        images_gdf (GeoDataFrame): A GeoDataFrame containing image data. It must include the following columns:
                                   - "id": The ID of the image.
    Returns:
        list[str]: A list of image IDs extracted from the input GeoDataFrame.
    Raises:
        KeyError: If the "id" column is missing from the input GeoDataFrame.
    """

    return images_gdf.pop("id").to_list()
