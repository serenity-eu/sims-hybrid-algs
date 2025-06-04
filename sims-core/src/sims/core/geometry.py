import logging
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import shapely
from geopandas import GeoDataFrame, GeoSeries
from shapely import Geometry, LineString, Point, Polygon
from shapely import ops as shapely_ops

log = logging.getLogger(Path(__file__).stem)

PLANAR_CRS = {"proj": "cea"}


def filter_by_max_intersect_percentage(
    clipped_images_gdf: GeoDataFrame, aoi_gdf: GeoDataFrame, max_intersect_percentage: float
) -> GeoDataFrame:
    """
    This function takes a GeoDataFrame containing clipped images and an area of interest (aoi) and filters the images that intersect the aoi by at most the given percentage.

    :param clipped_images_gdf: a GeoDataFrame containing clipped images
    :param aoi_gdf: a GeoDataFrame containing the area of interest
    :param min_intersect_percentage: a float representing the minimum percentage of the aoi that an image can intersect
    :return: a GeoDataFrame containing the images that intersect the aoi by at most the given percentage
    """

    # Calculate the area of the aoi
    aoi_area = aoi_gdf.to_crs(PLANAR_CRS).geometry[0].area  # type: ignore

    # Calculate the area of the intersection between the images and the aoi
    clipped_images_areas = np.array(clipped_images_gdf.to_crs(PLANAR_CRS).area)  # type: ignore

    # Filter the images that intersect the aoi by at most the given percentage
    filtered_images_gdf = clipped_images_gdf[
        (clipped_images_areas / aoi_area) <= max_intersect_percentage
    ]

    return filtered_images_gdf  # type: ignore


def fragmentize(images_gdf: GeoDataFrame, projected_crs: str) -> tuple[GeoSeries, list[list[int]]]:
    """
    This function takes a GeoDataFrame containing image geometries and returns a list of non-overlapping fragments that are covered by the images.

    :todo: This function may be extended by cloud detection/masking capabilities.

    :param image_gdf: a GeoDataFrame containing image geometries
    :return: a list of shapely Polygon objects, each representing a fragment of the aoi that is covered by the images
    """

    if images_gdf.crs != projected_crs:
        projected_images_gdf = images_gdf.copy().to_crs(projected_crs)
    else:
        projected_images_gdf = images_gdf

    # Convert the clipped images to a list of line rings representing the boundaries of the clipped images
    line_rings = projected_images_gdf.boundary.unary_union

    # Convert the line rings to a list of non-overlapping polygons - fragments of the aoi
    projected_fragments_gs = GeoSeries(
        shapely.get_parts(shapely.polygonize([line_rings])), crs=projected_images_gdf.crs
    )  # type: ignore # geopandas is poorly typed

    # Convert the list of polygons to a GeoDataFrame, preserving the original crs
    projected_fragments_gdf = GeoDataFrame(geometry=projected_fragments_gs)  # type: ignore # geopandas is poorly typed

    # Get middle left point
    minx, miny, _, maxy = projected_fragments_gdf.total_bounds
    left_middle_point = (minx, (miny + maxy) / 2)

    # Sort the GeoDataFrame by distance
    # projected_fragments_gdf = projected_fragments_gdf.sort_values(by="geometry", key=lambda geom: GeoSeries(geom, crs=projected_crs).centroid.distance(Point(left_middle_point)), ignore_index=True)

    # Spatially join the fragments geometries with the images that cover them, buffering the images by a small amount to avoid floating point errors
    buffered_projected_images_gdf = GeoDataFrame(
        geometry=projected_images_gdf.geometry.buffer(1e-9), crs=projected_images_gdf.crs
    )  # type: ignore # geopandas is poorly typed
    spatially_joined_fragments = buffered_projected_images_gdf.sjoin(
        projected_fragments_gdf, how="inner", predicate="covers", rsuffix="fragments"
    )

    # Group the fragments by the images that cover them, extract fragments' indices, sort them and convert them to lists
    images_to_fragments = (
        spatially_joined_fragments.groupby(spatially_joined_fragments.index)["index_fragments"]
        .apply(sorted)
        .apply(list)
        .tolist()
    )

    # Reset the fragments crs to the original crs
    fragments_gs = projected_fragments_gs.to_crs(images_gdf.crs)

    return (fragments_gs, images_to_fragments)


def reconstruct(fragments_gdf: GeoDataFrame, images_to_fragments: list[list[int]]) -> GeoDataFrame:
    """
    This function takes a list of fragments and a list of lists of fragment indices that are covered by the images and reconstructs the original images.
    """

    # Create a list of reconstructed images
    reconstructed_images = []

    # Iterate over the list of lists of fragment indices
    for image_fragments in images_to_fragments:
        # Extract the fragments that are covered by the image
        image_fragments_gdf = fragments_gdf.loc[list(image_fragments)]

        # Merge the fragments into a single geometry
        image_geometry = image_fragments_gdf.unary_union

        # Append the reconstructed image to the list
        reconstructed_images.append(image_geometry)

    # Convert the list of reconstructed images to a GeoDataFrame
    reconstructed_images_gdf = GeoDataFrame(geometry=reconstructed_images, crs=fragments_gdf.crs)  # type: ignore # geopandas is poorly typed

    return reconstructed_images_gdf


def does_image_set_cover_aoi(images_gs: GeoSeries, aoi_gs: GeoSeries, projected_crs: str) -> bool:
    if images_gs.crs != projected_crs:
        projected_images_gs = images_gs.copy().to_crs(projected_crs)
        projected_aoi_gs = aoi_gs.copy().to_crs(projected_crs)
    else:
        projected_images_gs = images_gs
        projected_aoi_gs = aoi_gs

    return projected_images_gs.buffer(0).unary_union.buffer(1e-9).covers(projected_aoi_gs)[0]  # type: ignore

def get_coverage_gaps(images_gs: GeoSeries, aoi_gs: GeoSeries, projected_crs: str) -> GeoDataFrame:
    """
    Get the coverage gaps in the preprocessed data.
    :param images_gs: The images GeoSeries.
    :param aoi_gs: The AOI GeoSeries.
    :param projected_crs: The projected CRS to use.
    :return: A GeoDataFrame representing the coverage gaps.
    """

    # Ensure the images and AOI are in the same CRS
    if images_gs.crs != projected_crs:
        projected_images_gs = images_gs.copy().to_crs(projected_crs)
        projected_aoi_gs = aoi_gs.copy().to_crs(projected_crs)
    else:
        projected_images_gs = images_gs
        projected_aoi_gs = aoi_gs

    # Get the union of all fragments
    projected_images_union = projected_images_gs.buffer(0).unary_union.buffer(1e-9)

    # Calculate the difference between the AOI and the union of all images
    coverage_gap_gs = projected_aoi_gs.difference(projected_images_union)

    coverage_gap_gdf = GeoDataFrame(geometry=coverage_gap_gs, crs=projected_crs)  # type: ignore # geopandas is poorly typed
    print(coverage_gap_gdf)

    # Return to the original CRS
    return coverage_gap_gdf.to_crs(images_gs.crs)  # type: ignore # geopandas is poorly typed


def reduce_to_counts(
    images_gdf: GeoDataFrame, aoi_gdf: GeoDataFrame, image_counts: Sequence[int], projected_crs: str
) -> Iterator[GeoDataFrame]:
    """Reduces the number of images to the specified count by removing largest ones while maintaining the coverage of the area of interest."""

    if images_gdf.crs != projected_crs:
        projected_images_gdf = images_gdf.copy().to_crs(projected_crs)
        projected_aoi_gdf = aoi_gdf.copy().to_crs(projected_crs)
    else:
        projected_images_gdf = images_gdf
        projected_aoi_gdf = aoi_gdf

    # Use only the geometries of the images
    projected_images_gs = projected_images_gdf.geometry

    # Get AOI geometry
    aoi_geometry = projected_aoi_gdf.geometry

    # Check if the union of the images covers the aoi
    if not does_image_set_cover_aoi(projected_images_gs, aoi_geometry, projected_crs):
        log.warning("The union of the images does not cover the area of interest.")
        return None

    # Calculate the area of each image
    images_areas = np.array(projected_images_gdf.area)  # type: ignore # geopandas is poorly typed

    # Sort the images by their area in descending order
    sorted_images_indices = np.argsort(images_areas)[::-1]

    # Initialize a list to store the indices of the images to be removed
    remaining_images_selector = [True for _ in range(len(projected_images_gs))]

    # Convert list of image counts to an iterator
    image_counts_iter = iter(image_counts)

    try:
        # Calculate the target image count
        image_count = next(image_counts_iter)
        if len(projected_images_gs[remaining_images_selector]) == image_count:
            # If the number of images equals the target count, return the reduced images_gdf
            yield images_gdf[remaining_images_selector].copy().to_crs(images_gdf.crs).reset_index(drop=True)  # type: ignore # geopandas is poorly typed
            image_count = next(image_counts_iter)
            log.info(
                f"Reducing the number of images from {len(images_gdf)} to {image_count}."
            )

        # Iterate over the sorted images
        for i in sorted_images_indices:
            log.debug(f"Trying to remove image {i}.")
            # Remove the current image from the images_gdf
            remaining_images_selector[i] = False

            # Check if the union of the remaining images covers the aoi
            if does_image_set_cover_aoi(projected_images_gs[remaining_images_selector], aoi_geometry, projected_crs):
                log.debug(f"Image {i} can be removed.")

                if len(projected_images_gs[remaining_images_selector]) == image_count:
                    # If the number of images equals the target count, return the reduced images_gdf
                    yield images_gdf[remaining_images_selector].copy().to_crs(images_gdf.crs).reset_index(drop=True)  # type: ignore # geopandas is poorly typed
                    image_count = next(image_counts_iter)
                    log.info(
                        f"Reducing the number of images from {len(images_gdf)} to {image_count}."
                    )
            else:
                log.info(f"Image {i} is necessary for the coverage.")
                # If the union does not cover the aoi, the image is necessary for the coverage, do not remove it
                remaining_images_selector[i] = True
                continue
    except StopIteration:
        return None


def split_into_quadrants(polygon: Polygon) -> tuple[Polygon, Polygon, Polygon, Polygon]:
    centroid = polygon.centroid
    minx, miny, maxx, maxy = polygon.bounds

    # Create two lines (vertical and horizontal) that pass through the centroid
    vertical_line = LineString([(centroid.x, miny - 1), (centroid.x, maxy + 1)])
    horizontal_line = LineString([(minx - 1, centroid.y), (maxx + 1, centroid.y)])

    # Split the polygon using these lines
    quadrants = []
    halfs = shapely_ops.split(polygon, vertical_line)
    for half in halfs.geoms:
        quadrants.extend(shapely_ops.split(half, horizontal_line).geoms)

    return tuple(quadrants)
