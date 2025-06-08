import ast
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from . import geometry

logger = logging.getLogger("image_set")


@dataclass
class PreprocessedData:
    aoi_gdf: GeoDataFrame
    up42_image_ids: list[str] | None
    covering_images_gdf: GeoDataFrame
    clipped_images_gs: GeoSeries
    fragments_gs: GeoSeries
    images_to_fragments_mapping: list[list[int]]

    def save(self, output_dir: Path) -> None:
        """
        Save the preprocessed data to the specified output directory.
        :param output_dir: The directory to save the data to.
        """
        self.aoi_gdf.to_file(f"{output_dir}/aoi.geojson", driver="GeoJSON")
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.up42_image_ids is not None:
            (output_dir / "up42_image_ids.json").write_text(json.dumps(self.up42_image_ids))
        self.covering_images_gdf.to_file(f"{output_dir}/covering_images.geojson", driver="GeoJSON")
        self.clipped_images_gs.to_file(f"{output_dir}/clipped_images.geojson", driver="GeoJSON")
        self.fragments_gs.to_file(f"{output_dir}/fragments.geojson", driver="GeoJSON")
        (output_dir / "images_to_fragments_map.txt").write_text(str(self.images_to_fragments_mapping))
    
    @staticmethod
    def load(output_dir: Path) -> "PreprocessedData":
        """
        Load the preprocessed data from the specified output directory.
        :param output_dir: The directory to load the data from.
        :return: A PreprocessedData object containing the loaded data.
        """
        aoi_gdf = GeoDataFrame.from_file(f"{output_dir}/aoi.geojson")
        up42_image_id_path = output_dir / "up42_image_ids.json"
        if up42_image_id_path.exists():
            up42_image_ids = json.loads(up42_image_id_path.read_text())
        else:
            up42_image_ids = None
        covering_images_gdf = GeoDataFrame.from_file(f"{output_dir}/covering_images.geojson")
        clipped_images_gs = GeoSeries.from_file(f"{output_dir}/clipped_images.geojson")
        fragments_gs = GeoSeries.from_file(f"{output_dir}/fragments.geojson")
        images_to_fragments_mapping = ast.literal_eval((output_dir / "images_to_fragments_map.txt").read_text())
        return PreprocessedData(
            aoi_gdf=aoi_gdf,
            up42_image_ids=up42_image_ids,
            covering_images_gdf=covering_images_gdf,
            clipped_images_gs=clipped_images_gs,
            fragments_gs=fragments_gs,
            images_to_fragments_mapping=images_to_fragments_mapping
        )
    
    def validate(self, aoi_gdf: GeoDataFrame, projected_crs: str) -> bool:
        """
        Validate the preprocessed data
        """

        is_valid = True

        # Check that all indices are RangeIndex from 0 to n
        logger.info("Checking index type of covering_images_gdf")
        if not self.covering_images_gdf.index.equals(pd.RangeIndex(stop=len(self.covering_images_gdf))):
            logger.error(f"Indices of covering_images_gdf are not continuous RangeIndex starting from 0. Index: {self.covering_images_gdf.index}")
            is_valid = False
        logger.info("Checking index type of clipped_images_gs")
        if not self.clipped_images_gs.index.equals(pd.RangeIndex(stop=len(self.clipped_images_gs))):
            logger.error(f"Indices of clipped_images_gs are not continuous RangeIndex starting from 0. Index: {self.clipped_images_gs.index}")
            is_valid = False
        logger.info("Checking index type of fragments_gs")
        if not self.fragments_gs.index.equals(pd.RangeIndex(stop=len(self.fragments_gs))):
            logger.error(f"Indices of fragments_gs are not continuous RangeIndex starting from 0. Index: {self.fragments_gs.index}")
            is_valid = False

        # Check that all parts are in fact covering AOI
        logger.info("Checking if covering images cover AOI")
        if not geometry.does_image_set_cover_aoi(self.covering_images_gdf.geometry, aoi_gdf.geometry, projected_crs):
            logger.error("Covering images do not cover the AOI.")
            is_valid = False
        logger.info("Checking if clipped images cover AOI")
        if not geometry.does_image_set_cover_aoi(self.clipped_images_gs, aoi_gdf.geometry, projected_crs):
            logger.error("Clipped images do not cover the AOI.")
            is_valid = False
        logger.info("Checking if fragments cover AOI")
        if not geometry.does_image_set_cover_aoi(self.fragments_gs, aoi_gdf.geometry, projected_crs):
            logger.error("Fragments do not cover the AOI.")
            is_valid = False
        
        # Check that images_to_fragments_mapping has the correct structure, and valid indices
        logger.info("Checking images_to_fragments_mapping size")
        if not len(self.images_to_fragments_mapping) == len(self.clipped_images_gs):
            logger.error(f"images_to_fragments_mapping has incorrect length. Expected: {len(self.clipped_images_gs)}, got: {len(self.images_to_fragments_mapping)}")
            is_valid = False
        
        all_indices = sorted(itertools.chain.from_iterable(self.images_to_fragments_mapping))
        max_index = all_indices[-1]

        logger.info("Checking images_to_fragments_mapping indices range")
        if not max_index < len(self.fragments_gs):
            logger.error(f"images_to_fragments_mapping has invalid indices. Max index: {max_index}, length of fragments_gs: {len(self.fragments_gs)}")
            is_valid = False
        
        # Check that up42_image_ids has the same length as clipped_images_gs
        if len(self.up42_image_ids) != len(self.clipped_images_gs):
            logger.error(f"up42_image_ids has incorrect length. Expected: {len(self.clipped_images_gs)}, got: {len(self.up42_image_ids)}")
            is_valid = False

        return is_valid
    
    def get_coverage_gaps(self, aoi_gdf: GeoDataFrame, projected_crs: str) -> GeoDataFrame:
        """
        Get the coverage gaps in the preprocessed data.
        :param aoi_gdf: The AOI GeoDataFrame.
        :param projected_crs: The projected CRS to use.
        :return: A list of GeoDataFrames representing the coverage gaps.
        """

        projected_aoi_gdf = _ensure_projected_crs(aoi_gdf, projected_crs)
        projected_clipped_images_gdf = _ensure_projected_crs(GeoDataFrame(self.clipped_images_gs, crs=aoi_gdf.crs), projected_crs)
        
        # Get the union of all clipped images
        all_clipped_images = projected_clipped_images_gdf.unary_union

        # Get the union of the AOI
        projected_aoi_union = projected_aoi_gdf.unary_union

        # Find the gaps by subtracting the union of clipped images from the AOI
        gaps = projected_aoi_union.difference(all_clipped_images)

        # Restore the original CRS
        gaps_gdf = GeoDataFrame(geometry=gaps, crs=projected_crs).to_crs(aoi_gdf.crs)

        return gaps_gdf

def _ensure_projected_crs(gdf: GeoDataFrame, projected_crs: str) -> GeoDataFrame:
    """
    Ensure that the GeoDataFrame is in the projected CRS. If not, reproject it.
    :param gdf: The GeoDataFrame to check.
    :param projected_crs: The projected CRS to check against.
    :return: The GeoDataFrame in the projected CRS.
    """
    if gdf.crs != projected_crs:
        projected_gdf = gdf.copy().to_crs(projected_crs)
    else:
        projected_gdf = gdf
    return projected_gdf


def get_covering_images(aoi_gdf: GeoDataFrame, image_set_gdf: GeoDataFrame, projected_crs: str) -> GeoDataFrame:
    projected_aoi_gdf = _ensure_projected_crs(aoi_gdf, projected_crs)
    projected_image_set_gdf = _ensure_projected_crs(image_set_gdf, projected_crs)

    # Find all images which intersect with bounding box https://gis.stackexchange.com/a/266833
    xmin, ymin, xmax, ymax = projected_aoi_gdf.total_bounds
    projected_covering_images_gdf = projected_image_set_gdf.cx[xmin:xmax, ymin:ymax]
    covering_images_gdf = projected_covering_images_gdf.to_crs(image_set_gdf.crs).reset_index(drop=True)
    return covering_images_gdf

def clip_gdf(aoi_gdf: GeoDataFrame, image_set_gdf: GeoDataFrame, projected_crs: str) -> GeoDataFrame:
    """
    Clip the image set to the area of interest (AOI) and return the clipped images.

    :param
        aoi_gdf: A GeoDataFrame containing the AOI.
        image_set_gdf: A GeoDataFrame containing the image set.
        projected_crs: The coordinate reference system to project the AOI and image set to.
    :return: A GeoDataFrame containing the clipped images.
    """
    projected_aoi_gdf = _ensure_projected_crs(aoi_gdf, projected_crs)
    projected_image_set_gdf = _ensure_projected_crs(image_set_gdf, projected_crs)

    # Clip the image set to the AOI
    projected_clipped_images_gdf = projected_image_set_gdf.clip(projected_aoi_gdf, keep_geom_type=True)
    # Reproject the clipped images back to the original CRS
    clipped_images_gdf = projected_clipped_images_gdf.to_crs(image_set_gdf.crs)
    clipped_images_gdf.sort_index(inplace=True)
    return clipped_images_gdf


def preprocess(image_set_gdf: GeoDataFrame, aoi_gdf: GeoDataFrame, projected_crs: str, max_intersect_percentage: float | None = None) -> PreprocessedData:
    logger.info("Extracting image ids from image set...")
    # up42_image_ids = image_set_gdf.pop("id").tolist()
    up42_image_ids = image_set_gdf["id"].tolist()

    logger.info("Looking for all images which intersect aoi...")
    projected_aoi_gdf = _ensure_projected_crs(aoi_gdf, projected_crs)
    projected_image_set_gdf = _ensure_projected_crs(image_set_gdf, projected_crs)

    projected_covering_images_gdf = get_covering_images(projected_aoi_gdf, projected_image_set_gdf, projected_crs)
    logger.info(f"Found {len(projected_covering_images_gdf)} covering images")

    logger.info("Clipping images...")
    projected_clipped_images_gdf = clip_gdf(projected_aoi_gdf, projected_covering_images_gdf, projected_crs)
    projected_clipped_images_gs = projected_clipped_images_gdf.geometry
    logger.debug(f"Found {len(projected_clipped_images_gs)} clipped images")

    if max_intersect_percentage is not None:
        logger.info("Filtering the images by the maximum intersect percentage...")
        projected_clipped_images_gdf = geometry.filter_by_max_intersect_percentage(
            projected_clipped_images_gdf, projected_aoi_gdf, max_intersect_percentage
        )
        logger.debug(f"Found {len(projected_clipped_images_gdf)} images after filtering by intersect percentage")

    logger.info("Fragmentizing images...")
    projected_fragments_gs, images_to_fragments_mapping = geometry.fragmentize(projected_clipped_images_gdf, projected_crs)
    logger.debug(f"Found {len(projected_fragments_gs)} fragments. Images to fragments mapping contains {len(images_to_fragments_mapping)} entries")

    logger.info("Reprojecting all data to original CRS...")
    covering_images_gdf = projected_covering_images_gdf.to_crs(image_set_gdf.crs)
    clipped_images_gs = projected_clipped_images_gs.to_crs(image_set_gdf.crs)
    fragments_gs = projected_fragments_gs.to_crs(image_set_gdf.crs)

    return PreprocessedData(
        aoi_gdf=aoi_gdf,
        up42_image_ids=up42_image_ids,
        covering_images_gdf=covering_images_gdf,
        clipped_images_gs=clipped_images_gs,
        fragments_gs=fragments_gs,
        images_to_fragments_mapping=images_to_fragments_mapping
    )
