from __future__ import annotations
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias
from geopandas import GeoDataFrame
from typing import TypeAlias

RectangleBounds: TypeAlias = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class Geodata:
    aoi_gdf: GeoDataFrame
    original_images_gdf: GeoDataFrame
    preprocessed_images_gdf: GeoDataFrame
    clipped_images_gdf: GeoDataFrame
    fragments_gdf: GeoDataFrame
    images_to_fragments_map: list[list[int]]

    @staticmethod
    def load(input_dir: Path) -> Geodata:
        aoi_gdf = GeoDataFrame.from_file(input_dir / "aoi.geojson")
        original_images_gdf = GeoDataFrame.from_file(input_dir / "original_images.geojson")
        preprocessed_images_gdf = GeoDataFrame.from_file(input_dir / "preprocessed_images.geojson")
        clipped_images_gdf = GeoDataFrame.from_file(input_dir / "clipped_images.geojson")
        fragments_gdf = GeoDataFrame.from_file(input_dir / "fragments.geojson")
        images_to_fragments_map = ast.literal_eval(
            (input_dir / "images_to_fragments_map.txt").read_text()
        )

        return Geodata(
            aoi_gdf=aoi_gdf,
            original_images_gdf=original_images_gdf,
            preprocessed_images_gdf=preprocessed_images_gdf,
            clipped_images_gdf=clipped_images_gdf,
            fragments_gdf=fragments_gdf,
            images_to_fragments_map=images_to_fragments_map,
        )

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        self.aoi_gdf.to_file(output_dir / "aoi.geojson")
        self.original_images_gdf.to_file(output_dir / "original_images.geojson")
        self.preprocessed_images_gdf.to_file(output_dir / "preprocessed_images.geojson")
        self.clipped_images_gdf.to_file(output_dir / "clipped_images.geojson")
        self.fragments_gdf.to_file(output_dir / "fragments.geojson")
        (output_dir / "images_to_fragments_map.txt").write_text(str(self.images_to_fragments_map))

    def is_valid(self) -> bool:
        buffered_unary_union = self.clipped_images_gdf.unary_union.buffer(1e-6)
        return buffered_unary_union.contains(self.aoi_gdf.unary_union)

    def fragments_count(self) -> int:
        return len(self.fragments_gdf)

    def images_count(self) -> int:
        return len(self.clipped_images_gdf)

    def aoi_as_rectagle_bounds(self) -> RectangleBounds:
        bounds = self.aoi_gdf.total_bounds
        return ((bounds[1], bounds[0]), (bounds[3], bounds[2]))
