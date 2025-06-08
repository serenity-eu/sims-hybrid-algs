from __future__ import annotations

import ast
import random
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from ..image_set import PreprocessedData
from .geodata import Geodata


@dataclass
class SimsProblem:
    num_images: int
    universe: int
    images: list[frozenset[int]]
    costs: list[int]
    cloud_coverages: list[float]
    areas: list[float]
    resolution: list[float]
    incidence_angle: list[float]
    max_cloud_area: float

    @staticmethod
    def from_geodata(geodata: Geodata) -> SimsProblem:
        num_images = len(geodata.clipped_images_gdf)
        universe = len(geodata.fragments_gdf)
        images = [
            frozenset(sorted(image_fragments))
            for image_fragments in geodata.images_to_fragments_map
        ]
        costs = geodata.preprocessed_images_gdf["cost"].tolist()  # type: ignore # geopandas is poorly typed
        areas_m2 = geodata.fragments_gdf.geometry.to_crs({"proj": "cea"}).area.to_list()
        resolution = geodata.preprocessed_images_gdf["resolution"].tolist()  # type: ignore # geopandas is poorly typed
        cloud_coverages = geodata.preprocessed_images_gdf["cloud_coverage"].tolist()  # type: ignore # geopandas is poorly typed
        incidence_angle = geodata.preprocessed_images_gdf["incidence_angle"].tolist()  # type: ignore # geopandas is poorly typed
        max_cloud_area = sum(areas_m2)

        return SimsProblem(
            num_images=num_images,
            universe=universe,
            images=images,
            costs=costs,
            cloud_coverages=cloud_coverages,
            areas=areas_m2,
            resolution=resolution,
            incidence_angle=incidence_angle,
            max_cloud_area=max_cloud_area,
        )

    @classmethod
    def from_preprocessed_data(cls, preprocessed_data: PreprocessedData) -> SimsProblem:
        num_images = len(preprocessed_data.covering_images_gdf)
        universe = len(preprocessed_data.fragments_gs)
        images = [
            frozenset(sorted(image_fragments))
            for image_fragments in preprocessed_data.images_to_fragments_mapping
        ]
        costs = preprocessed_data.covering_images_gdf["cost"].tolist()  # type: ignore # geopandas is poorly typed
        areas_m2 = preprocessed_data.fragments_gs.geometry.to_crs({"proj": "cea"}).area.to_list()
        resolution = preprocessed_data.covering_images_gdf["resolution"].tolist()  # type: ignore # geopandas is poorly typed
        cloud_coverages = preprocessed_data.covering_images_gdf["cloud_coverage"].tolist()  # type: ignore # geopandas is poorly typed
        incidence_angle = preprocessed_data.covering_images_gdf["incidence_angle"].tolist()  # type: ignore # geopandas is poorly typed
        max_cloud_area = sum(areas_m2)

        return cls(
            num_images=num_images,
            universe=universe,
            images=images,
            costs=costs,
            cloud_coverages=cloud_coverages,
            areas=areas_m2,
            resolution=resolution,
            incidence_angle=incidence_angle,
            max_cloud_area=max_cloud_area,
        )

    def _generate_bool_clouds(self) -> list[frozenset[int]]:
        """
        Generate the cloud sets from the cloud coverages of the images.
        """
        cloudy_fragments = []
        for image_idx, image_fragments in enumerate(self.images):
            image_area = sum(self.areas[fragment_idx] for fragment_idx in image_fragments)
            cloudy_area = self.cloud_coverages[image_idx] / 100 * image_area
            cloudy_fragments.append(set())

            # If the image has no clouds, add an empty set
            if cloudy_area == 0:
                continue

            cloudy_fragments_area = 0

            # If the image has clouds, add fragments until the cloud area is reached
            random_fragments_order = list(image_fragments)
            random.shuffle(random_fragments_order)

            for fragment_idx in random_fragments_order:
                fragment_area = self.areas[fragment_idx]

                # if the addition of the fragment area would exceed the cloud area, skip it
                if cloudy_fragments_area + fragment_area >= 1.1 * cloudy_area:
                    continue

                cloudy_fragments_area += fragment_area
                cloudy_fragments[-1].add(fragment_idx)

        cloudy_fragments = [frozenset(cloudy_fragments) for cloudy_fragments in cloudy_fragments]
        return cloudy_fragments

    def discretize(self) -> SimsDiscreteProblem:
        costs_int = [int(cost) for cost in self.costs]
        areas_int = [int(area) for area in self.areas]
        resolution_int = [int(res * 100) for res in self.resolution]
        incidence_angle_int = [int(round(angle * 10)) for angle in self.incidence_angle]
        max_cloud_area_int = sum(areas_int)
        clouds = self._generate_bool_clouds()

        return SimsDiscreteProblem(
            num_images=self.num_images,
            universe=self.universe,
            images=self.images,
            costs=costs_int,
            clouds=clouds,
            areas=areas_int,
            resolution=resolution_int,
            incidence_angle=incidence_angle_int,
            max_cloud_area=max_cloud_area_int,
        )

    def to_dict(self) -> dict:
        return {
            "num_images": self.num_images,
            "universe": self.universe,
            "images": self.images,
            "costs": self.costs,
            "resolution": self.resolution,
            "incidence_angle": self.incidence_angle,
            "clouds": [],
            "areas": self.areas,
            "max_cloud_area": self.max_cloud_area,
        }

    def to_dzn(self, output_path: PathLike):
        problem_dictionary = self.to_dict()
        output_path = Path(output_path).resolve()

        for key in ["images", "clouds"]:
            if key not in problem_dictionary:
                continue
            # Adjust index to start at 1, make it a list of sets, hadle the case of empty sets
            problem_dictionary[key] = [
                {i + 1 for i in indices} for indices in problem_dictionary[key]
            ]

        dzn_string = "\n".join(f"{key} = {value};" for key, value in problem_dictionary.items())
        output_path.write_text(dzn_string)


@dataclass
class SimsDiscreteProblem:
    num_images: int
    universe: int
    images: list[frozenset[int]]
    costs: list[int]
    clouds: list[frozenset[int]]
    areas: list[int]
    resolution: list[int]
    incidence_angle: list[int]
    max_cloud_area: int

    @staticmethod
    def from_dzn(input_path: PathLike) -> SimsDiscreteProblem:
        data = Path(input_path).read_text()
        parsed_data = {}
        for line in data.splitlines():
            if line.strip() == "":
                continue
            line = line.replace(";", "")
            key, value = map(str.strip, line.split("="))
            parsed_data[key] = ast.literal_eval(value)
            if key == "images" or key == "clouds":
                # Adjust index to start at 0
                parsed_data[key] = [
                    frozenset(sorted(i - 1 for i in indices)) for indices in parsed_data[key]
                ]
        return SimsDiscreteProblem(**parsed_data)

    def to_dzn(self, output_path: Path):
        problem_dictionary = self.to_dict()
        output_path = output_path.resolve()

        for key in ["images", "clouds"]:
            if key not in problem_dictionary:
                continue
            # Adjust index to start at 1, make it a list of sets
            list_of_sets = [{i + 1 for i in indices} for indices in problem_dictionary[key]]

            problem_dictionary[key] = (
                "[" + ", ".join(str(s) if s else "{}" for s in list_of_sets) + "]"
            )

        dzn_string = "\n".join(f"{key} = {value};" for key, value in problem_dictionary.items())
        output_path.write_text(dzn_string)

    def to_dict(self) -> dict:
        result_dict = {
            "num_images": self.num_images,
            "universe": self.universe,
            "images": [list(image_fragments) for image_fragments in self.images],
            "costs": self.costs,
            "clouds": [list(cloudy_fragments) for cloudy_fragments in self.clouds],
            "areas": self.areas,
            "resolution": self.resolution,
            "incidence_angle": self.incidence_angle,
            "max_cloud_area": self.max_cloud_area,
        }
        return result_dict

    @staticmethod
    def from_dict(data: dict) -> SimsDiscreteProblem:
        return SimsDiscreteProblem(
            num_images=data["num_images"],
            universe=data["universe"],
            images=[frozenset(image_fragments) for image_fragments in data["images"]],
            costs=data["costs"],
            clouds=[frozenset(cloudy_fragments) for cloudy_fragments in data["clouds"]],
            areas=data["areas"],
            resolution=data["resolution"],
            incidence_angle=data["incidence_angle"],
            max_cloud_area=data["max_cloud_area"],
        )

    def get_max_values(self):
        return sum(self.costs), sum(self.areas)

    def get_ref_point(self):
        max_values = self.get_max_values()
        return max_values[0] + 1, max_values[1] + 1
    
    def validate(self):
        if not isinstance(self.num_images, int) or self.num_images <= 0:
            raise ValueError("num_images must be a positive integer")
        if not isinstance(self.universe, int) or self.universe <= 0:
            raise ValueError("universe must be a positive integer")
        if len(self.images) != self.num_images:
            raise ValueError("Number of images does not match num_images")
        if len(self.costs) != self.num_images:
            raise ValueError("Number of costs does not match num_images")
        if len(self.clouds) != self.num_images:
            raise ValueError("Number of clouds does not match num_images")
        if len(self.areas) != self.universe:
            raise ValueError("Number of areas does not match universe")
        if len(self.resolution) != self.num_images:
            raise ValueError("Number of resolutions does not match num_images")
        if len(self.incidence_angle) != self.num_images:
            raise ValueError("Number of incidence angles does not match num_images")
        # Check if the union of all image fragments covers all indices from 0 to self.universe-1
        all_indices = set().union(*self.images)
        if all_indices != set(range(self.universe)):
            difference = set(range(self.universe)) - all_indices
            raise ValueError("Images do not cover all indices from 0 to universe-1. Missing indices: " + str(difference))


@dataclass
class ProblemInstance:
    name: str
    problem: SimsDiscreteProblem
    path: Path | None = None

    def to_json(self, include_definition=False) -> dict:
        return self.to_dict(include_definition)

    def to_dict(self, include_definition=False):
        result_dict: dict[str, object] = {"name": self.name}

        if include_definition:
            result_dict["problem"] = self.problem.to_dict()
        return result_dict

    @staticmethod
    def from_dzn(input_path: Path) -> ProblemInstance:
        problem = SimsDiscreteProblem.from_dzn(input_path)
        name = input_path.stem
        return ProblemInstance(name, problem, input_path)

    def to_dzn(self, output_path: Path):
        self.problem.to_dzn(output_path)
