from __future__ import annotations

from abc import ABC

from sims_solvers import constants
from sims_solvers.Models.GenericModel import GenericModel
from sims_solvers.Instances.InstanceSIMS import InstanceSIMS
from sims_solvers.Config import Config


class SatelliteImageMosaicSelectionGeneralModel(GenericModel, ABC):
    # Model variables (initialized by concrete implementations)
    # These represent solver-specific variable objects (e.g., gp.Var, cp_model variable, etc.)
    select_image: object
    cloud_covered: object
    resolution_element: list[object] = []
    effective_incidence_angle: object
    current_max_incidence_angle: object
    total_area_clouds: int | None = None

    def __init__(self, instance: InstanceSIMS, config: Config | None = None) -> None:
        # Store config before calling parent
        self.config: Config | None = config
        
        super().__init__(instance)
    
    @property
    def sims_instance(self) -> InstanceSIMS:
        """Return the instance cast to InstanceSIMS type."""
        assert isinstance(self.instance, InstanceSIMS), f"Expected InstanceSIMS, got {type(self.instance)}"
        return self.instance

    def problem_name(self) -> str:  # type: ignore[override]
        return constants.Problem.SATELLITE_IMAGE_SELECTION_PROBLEM.value

    def is_a_minimization_model(self) -> bool:  # type: ignore[override]
        return True

    def get_nadir_bound_estimation(self) -> list[float]:  # type: ignore[override]
        # Config must be provided and contain valid objectives
        if not self.config:
            raise ValueError("Config is required but was not provided")
        if not hasattr(self.config, 'objectives'):
            raise ValueError("Config must contain 'objectives' attribute")
        
        objectives_to_use: list[str] = self.config.objectives
        
        nadir_objectives: list[float] = []
        for obj_name in objectives_to_use:
            match obj_name:
                case "min_cost":
                    nadir_objectives.append(float(sum(self.sims_instance.costs)))
                case "cloud_coverage":
                    nadir_objectives.append(float(sum(self.sims_instance.areas)))
                case "min_resolution":
                    nadir_objectives.append(float(self.get_resolution_nadir_for_ref_point()))
                case "min_max_incidence_angle":
                    nadir_objectives.append(float(max(self.sims_instance.incidence_angle)))
                case _:
                    raise ValueError(f"Invalid objective '{obj_name}'. Valid objectives are: min_cost, cloud_coverage, min_resolution, min_max_incidence_angle")
        return nadir_objectives

    def get_ref_points_for_hypervolume(self) -> list[float]:  # type: ignore[override]
        # Config must be provided and contain valid objectives
        if not self.config:
            raise ValueError("Config is required but was not provided")
        if not hasattr(self.config, 'objectives'):
            raise ValueError("Config must contain 'objectives' attribute")
        
        objectives_to_use: list[str] = self.config.objectives
        
        ref_points: list[float] = []
        for obj_name in objectives_to_use:
            match obj_name:
                case "min_cost":
                    ref_points.append(float(sum(self.sims_instance.costs) + 1))
                case "cloud_coverage":
                    ref_points.append(float(sum(self.sims_instance.areas) + 1))
                case "min_resolution":
                    ref_points.append(float(self.get_resolution_nadir_for_ref_point() + 1))
                case "min_max_incidence_angle":
                    ref_points.append(900.0)
                case _:
                    raise ValueError(f"Invalid objective '{obj_name}'. Valid objectives are: min_cost, cloud_coverage, min_resolution, min_max_incidence_angle")
        return ref_points

    def get_resolution_nadir_for_ref_point(self) -> float:
        resolution_parts_max: dict[int, float] = {}
        for idx, image in enumerate(self.sims_instance.images):
            for u in image:
                if u not in resolution_parts_max:
                    resolution_parts_max[u] = float(self.sims_instance.resolution[idx])
                else:
                    if resolution_parts_max[u] < self.sims_instance.resolution[idx]:
                        resolution_parts_max[u] = float(self.sims_instance.resolution[idx])
        return float(sum(resolution_parts_max.values()))

    def assert_solution(self, solution: list[float], selected_images: list[int]) -> None:
        self.assert_is_a_cover(selected_images)
        self.assert_cost(selected_images, solution[0])
        self.assert_cloud_covered(selected_images, solution[1])
        self.assert_resolution(selected_images, solution[2])
        self.assert_incidence_angle(selected_images, solution[3])

    def assert_is_a_cover(self, selected_images: list[int]) -> None:
        # check if it is a cover
        covered_elements: set[int] = set()
        for image in selected_images:
            for element in self.sims_instance.images[image]:
                covered_elements.add(element)
        assert len(covered_elements) == len(self.sims_instance.areas)

    def assert_cost(self, selected_images: list[int], cost: float) -> None:
        total_cost = self.calculate_cost(selected_images)
        assert total_cost == cost

    def calculate_cost(self, selected_images: list[int]) -> float:
        total_cost: float = 0.0
        for image in selected_images:
            total_cost += self.sims_instance.costs[image]
        return total_cost

    def assert_cloud_covered(self, selected_images: list[int], cloud_uncovered: float) -> None:
        calculated_cloud_uncovered = self.calculate_cloud_uncovered(selected_images)
        assert calculated_cloud_uncovered == cloud_uncovered

    def calculate_cloud_uncovered(self, selected_images: list[int]) -> float:
        total_cloud_covered: float = 0.0
        cloud_covered: set[int] = set()
        for image in selected_images:
            if image in self.sims_instance.cloud_covered_by_image:
                for cloud in self.sims_instance.cloud_covered_by_image[image]:
                    if cloud not in cloud_covered:
                        cloud_covered.add(cloud)
                        total_cloud_covered += self.sims_instance.clouds_id_area[cloud]
        total_area_clouds: float = float(sum(self.sims_instance.clouds_id_area.values()))
        return total_area_clouds - total_cloud_covered

    def assert_resolution(self, selected_images: list[int], resolution: float) -> None:
        calculated_total_resolution = self.calculate_resolution(selected_images)
        assert calculated_total_resolution == resolution

    def calculate_resolution(self, selected_images: list[int]) -> float:
        calculated_total_resolution: float = 0.0
        for element in range(len(self.sims_instance.areas)):
            element_resolution: float = float(max(self.sims_instance.resolution))
            for image in selected_images:
                if element in self.sims_instance.images[image]:
                    if self.sims_instance.resolution[image] < element_resolution:
                        element_resolution = float(self.sims_instance.resolution[image])
            calculated_total_resolution += element_resolution
        return calculated_total_resolution

    def assert_incidence_angle(self, selected_images: list[int], incidence_angle: float) -> None:
        max_incidence_angle: float = 0.0
        for image in selected_images:
            if self.sims_instance.incidence_angle[image] > max_incidence_angle:
                max_incidence_angle = float(self.sims_instance.incidence_angle[image])
        assert max_incidence_angle == incidence_angle
