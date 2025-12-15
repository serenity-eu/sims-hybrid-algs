from __future__ import annotations

import math

from sims_solvers.Models.OrtoolsCPModels.OrtoolsCPModel import OrtoolsCPModel
from sims_solvers.Models.SatelliteImageMosaicSelectionGeneralModel import (
    SatelliteImageMosaicSelectionGeneralModel,
)


class SatelliteImageMosaicSelectionOrtoolsCPModel(OrtoolsCPModel, SatelliteImageMosaicSelectionGeneralModel):
    """OrTools CP model for satellite image mosaic selection optimization."""

    def __init__(self, sims_instance, config=None) -> None:
        """Initialize the OrTools CP model.
        
        Args:
            sims_instance: The SIMS instance data
            config: Configuration object with solver settings
        """
        # Handle numerical problems before model creation
        self.tackle_numerical_problems()
        
        # Initialize config
        self.config = config
        
        # Initialize collections
        self.effective_image_resolution = []
        self.resolution_element = []
        
        OrtoolsCPModel.__init__(self)
        SatelliteImageMosaicSelectionGeneralModel.__init__(self, sims_instance, config)

    def is_numerically_possible_augment_objective(self) -> bool:
        """Check if it's numerically possible to augment the objective.
        
        Returns:
            Always returns True for this implementation
        """
        return False

    def get_data_from_instance(self) -> None:
        """Extract and process data from the SIMS instance."""
        self.total_area_clouds = int(sum(self.sims_instance.clouds_id_area.values()))

    def add_variables_to_model(self) -> None:
        """Add decision variables to the OrTools CP model."""
        self.select_image = [self.ortools_solver_model.NewBoolVar(f"select_image{i}") for i in range(len(self.sims_instance.images))]
        self.solution_variables.append(self.select_image)
        self.cloud_covered = {}
        self.cloud_area = {}
        for cloud in self.sims_instance.clouds_id_area:
            self.cloud_covered[cloud] = self.ortools_solver_model.NewBoolVar(f"cloud_covered{cloud}")
            self.cloud_area[cloud] = self.ortools_solver_model.NewIntVar(0, self.sims_instance.clouds_id_area[cloud], f"cloud_area{cloud}")
        
        # Add variables for incidence angle objective
        self.effective_incidence_angle = [
            self.ortools_solver_model.NewIntVar(0, max(self.sims_instance.incidence_angle), f"effective_incidence{i}")
            for i in range(len(self.sims_instance.images))
        ]

    def add_constraints_to_model(self) -> None:
        """Add constraints to the OrTools CP model."""
        # Config must be provided and contain valid objectives
        if not self.config:
            raise ValueError("Config is required but was not provided")
        if not hasattr(self.config, 'objectives'):
            raise ValueError("Config must contain 'objectives' attribute")
        
        objectives_to_use = self.config.objectives
        
        # Coverage constraint - always required
        for i in range(len(self.sims_instance.areas)):
            images_covering_element = self.get_images_covering_element(i)
            self.constraints.append(self.ortools_solver_model.AddAtLeastOne(self.select_image[j] for j in images_covering_element))

        # Cloud constraints - only add if cloud_coverage objective is used
        if "cloud_coverage" in objectives_to_use:
            for cloud in self.cloud_area:
                potential_images_covering_cloud = self.get_images_covering_cloud(cloud)
                for i in potential_images_covering_cloud:
                    self.ortools_solver_model.Add(self.cloud_covered[cloud] == 1).OnlyEnforceIf(self.select_image[i])
                self.constraints.append(self.ortools_solver_model.AddAtLeastOne(self.select_image[i] for i in potential_images_covering_cloud).OnlyEnforceIf(
                    self.cloud_covered[cloud]))
                self.constraints.append(self.ortools_solver_model.Add(self.cloud_area[cloud] == 0).OnlyEnforceIf(self.cloud_covered[cloud]))
                self.constraints.append(self.ortools_solver_model.Add(self.cloud_area[cloud] == self.sims_instance.clouds_id_area[cloud]).OnlyEnforceIf(
                    self.cloud_covered[cloud].Not()))
        
        # Incidence angle constraints - only add if min_max_incidence_angle objective is used
        if "min_max_incidence_angle" in objectives_to_use:
            for i in range(len(self.sims_instance.images)):
                # If image is not selected, effective incidence is 0
                self.ortools_solver_model.Add(self.effective_incidence_angle[i] == 0).OnlyEnforceIf(self.select_image[i].Not())
                # If image is selected, effective incidence equals actual incidence angle
                self.ortools_solver_model.Add(self.effective_incidence_angle[i] == self.sims_instance.incidence_angle[i]).OnlyEnforceIf(self.select_image[i])

    def define_objectives(self) -> None:
        """Define optimization objectives based on configuration."""
        # Config must be provided and contain valid objectives
        if not self.config:
            raise ValueError("Config is required but was not provided")
        if not hasattr(self.config, 'objectives'):
            raise ValueError("Config must contain 'objectives' attribute")
        
        objectives_to_use = self.config.objectives
        
        available_objectives = {
            "min_cost": self._define_cost_objective,
            "cloud_coverage": self._define_cloud_coverage_objective,
            "min_resolution": self._define_resolution_objective,
            "min_max_incidence_angle": self._define_incidence_angle_objective
        }
        
        # Add only the requested objectives
        for obj_name in objectives_to_use:
            if obj_name in available_objectives:
                available_objectives[obj_name]()
            else:
                raise ValueError(f"Invalid objective '{obj_name}'. Valid objectives are: {list(available_objectives.keys())}")
    
    def _define_cost_objective(self) -> None:
        """Define cost objective"""
        self.total_cost = self.ortools_solver_model.NewIntVar(0, sum(self.sims_instance.costs), "cost")
        (self.ortools_solver_model.Add(self.total_cost == sum(self.select_image[i] * self.sims_instance.costs[i]
                                                     for i in range(len(self.select_image)))).
         WithName("cost_obj_constraint"))
        self.objectives.append(self.total_cost)
    
    def _define_cloud_coverage_objective(self) -> None:
        """Define cloud coverage objective"""
        if self.total_area_clouds is None:
            raise ValueError("total_area_clouds must be initialized before defining cloud coverage objective")
        self.total_cloudy_area = self.ortools_solver_model.NewIntVar(0, self.total_area_clouds, "total_cloudy_area")
        (self.ortools_solver_model.Add(self.total_cloudy_area == sum(self.cloud_area[i] for i in self.cloud_area)).
         WithName("cloud_covered_obj_constraint"))
        self.objectives.append(self.total_cloudy_area)
    
    def _define_resolution_objective(self) -> None:
        """Define resolution objective"""
        max_total_resolution = max(self.sims_instance.resolution) * len(self.sims_instance.areas)
        self.total_resolution = self.ortools_solver_model.NewIntVar(0, max_total_resolution, "total_resolution")
        self.effective_image_resolution = [
            self.ortools_solver_model.NewIntVar(self.sims_instance.resolution[i], max(self.sims_instance.resolution) + 10,
                                 f"effective_resolution{i}") for i in
            range(len(self.sims_instance.images))]
        for i in range(len(self.sims_instance.images)):
            self.ortools_solver_model.Add(self.effective_image_resolution[i] == self.sims_instance.resolution[i]).OnlyEnforceIf(
                self.select_image[i])
            self.ortools_solver_model.Add(self.effective_image_resolution[i] == max(self.sims_instance.resolution) + 10).OnlyEnforceIf(
                self.select_image[i].Not())
        for i in range(len(self.sims_instance.areas)):
            self.resolution_element.append(self.ortools_solver_model.NewIntVar(0, max(self.sims_instance.resolution), f"resolution{i}"))
            images = self.get_images_covering_element(i)
            self.ortools_solver_model.AddMinEquality(self.resolution_element[i],
                                      [self.effective_image_resolution[ima] for ima in images])

        self.ortools_solver_model.Add(self.total_resolution == sum(self.resolution_element))
        self.objectives.append(self.total_resolution)
    
    def _define_incidence_angle_objective(self) -> None:
        """Define incidence angle objective - minimize the maximum incidence angle among selected images"""
        self.current_max_incidence_angle = self.ortools_solver_model.NewIntVar(
            0,  # Can be 0 if no images selected (though coverage constraint prevents this)
            max(self.sims_instance.incidence_angle),
            "max_incidence_angle"
        )
        # Set current_max_incidence_angle to be the maximum of all effective_incidence_angle values
        self.ortools_solver_model.AddMaxEquality(
            self.current_max_incidence_angle,
            self.effective_incidence_angle
        )
        self.objectives.append(self.current_max_incidence_angle)

    def get_images_covering_element(self, element: int) -> list[int]:
        """Get list of image indices that cover the specified element.
        
        Args:
            element: The element index to find covering images for
            
        Returns:
            List of image indices that cover the element
        """
        return [i for i in range(len(self.sims_instance.images)) if element in self.sims_instance.images[i]]

    def get_images_covering_cloud(self, cloud: int) -> list[int]:
        """Get list of image indices that cover the specified cloud.
        
        Args:
            cloud: The cloud index to find covering images for
            
        Returns:
            List of image indices that cover the cloud
        """
        return [i for i in self.sims_instance.cloud_covered_by_image if cloud in self.sims_instance.cloud_covered_by_image[i]]

    def get_solution_values(self) -> list[int]:
        """Get the solution values from the solved model.
        
        Returns:
            List of selected image indices
        """
        selected_images = [index for index in range(len(self.select_image)) if
                           self.solver_values[index] == 1]
        return selected_images

    # for testing only
    def print_solution_values_model_values(self) -> None:
        """Print the values of selected images for debugging purposes."""
        for index in range(len(self.select_image)):
            if self.solver_values[index] == 1:
                print(f"Image {index} is selected with a value of {self.solver_values[index]}")

    def tackle_numerical_problems(self) -> None:
        """Handle numerical problems in the model (currently no implementation)."""
        # self.sims_instance.costs = [int(x/self.gcd(self.instance.costs)) for x in self.instance.costs]
        pass

    def gcd(self, list_to_gcd: list[int]) -> int:
        """Calculate the greatest common divisor of a list of integers.
        
        Args:
            list_to_gcd: List of integers to find GCD for
            
        Returns:
            The greatest common divisor of all integers in the list
        """
        gcd = list_to_gcd[0]
        for i in range(1, len(list_to_gcd)):
            gcd = math.gcd(gcd, list_to_gcd[i])
        return gcd

    def add_necessary_solver_configuration(self) -> None:
        """Add necessary solver configuration (currently no additional configuration needed)."""
        print("Extra solver configuration no needed")