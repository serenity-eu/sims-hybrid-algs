from __future__ import annotations

import gurobipy as gp
from gurobipy import max_

from sims_solvers.Models.GurobiModels.GurobiModel import GurobiModel
from sims_solvers.Models.SatelliteImageMosaicSelectionGeneralModel import (
    SatelliteImageMosaicSelectionGeneralModel,
)
from sims_solvers.Instances.InstanceSIMS import InstanceSIMS
from sims_solvers.Config import Config


class SatelliteImageMosaicSelectionGurobiModel(GurobiModel, SatelliteImageMosaicSelectionGeneralModel):
    # Decision variables (initialized by add_variables_to_model)
    select_image: gp.tupledict
    cloud_covered: gp.tupledict
    
    # Support variables (initialized by add_variables_to_model)
    resolution_element: gp.tupledict
    effective_incidence_angle: gp.tupledict
    current_max_incidence_angle: gp.Var
    auxiliary_variables_for_resolution: list[dict[int, gp.Var] | int] = []
    
    # Data attributes (initialized by get_data_from_instance)
    elements: gp.tuplelist
    areas: gp.tupledict
    images_id: gp.tuplelist
    images: gp.tupledict
    costs: gp.tupledict
    cloud_covered_by_image: gp.tupledict
    clouds_id: gp.tuplelist
    area_clouds: gp.tupledict
    resolution: gp.tupledict
    incidence_angle: gp.tupledict
    images_covering_element: gp.tupledict 

    def __init__(self, sims_instance: InstanceSIMS, config: Config | None = None) -> None:
        self.config: Config | None = config
        SatelliteImageMosaicSelectionGeneralModel.__init__(self, sims_instance, config)

    def is_numerically_possible_augment_objective(self) -> bool:
        return False  # For Ortools-cp it is not possible

    def create_model(self) -> gp.Model:
        return gp.Model("SIMSModel")

    def get_data_from_instance(self) -> None:
        self.elements, self.areas = gp.multidict({i: self.sims_instance.areas[i] for i in range(len(self.sims_instance.areas))})
        self.images_id, self.images, self.costs = gp.multidict({i: [self.sims_instance.images[i], self.sims_instance.costs[i]]
                                                                for i in range(len(self.sims_instance.images))})
        # cloud processing
        self.cloud_covered_by_image = gp.tupledict(self.sims_instance.cloud_covered_by_image)
        self.clouds_id, self.area_clouds = gp.multidict(self.sims_instance.clouds_id_area)
        self.total_area_clouds = int(sum(self.area_clouds.values()))
        # resolution processing
        self.resolution = gp.tupledict(zip(self.images_id, self.sims_instance.resolution))
        self.min_resolution = min(self.sims_instance.resolution)
        images_covering_element = {}
        for i in self.images_id:
            for e in self.images[i]:
                if e not in images_covering_element:
                    images_covering_element[e] = [i]
                else:
                    images_covering_element[e].append(i)
        self.images_covering_element = gp.tupledict(images_covering_element)
        # incidence angle processing
        self.incidence_angle = gp.tupledict(zip(self.images_id, self.sims_instance.incidence_angle))

    def add_variables_to_model(self) -> None:
        # decision variables
        self.select_image = self.gurobi_solver_model.addVars(len(self.images), vtype=gp.GRB.BINARY, name="select_image_i")
        self.cloud_covered = self.gurobi_solver_model.addVars(self.clouds_id, vtype=gp.GRB.BINARY, name="cloud_covered_e")
        # support variables
        self.resolution_element = self.gurobi_solver_model.addVars(self.elements, lb=self.min_resolution,
                                                            ub=max(self.resolution.values()), vtype=gp.GRB.INTEGER,
                                                            name="resolution_element_i")
        self.auxiliary_variables_for_resolution = [0] * len(self.elements)
        for element in self.elements:
            self.auxiliary_variables_for_resolution[element] = {}
            for image in self.images_covering_element[element]:
                self.auxiliary_variables_for_resolution[element][image] = self.gurobi_solver_model.addVar(
                    vtype=gp.GRB.BINARY,
                    name=f"auxiliary_variable_for_resolution{element}_{image}")
        self.effective_incidence_angle = self.gurobi_solver_model.addVars(len(self.images), vtype=gp.GRB.INTEGER,
                                                                   name="effective_incidence_angle_i")
        self.current_max_incidence_angle = self.gurobi_solver_model.addVar(vtype=gp.GRB.INTEGER, name="max_allowed_incidence_angle")

    def define_objectives(self) -> None:
        # Config must be provided and contain valid objectives
        if not self.config:
            raise ValueError("Config is required but was not provided")
        if not hasattr(self.config, 'objectives'):
            raise ValueError("Config must contain 'objectives' attribute")
        
        objectives_to_use = self.config.objectives
        
        available_objectives = {
            "min_cost": lambda: gp.quicksum(self.select_image[i] * self.costs[i] for i in self.images_id),
            "cloud_coverage": lambda: self.total_area_clouds - (gp.quicksum(self.cloud_covered[c] * self.area_clouds[c]
                                                              for c in self.clouds_id)),
            "min_resolution": lambda: gp.quicksum(self.resolution_element[e] for e in self.elements),
            "min_max_incidence_angle": lambda: self.current_max_incidence_angle
        }
        
        # Add only the requested objectives
        for obj_name in objectives_to_use:
            if obj_name in available_objectives:
                self.objectives.append(available_objectives[obj_name]())
            else:
                raise ValueError(f"Invalid objective '{obj_name}'. Valid objectives are: {list(available_objectives.keys())}")

    def review_objective_values(self, objective_values: list[float]) -> None:
        # for the current model the resolution value cannot be obtained from Gurobi, so it is calculated manually
        selected_images = self.get_solution_values()
        
        # Config must be provided and contain valid objectives
        if not self.config:
            raise ValueError("Config is required but was not provided")
        if not hasattr(self.config, 'objectives'):
            raise ValueError("Config must contain 'objectives' attribute")
        
        objectives_to_use = self.config.objectives
        
        # Find the index of resolution objective if it exists and calculate it manually
        for i, obj_name in enumerate(objectives_to_use):
            if obj_name == "min_resolution":
                objective_values[i] = self.calculate_resolution(selected_images)
                break
            elif obj_name not in ["min_cost", "cloud_coverage", "min_resolution", "min_max_incidence_angle"]:
                raise ValueError(f"Invalid objective '{obj_name}'. Valid objectives are: min_cost, cloud_coverage, min_resolution, min_max_incidence_angle")

    def add_constraints_to_model(self) -> None:
        # Config must be provided and contain valid objectives
        if not self.config:
            raise ValueError("Config is required but was not provided")
        if not hasattr(self.config, 'objectives'):
            raise ValueError("Config must contain 'objectives' attribute")
        
        objectives_to_use = self.config.objectives
        
        # Cover constraint - always required (fundamental constraint)
        self.constraints.append(self.gurobi_solver_model.addConstrs(gp.quicksum(self.select_image[i] for i in self.images_id if e in self.images[i]) >= 1
                                     for e in self.elements))
        
        # Cloud constraints - only add if cloud_coverage objective is used
        if "cloud_coverage" in objectives_to_use:
            self.constraints.append(self.gurobi_solver_model.addConstrs(gp.quicksum(self.select_image[i] for i in self.cloud_covered_by_image.keys()
                                                     if c in self.cloud_covered_by_image[i]) >= self.cloud_covered[c]
                                         for c in self.clouds_id))
            self.constraints.append(self.gurobi_solver_model.addConstrs(gp.quicksum(self.select_image[i] for i in self.cloud_covered_by_image.keys()
                                                     if c in self.cloud_covered_by_image[i]) <=
                                         self.cloud_covered[c] * len(self.images) for c in self.clouds_id))

        # Resolution constraints - only add if min_resolution objective is used
        if "min_resolution" in objectives_to_use:
            big_resolution = max(self.resolution.values()) + 1
            for element in self.elements:
                total_auxiliary_variables = len(self.auxiliary_variables_for_resolution[element])
                self.constraints.append(
                    self.gurobi_solver_model.addConstr(gp.quicksum(
                        self.auxiliary_variables_for_resolution[element][i] for i in
                        self.auxiliary_variables_for_resolution[element]) == total_auxiliary_variables - 1,
                                                name=f"constraint_auxiliary_variables_for_resolution{element}"))

            self.constraints.append(self.gurobi_solver_model.addConstrs(self.resolution_element[e] >=
                                                                 self.resolution[i] * self.select_image[i] +
                                                                 big_resolution * (1 - self.select_image[i]) -
                                                                 2 * big_resolution * (
                                                                     self.auxiliary_variables_for_resolution[e][i])
                                                                 for e in self.elements
                                                                 for i in self.images_id
                                                                 if e in self.images[i]))

        # Incidence angle constraints - only add if min_max_incidence_angle objective is used
        if "min_max_incidence_angle" in objectives_to_use:
            # The below approach using indicator constraints is faster than the one commented below
            self.constraints.append(self.gurobi_solver_model.addConstrs(((self.select_image[i] == 0) >> (self.effective_incidence_angle[i] == 0)
                                  for i in self.images_id)))
            self.constraints.append(self.gurobi_solver_model.addConstrs(
                ((self.select_image[i] == 1) >> (self.effective_incidence_angle[i] == self.incidence_angle[i])
                 for i in self.images_id)))
            # Approach not using indicator constraints, it is slower than the one above
            # self.model.addConstrs(self.effective_incidence_angle[i] == self.select_image[i] * self.incidence_angle[i]
            #                       for i in self.images_id)
            self.constraints.append(self.gurobi_solver_model.addConstr(self.current_max_incidence_angle == max_(self.effective_incidence_angle[i]
                                                                                 for i in self.images_id)))
        # constraints end--------------------------------------------------------------

    def get_solution_values(self) -> list[int]:
        selected_images = []
        for image in self.select_image.keys():
            if abs(self.select_image[image].x) > 1e-6:
                selected_images.append(image)
        return selected_images

    # for testing only
    def print_solution_values_model_values(self) -> None:
        for image in self.select_image.keys():
            if abs(self.select_image[image].x) > 1e-6:
                print(f"Image {image} is selected with a value of {self.select_image[image].x}")

    def add_necessary_solver_configuration(self) -> None:
        print("Extra solver configuration needed")
        self.gurobi_solver_model.Params.IntegralityFocus = 1
