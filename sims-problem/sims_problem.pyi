"""
Type stubs for sims-problem

This package provides Python bindings for the SIMS (Satellite Image Mosaicking Selection) problem solver
implemented in Rust. It offers both heuristic (Pareto Local Search) and exact (MILP with AUGMECON) 
algorithms for multi-objective optimization.

Key Classes:
- SimsDiscreteProblem: Represents a SIMS problem instance with satellite images, coverage areas, costs, etc.
- Solution: Represents a solution containing selected images and objective values

Key Functions:
- solve_with_pls: Solves using Pareto Local Search (heuristic, fast)
- solve_with_milp: Solves using Mixed Integer Linear Programming (exact, slower)
"""

from __future__ import annotations

from typing import Optional, Any, overload
from datetime import timedelta

class SimsDiscreteProblem:
    """
    Represents a SIMS (Satellite Image Mosaicking Selection) discrete problem instance.
    
    The SIMS problem involves selecting a subset of satellite images to cover a geographic area
    while optimizing multiple objectives such as cost, cloud coverage, resolution, and incidence angle.
    
    Attributes:
        num_images: Number of available satellite images
        universe: Total number of geographic fragments/tiles to be covered
        images: List of image coverage sets (which fragments each image covers)
        costs: Cost of acquiring each image
        clouds: Cloud coverage for each image (which fragments are cloudy)
        areas: Area/importance of each geographic fragment
        resolution: Resolution quality of each image
        incidence_angle: Viewing angle of each image (affects quality)
        max_cloud_area: Maximum allowed total cloud area in solution
    """
    
    # Properties (can be read and written)
    num_images: int
    universe: int
    images: list[list[int]]
    costs: list[int]
    clouds: list[list[int]]
    areas: list[int]
    resolution: list[int]
    incidence_angle: list[int]
    max_cloud_area: int
    
    def __init__(
        self,
        num_images: int,
        universe: int,
        images: list[list[int]],
        costs: list[int],
        clouds: list[list[int]],
        areas: list[int],
        resolution: list[int],
        incidence_angle: list[int],
        max_cloud_area: int,
    ) -> None:
        """
        Create a new SIMS discrete problem instance.
        
        Args:
            num_images: Number of available satellite images
            universe: Total number of geographic fragments to cover
            images: List of coverage sets - images[i] contains fragment indices covered by image i
            costs: List of acquisition costs for each image
            clouds: List of cloud coverage sets - clouds[i] contains cloudy fragment indices in image i
            areas: List of area values for each fragment
            resolution: List of resolution values for each image (higher = better quality)
            incidence_angle: List of incidence angle values for each image (affects quality)
            max_cloud_area: Maximum allowed total cloud area in any solution
            
        Note:
            All fragment indices in images and clouds should be 0-based and < universe.
            The union of all image fragments should cover all indices 0 to universe-1.
        """
        ...
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimsDiscreteProblem:
        """
        Create a SimsDiscreteProblem from a dictionary.
        
        Args:
            data: Dictionary containing all required problem data with keys matching constructor parameters
            
        Returns:
            New SimsDiscreteProblem instance
            
        Raises:
            ValueError: If required keys are missing or data is invalid
        """
        ...
    
    @classmethod
    def from_dzn(cls, file_path: str) -> SimsDiscreteProblem:
        """
        Create a SimsDiscreteProblem from a MiniZinc data file (.dzn).
        
        Automatically converts from 1-based indexing (MiniZinc convention) to 0-based indexing (Python convention).
        
        Args:
            file_path: Path to the .dzn file containing problem data
            
        Returns:
            New SimsDiscreteProblem instance
            
        Raises:
            ValueError: If file cannot be read or contains invalid data
        """
        ...
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert the problem to a dictionary representation.
        
        Returns:
            Dictionary containing all problem data suitable for serialization
        """
        ...
    
    def get_max_values(self) -> tuple[int, int]:
        """
        Get maximum possible values for cost and area objectives.
        
        Returns:
            Tuple of (max_cost, max_area) where max_cost is sum of all image costs
            and max_area is sum of all fragment areas
        """
        ...
    
    def get_ref_point(self) -> tuple[int, int]:
        """
        Get reference point for Pareto optimization (max values + 1).
        
        Returns:
            Tuple of (max_cost + 1, max_area + 1) suitable as reference point for Pareto algorithms
        """
        ...
    
    def validate(self) -> None:
        """
        Validate the problem instance for consistency and completeness.
        
        Checks:
        - All list lengths match expected dimensions
        - Images cover all universe fragments exactly once
        - All indices are within valid ranges
        
        Raises:
            ValueError: If validation fails with description of the issue
        """
        ...
    
    def get_image_fragments(self, image_idx: int) -> list[int]:
        """
        Get the geographic fragments covered by a specific image.
        
        Args:
            image_idx: Index of the image (0-based)
            
        Returns:
            List of fragment indices covered by this image
            
        Raises:
            IndexError: If image_idx is out of bounds
        """
        ...
    
    def get_cloud_fragments(self, image_idx: int) -> list[int]:
        """
        Get the cloud-covered fragments for a specific image.
        
        Args:
            image_idx: Index of the image (0-based)
            
        Returns:
            List of fragment indices that are cloudy in this image
            
        Raises:
            IndexError: If image_idx is out of bounds
        """
        ...
    
    def is_fragment_cloudy(self, image_idx: int, fragment_idx: int) -> bool:
        """
        Check if a specific fragment is covered by clouds in a specific image.
        
        Args:
            image_idx: Index of the image (0-based)
            fragment_idx: Index of the fragment (0-based)
            
        Returns:
            True if the fragment is cloudy in this image, False otherwise
            
        Raises:
            IndexError: If either index is out of bounds
        """
        ...
    
    def calculate_total_cost(self, selected_images: list[int]) -> int:
        """
        Calculate total cost for a set of selected images.
        
        Args:
            selected_images: List of image indices
            
        Returns:
            Sum of costs for all selected images
            
        Raises:
            IndexError: If any image index is out of bounds
        """
        ...
    
    def calculate_total_cloud_area(self, selected_images: list[int]) -> int:
        """
        Calculate total cloud area for a set of selected images.
        
        Args:
            selected_images: List of image indices
            
        Returns:
            Sum of areas for all cloud-covered fragments across selected images
            
        Raises:
            IndexError: If any image index is out of bounds
        """
        ...
    
    def __repr__(self) -> str:
        """String representation showing key problem dimensions."""
        ...
    
    def __str__(self) -> str:
        """String representation showing key problem dimensions."""
        ...


class Solution:
    """
    Represents a solution to the SIMS problem.
    
    A solution contains the set of selected satellite images and the resulting
    objective values (cost, cloudy area coverage, and optionally resolution and incidence angle).
    
    Attributes:
        selected_images: Set of selected image indices
        cost: Total cost of selected images
        cloudy_area: Total cloudy area in the solution
        timestamp: Time when solution was found (as timedelta from start)
        max_incidence_angle: Maximum incidence angle among selected images (optional)
        min_resolutions_sum: Sum of minimum resolutions for all fragments (optional)
    """
    
    # Properties (can be read and written)
    selected_images: set[int]
    cost: int
    cloudy_area: int
    timestamp: timedelta
    max_incidence_angle: Optional[int]
    min_resolutions_sum: Optional[int]
    
    def __new__(cls) -> None:
        """
        Cannot create Solution directly. Use Solution.create() instead.
        
        Raises:
            ValueError: Always, directing users to use Solution.create()
        """
        ...
    
    @staticmethod
    def create(
        selected_images: list[int],
        cost: int,
        cloudy_area: int,
        timestamp_us: int,
        max_incidence_angle: Optional[int] = None,
        min_resolutions_sum: Optional[int] = None,
    ) -> Solution:
        """
        Create a new solution instance.
        
        Args:
            selected_images: List of selected image indices
            cost: Total cost of the solution
            cloudy_area: Total cloudy area in the solution
            timestamp_us: Timestamp in microseconds from algorithm start
            max_incidence_angle: Maximum incidence angle among selected images (for 4D optimization)
            min_resolutions_sum: Sum of minimum resolutions per fragment (for 4D optimization)
            
        Returns:
            New Solution instance
        """
        ...
    
    def to_json(self) -> dict[str, Any]:
        """
        Convert solution to JSON-compatible dictionary.
        
        Returns:
            Dictionary with all solution data, suitable for JSON serialization
        """
        ...
    
    def get_selected_images_list(self) -> list[int]:
        """
        Get selected images as a sorted list.
        
        Returns:
            List of selected image indices in ascending order
        """
        ...
    
    def add_image(self, image_idx: int) -> None:
        """
        Add an image to the selection.
        
        Args:
            image_idx: Index of image to add
        """
        ...
    
    def remove_image(self, image_idx: int) -> bool:
        """
        Remove an image from the selection.
        
        Args:
            image_idx: Index of image to remove
            
        Returns:
            True if image was present and removed, False if not present
        """
        ...
    
    def contains_image(self, image_idx: int) -> bool:
        """
        Check if an image is in the selection.
        
        Args:
            image_idx: Index of image to check
            
        Returns:
            True if image is selected, False otherwise
        """
        ...
    
    def num_selected_images(self) -> int:
        """
        Get the number of selected images.
        
        Returns:
            Count of selected images
        """
        ...
    
    def validate(self, problem: SimsDiscreteProblem) -> bool:
        """
        Validate the solution against a problem instance.
        
        Checks:
        - All universe fragments are covered by selected images
        - All image indices are valid
        - Objective values are correctly computed
        
        Args:
            problem: Problem instance to validate against
            
        Returns:
            True if solution is valid, False otherwise
            
        Raises:
            IndexError: If any image index is out of bounds
        """
        ...
    
    def compute_objectives(self, problem: SimsDiscreteProblem) -> tuple[int, int, int, int]:
        """
        Compute all objective values for this solution.
        
        Args:
            problem: Problem instance to compute objectives for
            
        Returns:
            Tuple of (total_cost, cloudy_area, max_incidence_angle, min_resolutions_sum)
            
        Raises:
            IndexError: If any selected image index is out of bounds
        """
        ...
    
    def validate_objectives(self, problem: SimsDiscreteProblem) -> bool:
        """
        Validate that stored objective values match computed values.
        
        Args:
            problem: Problem instance to validate against
            
        Returns:
            True if all stored objectives match computed values, False otherwise
        """
        ...
    
    def fix_objectives(self, problem: SimsDiscreteProblem) -> None:
        """
        Recompute and fix invalid objective values.
        
        Args:
            problem: Problem instance to compute objectives from
        """
        ...


@overload
def solve_with_pls(
    sims_instance: SimsDiscreteProblem,
    objectives: list[str] = ["min_cost", "cloud_coverage"],
    *,
    plots: bool = False,
    timeout_seconds: float = 240.0,
    max_iterations: int = 50000,
    is_deterministic: bool = False,
    initial_population_size: int = 100,
    neighborhood_size_min: int = 1,
    neighborhood_size_max: int = 6,
) -> list[Solution]:
    """
    Solve the SIMS problem using Pareto Local Search without plotting.
    """
    ...


@overload
def solve_with_pls(
    sims_instance: SimsDiscreteProblem,
    objectives: list[str] = ["min_cost", "cloud_coverage"],
    *,
    plots: bool = True,
    plot_output_path: Optional[str] = None,
    timeout_seconds: float = 240.0,
    max_iterations: int = 50000,
    is_deterministic: bool = False,
    initial_population_size: int = 100,
    neighborhood_size_min: int = 1,
    neighborhood_size_max: int = 6,
) -> list[Solution]:
    """
    Solve the SIMS problem using Pareto Local Search with plotting enabled.
    """
    ...


def solve_with_pls(
    sims_instance: SimsDiscreteProblem,
    objectives: list[str] = ["min_cost", "cloud_coverage"],
    plots: bool = False,
    plot_output_path: Optional[str] = None,
    timeout_seconds: float = 240.0,
    max_iterations: int = 50000,
    is_deterministic: bool = False,
    initial_population_size: int = 100,
    neighborhood_size_min: int = 1,
    neighborhood_size_max: int = 6,
) -> list[Solution]:
    """
    Solve the SIMS problem using Pareto Local Search (heuristic algorithm).
    
    This is a fast heuristic algorithm that finds good approximate Pareto-optimal solutions.
    Supports both 2D optimization (cost + cloud coverage) and 4D optimization 
    (cost + cloud coverage + resolution + incidence angle).
    
    The function has two overloads:
    1. plots=False (default): No plotting, plot_output_path is ignored
    2. plots=True: Plotting enabled, plot_output_path can specify custom path
    
    Args:
        sims_instance: The SIMS problem instance to solve
        objectives: List of objectives to optimize. Valid values:
            - "min_cost": Minimize total acquisition cost
            - "cloud_coverage": Minimize total cloudy area
            - "min_resolution": Minimize total resolution (better resolution = lower value)
            - "max_incidence_angle": Minimize maximum incidence angle
        plots: Whether to generate plots of the Pareto front (requires plotting feature)
        plot_output_path: Custom path for plot output file (only when plots=True).
            If None and plots=True, uses default naming (e.g., "pareto_solutions_2d.svg")
        timeout_seconds: Maximum runtime in seconds
        max_iterations: Maximum number of algorithm iterations
        is_deterministic: Whether to use deterministic random seed for reproducible results
        initial_population_size: Size of initial random population
        neighborhood_size_min: Minimum neighborhood size for local search
        neighborhood_size_max: Maximum neighborhood size for local search
        
    Returns:
        List of non-dominated solutions found by the algorithm
        
    Raises:
        ValueError: If invalid objectives are specified or other parameter validation fails
        
    Examples:
        # Basic 2D optimization (cost + cloud coverage)
        solutions = solve_with_pls(problem)
        
        # 4D optimization with all objectives
        solutions = solve_with_pls(
            problem,
            objectives=["min_cost", "cloud_coverage", "min_resolution", "max_incidence_angle"],
            timeout_seconds=60.0,
            max_iterations=10000
        )
        
        # Deterministic run for testing
        solutions = solve_with_pls(problem, is_deterministic=True)
        
        # Generate plots with default naming
        solutions = solve_with_pls(problem, plots=True)
        
        # Generate plots with custom output path
        solutions = solve_with_pls(
            problem, 
            plots=True, 
            plot_output_path="my_pareto_front.svg"
        )
    """
    ...


def solve_with_milp(
    sims_instance: SimsDiscreteProblem,
    objectives: list[str] = ["min_cost", "cloud_coverage"],
    grid_points: int = 50,
    timeout_seconds: float = 300.0,
    bypass_coefficient: bool = True,
    early_exit: bool = True,
    flag_array: bool = True,
    solver_name: str = "cbc",
) -> list[Solution]:
    """
    Solve the SIMS problem using Mixed Integer Linear Programming with AUGMECON (exact algorithm).
    
    This is an exact algorithm that finds all Pareto-optimal solutions within the specified
    grid resolution. Much slower than PLS but guarantees optimality.
    
    Args:
        sims_instance: The SIMS problem instance to solve
        objectives: List of objectives to optimize. Valid values:
            - "min_cost": Minimize total acquisition cost
            - "cloud_coverage": Minimize total cloudy area
            - "min_resolution": Minimize total resolution
            - "max_incidence_angle": Minimize maximum incidence angle
        grid_points: Number of grid points for epsilon-constraint method (higher = finer resolution)
        timeout_seconds: Maximum runtime in seconds (may not be strictly enforced)
        bypass_coefficient: Enable bypass coefficient optimization for faster solving
        early_exit: Enable early exit when no improvements are found
        flag_array: Enable flag array optimization for constraint handling
        solver_name: MILP solver to use ("cbc" is currently supported)
        
    Returns:
        List of Pareto-optimal solutions found by the algorithm
        
    Raises:
        ValueError: If invalid objectives are specified or solver setup fails
        
    Examples:
        # Basic 2D MILP optimization
        solutions = solve_with_milp(problem, grid_points=25, timeout_seconds=120.0)
        
        # High-resolution 4D optimization
        solutions = solve_with_milp(
            problem,
            objectives=["min_cost", "cloud_coverage", "min_resolution", "max_incidence_angle"],
            grid_points=100,
            timeout_seconds=600.0
        )
    """
    ...


# Module-level exports
__all__ = [
    "SimsDiscreteProblem",
    "Solution", 
    "solve_with_pls",
    "solve_with_milp",
]
