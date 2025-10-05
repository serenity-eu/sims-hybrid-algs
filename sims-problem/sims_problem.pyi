from __future__ import annotations

from datetime import timedelta
from typing import Optional, overload, Literal, Any

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


class SolvingResult:
    """
    Results from solving a SIMS problem instance.
    
    Contains the final Pareto-optimal solutions and optionally a binary trace archive.
    
    Attributes:
        final_solutions: List of final Pareto-optimal solutions found
        trace: Optional binary trace archive of the optimization process
    """
    
    # Properties (can be read and written)
    final_solutions: list[Solution]
    trace: Optional[bytes]
    
    def __new__(
        cls,
        final_solutions: list[Solution],
    ) -> SolvingResult:
        """
        Create a new SolvingResult without trace.
        
        Args:
            final_solutions: Final Pareto-optimal solutions
        """
        ...
    
    @staticmethod
    def with_trace(
        final_solutions: list[Solution],
        trace: bytes
    ) -> SolvingResult:
        """
        Create a new SolvingResult with trace archive.
        
        Args:
            final_solutions: Final Pareto-optimal solutions
            trace: Binary trace archive of optimization process
        """
        ...


class MilpConfig:
    """
    Configuration class for MILP solver parameters.
    
    This class encapsulates all parameters needed to configure the MILP solver
    using the AUGMECON method for multi-objective optimization.
    
    Note: Timeout is not specified here as it's calculated by solve_with_hybrid 
    based on the ratio parameter.
    """
    
    objectives: list[str]
    grid_points: int 
    bypass_coefficient: bool
    early_exit: bool
    flag_array: bool
    solver_name: str
    
    def __init__(
        self,
        objectives: list[str] = ["min_cost", "cloud_coverage"],
        grid_points: int = 50,
        bypass_coefficient: bool = True,
        early_exit: bool = True,
        flag_array: bool = True,
        solver_name: str = "cbc",
    ) -> None:
        """
        Initialize MILP configuration.
        
        Args:
            objectives: List of objectives to optimize (e.g., ["min_cost", "cloud_coverage"])
            grid_points: Number of grid points for epsilon-constraint method
            bypass_coefficient: Enable AUGMECON2 bypass coefficient optimization
            early_exit: Enable early exit when no more solutions can be found
            flag_array: Enable AUGMECON-R flag array optimization  
            solver_name: Name of the MILP solver backend to use
        """
        ...


class PlsConfig:
    """
    Configuration class for PLS (Pareto Local Search) solver parameters.
    
    This class encapsulates all parameters needed to configure the PLS heuristic
    algorithm for multi-objective optimization.
    
    Note: Timeout is not specified here as it's calculated by solve_with_hybrid 
    based on the ratio parameter.
    """
    
    objectives: list[str]
    max_iterations: int
    is_deterministic: bool
    initial_population_size: int
    neighborhood_size_min: int
    neighborhood_size_max: int
    plots: bool
    plot_output_path: Optional[str]
    
    def __init__(
        self,
        objectives: list[str] = ["min_cost", "cloud_coverage"],
        max_iterations: int = 50000,
        is_deterministic: bool = False,
        initial_population_size: int = 100,
        neighborhood_size_min: int = 1,
        neighborhood_size_max: int = 6,
        plots: bool = False,
        plot_output_path: Optional[str] = None,
    ) -> None:
        """
        Initialize PLS configuration.
        
        Args:
            objectives: List of objectives to optimize (e.g., ["min_cost", "cloud_coverage"])
            max_iterations: Maximum number of PLS iterations
            is_deterministic: Whether to use deterministic random seed for reproducibility
            initial_population_size: Size of initial solution population
            neighborhood_size_min: Minimum neighborhood size for local search
            neighborhood_size_max: Maximum neighborhood size for local search
            plots: Whether to generate visualization plots
            plot_output_path: Path where to save plot files (if plots=True)
        """
        ...


@overload
def solve_with_pls(
    sims_instance: SimsDiscreteProblem,
    objectives: list[str] = ["min_cost", "cloud_coverage"],
    plots: bool = False,
    plot_output_path: Optional[str] = None,
    timeout: timedelta = timedelta(seconds=240),
    max_iterations: int = 50000,
    is_deterministic: bool = False,
    initial_population_size: int = 100,
    initial_population: Optional[list[Solution]] = None,
    neighborhood_size_min: int = 1,
    neighborhood_size_max: int = 6,
    *,
    trace: Literal[False]
) -> SolvingResult: ...

@overload
def solve_with_pls(
    sims_instance: SimsDiscreteProblem,
    objectives: list[str] = ["min_cost", "cloud_coverage"],
    plots: bool = False,
    plot_output_path: Optional[str] = None,
    timeout: timedelta = timedelta(seconds=240),
    max_iterations: int = 50000,
    is_deterministic: bool = False,
    initial_population_size: int = 100,
    initial_population: Optional[list[Solution]] = None,
    neighborhood_size_min: int = 1,
    neighborhood_size_max: int = 6,
    *,
    trace: Literal[True] = True
) -> SolvingResult: ...

def solve_with_pls(
    sims_instance: SimsDiscreteProblem,
    objectives: list[str] = ["min_cost", "cloud_coverage"],
    plots: bool = False,
    plot_output_path: Optional[str] = None,
    timeout: timedelta = timedelta(seconds=240),
    max_iterations: int = 50000,
    is_deterministic: bool = False,
    initial_population_size: int = 100,
    initial_population: Optional[list[Solution]] = None,
    neighborhood_size_min: int = 1,
    neighborhood_size_max: int = 6,
    trace: bool = True
) -> SolvingResult:
    """
    Solve the SIMS problem using Pareto Local Search (heuristic algorithm).
    
    This is a fast heuristic algorithm that finds good approximate Pareto-optimal solutions.
    Supports both 2D optimization (cost + cloud coverage) and 4D optimization 
    (cost + cloud coverage + resolution + incidence angle).
    
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
        timeout: Maximum runtime as timedelta
        max_iterations: Maximum number of algorithm iterations
        is_deterministic: Whether to use deterministic random seed for reproducible results
        initial_population_size: Size of initial random population (used only if initial_population is None)
        initial_population: Optional list of Solution objects to use as initial population.
            If provided, these solutions will be used to initialize the search.
            If None, random solutions will be generated using initial_population_size.
        neighborhood_size_min: Minimum neighborhood size for local search
        neighborhood_size_max: Maximum neighborhood size for local search
        trace: Whether to generate optimization trace archive (default True)
        
    Returns:
        When trace=True (default): SolvingResult containing solutions and trace
        When trace=False: List of non-dominated solutions only (for backward compatibility)
        
    Raises:
        ValueError: If invalid objectives are specified or other parameter validation fails
        
    Examples:
        # Basic 2D optimization with trace (default)
        result = solve_with_pls(problem)
        solutions = result.final_solutions
        
        # Get trace archive
        result = solve_with_pls(problem, trace=True)
        if result.trace:
            with open("trace.gz", "wb") as f:
                f.write(result.trace)
        
        # Backward compatible: no trace, just solutions
        solutions = solve_with_pls(problem, trace=False)
        
        # 4D optimization with trace
        result = solve_with_pls(
            problem,
            objectives=["min_cost", "cloud_coverage", "min_resolution", "max_incidence_angle"],
            timeout=timedelta(seconds=60),
            max_iterations=10000
        )
        
        # Generate plots with custom output path
        result = solve_with_pls(
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
    timeout: timedelta = timedelta(seconds=300),
    bypass_coefficient: bool = True,
    early_exit: bool = True,
    flag_array: bool = True,
    solver_name: str = "cbc",
) -> SolvingResult:
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
        timeout: Maximum runtime as timedelta (may not be strictly enforced)
        bypass_coefficient: Enable bypass coefficient optimization for faster solving
        early_exit: Enable early exit when no improvements are found
        flag_array: Enable flag array optimization for constraint handling
        solver_name: MILP solver to use ("cbc" is currently supported)
        
    Returns:
        When SolvingResult containing solutions
        
    Raises:
        ValueError: If invalid objectives are specified or solver setup fails
        
    Examples:
        # Basic 2D MILP optimization with result structure (default)
        result = solve_with_milp(problem, grid_points=25, timeout=timedelta(seconds=120))
        solutions = result.final_solutions
        
        # Backward compatible: no trace, just solutions
        solutions = solve_with_milp(problem, trace=False)
        
        # High-resolution 4D optimization
        result = solve_with_milp(
            problem,
            objectives=["min_cost", "cloud_coverage", "min_resolution", "max_incidence_angle"],
            grid_points=100,
            timeout=timedelta(seconds=600)
        )
    """
    ...


@overload
def solve_with_hybrid(
    sims_instance: SimsDiscreteProblem,
    milp_config: MilpConfig,
    pls_config: PlsConfig,
    ratio: tuple[int, int],
    timeout: timedelta = timedelta(seconds=300),
    *,
    trace: Literal[False]
) -> list[Solution]: ...

@overload
def solve_with_hybrid(
    sims_instance: SimsDiscreteProblem,
    milp_config: MilpConfig,
    pls_config: PlsConfig,
    ratio: tuple[int, int],
    timeout: timedelta = timedelta(seconds=300),
    *,
    trace: Literal[True] = True
) -> SolvingResult: ...

def solve_with_hybrid(
    sims_instance: SimsDiscreteProblem,
    milp_config: MilpConfig,
    pls_config: PlsConfig,
    ratio: tuple[int, int],
    timeout: timedelta = timedelta(seconds=300),
    trace: bool = True
) -> SolvingResult:
    """
    Solve the SIMS problem using a hybrid approach: MILP first, then PLS with MILP solutions as initial population.
    
    This hybrid algorithm combines the strengths of both exact (MILP) and heuristic (PLS) approaches:
    1. First phase: Runs MILP to find high-quality exact solutions
    2. Second phase: Uses MILP solutions as initial population for PLS to explore more of the solution space
    
    The total runtime is divided between the two phases according to the specified ratio.
    
    Args:
        sims_instance: The SIMS problem instance to solve
        milp_config: Configuration parameters for the MILP phase (excluding timeout)
        pls_config: Configuration parameters for the PLS phase (excluding timeout)
        ratio: Time allocation ratio (milp_percentage, pls_percentage) as integers that must sum to 100.
               For example, (30, 70) means 30% MILP time, 70% PLS time; (50, 50) means equal 50%/50% split
        timeout: Total timeout for the entire hybrid algorithm. This will be split between 
                MILP and PLS phases according to the ratio parameter.
        trace: Whether to generate optimization trace archive (default True, but not yet implemented for hybrid)
    
    Returns:
        When trace=True (default): SolvingResult containing solutions (no trace for hybrid yet)
        When trace=False: List of solutions only (for backward compatibility)
        
    Raises:
        ValueError: If ratio values don't sum to 100, either value is non-positive,
                   configurations are invalid, or MILP and PLS objectives don't match
        
    Examples:
        # Balanced hybrid approach with result structure (default)
        result = solve_with_hybrid(problem, milp_cfg, pls_cfg, (50, 50), timeout=timedelta(seconds=300))
        solutions = result.final_solutions
        
        # Backward compatible: no trace, just solutions
        solutions = solve_with_hybrid(problem, milp_cfg, pls_cfg, (50, 50), trace=False)
        
        # MILP-heavy approach for high accuracy with 10-minute timeout
        milp_cfg = MilpConfig(grid_points=50)
        pls_cfg = PlsConfig(max_iterations=5000) 
        result = solve_with_hybrid(problem, milp_cfg, pls_cfg, (75, 25), timeout=timedelta(seconds=600))
        
        # PLS-heavy approach for exploration with 2-minute timeout
        milp_cfg = MilpConfig(grid_points=20)
        pls_cfg = PlsConfig(max_iterations=20000)
        result = solve_with_hybrid(problem, milp_cfg, pls_cfg, (20, 80), timeout=timedelta(seconds=120))
    
    Note:
        - Both MILP and PLS configs must use the same objectives list
        - Ratio values must be positive integers that sum exactly to 100 (percentages)
        - If MILP finds no solutions, the algorithm falls back to PLS-only mode
        - The hybrid approach typically finds more diverse solutions than either method alone
        - MILP timeout = timeout * (ratio[0] / 100)
        - PLS timeout = timeout * (ratio[1] / 100)
        - Trace generation is not yet implemented for hybrid algorithm
    """
    ...


def compute_hypervolume(
    solutions: list[Solution] | list[list[int]],
    objective_bounds: list[list[int]],
    reference_point: list[int] | None = None,
    normalized: bool = False
) -> float:
    """
    Compute the hypervolume indicator for a set of solutions or points.
    
    The hypervolume indicator measures the volume of the objective space dominated
    by a set of solutions, bounded by a reference point. It's a key quality metric
    for multi-objective optimization algorithms.
    
    This unified function automatically detects the input type and dimension (2D, 3D, or 4D)
    and applies the appropriate computation algorithm.
    
    Args:
        solutions: Either a list of Solution objects or a list of points (each point is a list of ints).
                  For Solution objects, objectives are extracted using objectives_2d(), objectives_3d(), 
                  or objectives_4d() based on the objective bounds dimension.
        objective_bounds: Bounds for each dimension. Format: [[min1, max1], [min2, max2], ...]
                         for each dimension. Used for scaling when scaled=True and determines
                         the problem dimension.
        reference_point: Optional reference point for hypervolume computation. If not provided,
                        computed as the maximum bounds [max1, max2, ...]. Must be within or
                        dominated by all solutions for meaningful results.
        scaled: If True, scales all points to [0, 1000] range using objective_bounds
               before computing hypervolume. This preserves dominance relationships while
               normalizing the scale. Default is False.
    
    Returns:
        The hypervolume value as an int. When scaled=True, the result is in the
        normalized [0, 1000] coordinate space.
        
    Raises:
        ValueError: If dimension is not 2, 3, or 4, or if inputs are inconsistent,
                   or if objective_bounds format is invalid, or if reference_point
                   dimension doesn't match objective_bounds
        TypeError: If negative coordinate values are encountered (use non-negative coordinates)
        
    Examples:
        # Basic usage with bounds only (reference computed as max bounds)
        solutions = [sol1, sol2, sol3]  # Solution objects
        bounds = [[0, 200], [0, 100]]  # [min, max] for each dimension
        hv = compute_hypervolume(solutions, bounds)
        
        # Using raw points with explicit reference point
        points = [[10, 20], [30, 40]]
        bounds = [[0, 100], [0, 100]]
        reference = [80, 90]
        hv = compute_hypervolume(points, bounds, reference_point=reference)
        
        # With scaling for normalized comparison
        hv_normalized = compute_hypervolume(solutions, bounds, normalized=True)
        
        # 3D optimization with custom reference point
        bounds_3d = [[0, 100], [0, 50], [0, 200]]
        reference_3d = [80, 40, 180]
        hv_3d = compute_hypervolume(solutions, bounds_3d, reference_point=reference_3d)
        
        # 4D with auto-computed reference (max bounds)
        bounds_4d = [[0, 100], [0, 50], [0, 200], [0, 90]]
        hv_4d = compute_hypervolume(solutions, bounds_4d)  # reference = [100, 50, 200, 90]
        
    Notes:
        - For Solution objects, the objective bounds dimension determines which objectives are used:
          * 2D bounds → calls objectives_2d() 
          * 3D bounds → calls objectives_3d()
          * 4D bounds → calls objectives_4d()
        - When reference_point is not provided, it's computed as [max1, max2, ...] from objective_bounds
        - Points outside the bounds are clamped when normalized=True
        - The normalized=True option normalizes points to [0,1] range like pymoo for cross-validation
        - objective_bounds is now required for consistent and predictable hypervolume computation
        - All coordinates must be non-negative for meaningful hypervolume computation
        - The reference point should be dominated by all input solutions/points
        - When scaled=True, normalization preserves the relative dominance structure
        - Uses optimized sweep-line algorithms with u128 precision for accuracy
    """
    ...


def generate_trace(
    solutions: list[Solution],
    objectives: list[str], 
    algorithm: str,
    num_objectives: int,
    objective_bounds: list[list[int]],
    reference_point: list[int]
) -> bytes:
    """
    Creates a gzipped tar archive containing binary optimization trace data.
    
    This function creates a compressed archive containing detailed optimization traces
    suitable for analysis and visualization. The archive contains:
    
    - **objectives.bin**: u64 LE objectives for each solution
    - **dominated.bin**: u32 LE domination indices (u32::MAX = not dominated)  
    - **timestamp.bin**: u32 LE timestamps in microseconds since start
    - **hypervolume.bin**: f64 LE cumulative hypervolume progression
    - **metadata.json**: JSON metadata about the trace including objective names
    
    Args:
        solutions: List of Solution objects (must be sorted by timestamp)
        objectives: Names of the objectives (e.g., ["min_cost", "cloud_coverage", "max_incidence_angle"])
        algorithm: Name of the algorithm that generated the solutions
        num_objectives: Number of objectives (2, 3, or 4)
        objective_bounds: Bounds for each objective [[min, max], ...]
        reference_point: Reference point for hypervolume calculation
        
    Returns:
        Compressed tar archive as bytes
        
    Example:
        ```python
        import sims_problem
        
        # Create some solutions
        solutions = [
            sims_problem.Solution.create([0, 2, 5], 1500, 250, 100000, 45, 800),
            sims_problem.Solution.create([1, 3, 7], 1200, 300, 500000, 40, 900),
        ]
        
        # Create trace archive
        archive_bytes = sims_problem.generate_trace(
            solutions, ["min_cost", "cloud_coverage", "max_incidence_angle"], 
            "MILP", 3, [[0, 10000], [0, 1000], [0, 2000]], [10000, 1000, 2000]
        )
        
        # Save to file
        with open("trace.tar.gz", "wb") as f:
            f.write(archive_bytes)
        ```
    """
    ...


def merge_traces(
    first_trace: bytes,
    second_trace: bytes, 
    combined_algorithm: str,
    objective_bounds: list[list[int]],
    reference_point: list[int]
) -> bytes:
    """
    Merges two trace archives into a single archive with adjusted timestamps.
    
    This function takes two compressed trace archives and combines them into one.
    The second trace's timestamps are offset by the execution time of the first phase.
    
    Args:
        first_trace: Bytes of the first trace archive (tar.gz)
        second_trace: Bytes of the second trace archive (tar.gz)
        combined_algorithm: Algorithm name for the merged trace
        objective_bounds: Bounds for each objective [[min, max], ...]
        reference_point: Reference point for hypervolume calculation
        
    Returns:
        Bytes of the merged trace archive
        
    Example:
        ```python
        import sims_problem
        
        # Load existing trace files
        with open("trace1.tar.gz", "rb") as f:
            first_trace = f.read()
        with open("trace2.tar.gz", "rb") as f:
            second_trace = f.read()
            
        # Merge traces
        merged_trace = sims_problem.merge_traces(
            first_trace, second_trace, "two-phase-50-50",
            [[0, 10000], [0, 1000]], [10000, 1000]
        )
        
        # Save merged trace
        with open("merged_trace.tar.gz", "wb") as f:
            f.write(merged_trace)
        ```
    """
    ...


# Module-level exports
__all__ = [
    "SimsDiscreteProblem",
    "Solution", 
    "SolvingResult",
    "MilpConfig",
    "PlsConfig",
    "solve_with_pls",
    "solve_with_milp", 
    "solve_with_hybrid",
    "compute_hypervolume",
    "generate_trace",
    "merge_traces",
]
