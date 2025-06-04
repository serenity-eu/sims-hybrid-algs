import itertools
import logging
import dataclasses
import heapq

from sims_solvers.FrontGenerators.FrontGeneratorStrategy import FrontGeneratorStrategy

log = logging.getLogger(__name__)

@dataclasses.dataclass(order=True)
class SortablePair:
    """
    A class to represent a pair of solutions with distance between them as key, for sorting.
    """
    points_pair: tuple[tuple[int], tuple[int]] = dataclasses.field(compare=False)
    squared_distance: int

class DistancePriorityQueue:
    """
    A priority queue implementation using a heap, where points pairs sorted by their squared distance in descending order.
    """

    def __init__(self):
        self._heap: list[SortablePair] = []

    def _squared_distance(self, point1: list[int], point2: list[int]):
        """
        Compute the squared distance between two objective vectors point1 and point2.
        This is used to determine the closeness of two solutions in the objective space.
        """
        return sum((a - b) ** 2 for a, b in zip(point1, point2))


    def push(self, point1: tuple[int], point2: tuple[int]):
        """
        Add a pair of points to the priority queue, computing the distance between them.
        """
        if type(point1) != tuple or type(point2) != tuple:
            raise TypeError(f"Both point1 and point2 must be tuples. Got {type(point1)} and {type(point2)} instead.")
        # NOTE: We negate the squared distance to use a min-heap as a max-heap.
        squared_distance = -self._squared_distance(list(point1), list(point2))
        pair = SortablePair(points_pair=(point1, point2), squared_distance=squared_distance)
        heapq.heappush(self._heap, pair)

    def pop(self) -> tuple[tuple[int], tuple[int]]:
        """
        Remove and return the SortablePair with the largest distance.
        """
        return heapq.heappop(self._heap).points_pair

    def is_empty(self) -> bool:
        """
        Check if the priority queue is empty.
        """
        return len(self._heap) == 0

class AnejaNair(FrontGeneratorStrategy):
    """
    A concrete front generator strategy using the Aneja and Nair iterative dichotomic method.
    It computes the supported efficient solutions for a bi-objective problem by
    repeatedly solving weighted single-objective problems.
    """

    def set_multiply_solution_by_minus_one(self):
        """
        For minimization models (like the set covering problem), no transformation is required.
        For maximization models, we might need to multiply the solution objectives by -1.
        """
        return False if self.solver.model.is_a_minimization_model() else True

    def always_add_new_solutions_to_front(self):
        """
        In the dichotomic method every new solution is a candidate to be part of the Pareto front.
        """
        return True

    @staticmethod
    def compute_weights(z1: list[int], z2: list[int]):
        """
        Given two objective vectors z1 = [a, b] and z2 = [c, d],
        compute a weights vector using the normal vector to the line joining them.
        
        A normal vector can be computed as (d - b, a - c). If any component is negative,
        we flip the signs to guarantee nonnegativity, and then we normalize so that w1 + w2 = 1.
        """
        w1 = z2[1] - z1[1]
        w2 = z1[0] - z2[0]
        if w1 < 0 or w2 < 0:
            w1, w2 = -w1, -w2
        s = w1 + w2
        if s == 0:
            return (0.5, 0.5)
        return (w1 / s, w2 / s)
    
    
    def solve(self):
        """
        Main method to generate the front of supported efficient solutions
        using an iterative dichotomic procedure.
        
        The method works by:
          1. Obtaining the two extreme solutions using weights (1, 0) and (0, 1).
          2. Maintaining a list of intervals (pairs of consecutive solutions).
          3. For each interval, computing the weight vector (via compute_weight), solving
             the weighted problem, and, if a new and distinct solution is found, inserting it.
          4. Repeating until no more new solutions are found or a timeout is hit
        """

        # Step 1: Solve for the extreme solutions.
        # In the case of 2 objectives, it will return the extreme points of the Pareto front
        log.info("Computing the edge solutions.")
        best_worst_list = list(self.get_best_worst_values())

        # Step 2: Initialize explored intervals dictionary.
        explored_intervals: dict[tuple[tuple[int], tuple[int]], tuple[int] | None] = {}
        # Initialize priority queue for distance computation.
        distance_queue = DistancePriorityQueue()

        # Add the extreme solutions to the distance queue.
        for (left, right) in itertools.pairwise(best_worst_list):
            left_obj = tuple(left['objs'])
            right_obj = tuple(right['objs'])
            distance_queue.push(left_obj, right_obj)

        for solution in best_worst_list:
            yield solution


        # Step 3: Iteratively refine the Pareto front.
        log.info("Iteratively refining the Pareto front.")
        has_new_solution = True
        while has_new_solution:
            has_new_solution = False

            while not distance_queue.is_empty():
                left_obj, right_obj = distance_queue.pop()
                weights = self.compute_weights(list(left_obj), list(right_obj))
                log.info(f"Computed weight vector: {weights}")

                self.solver.set_weighted_sum_objective(weights)
                new_time = self.get_solver_solution_for_timeout(optimize_not_satisfy=True)
                new_solution = self.process_feasible_solution(new_time)

                # Check if the new solution is None or identical to the left or right solutions.
                if new_solution is None or tuple(new_solution['objs']) in [left_obj, right_obj]:
                    explored_intervals[(left_obj, right_obj)] = None
                    continue

                # This should signal that we can stop searching.
                if self.solver.status_infeasible():
                    has_new_solution = False
                else:
                    has_new_solution = True

                yield new_solution
                new_solution_obj = tuple(new_solution['objs'])
                explored_intervals[(left_obj, right_obj)] = new_solution_obj
                # Add 2 new intervals to the distance queue.
                distance_queue.push(left_obj, new_solution_obj)
                distance_queue.push(new_solution_obj, right_obj)


    def get_best_worst_values(self):
        if len(self.solver.model.objectives) == 2:
            all_solutions = self.get_best_worst_for_2obj_lexicographically()
            for formatted_solution in all_solutions:
                if formatted_solution is not None:
                    yield formatted_solution
                else:
                    raise TimeoutError()
