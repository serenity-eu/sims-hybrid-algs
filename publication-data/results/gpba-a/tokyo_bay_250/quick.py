from serenity.core import ProblemInstance

problem_instance = ProblemInstance.from_dzn("tokyo_bay_250.dzn")
print(problem_instance.problem.num_images)
print(problem_instance.problem.universe)