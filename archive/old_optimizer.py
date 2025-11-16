from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from archive.old_system import System
from pathlib import Path
from typing import Callable
import json
from archive.old_objective import default_obj_function
from pymoo.core.mixed import MixedVariableGA
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from utils import to_python

class Optimizer():
    """
    A multi-objective Genetic Algorithm based optimizer to automatically identify design variables
    in a System class representation of a SysML system, descritize and assemble into an OpenTorsion Assembly,
    and optimize through any custom objective function.

    Arguments:
    - obj_function [optional]: A function taking in only an openTorsion assembly object and returning a dictionary of outputs. 
      Only the output dictionary keys containing a number will be passed to optimizer for a step, rest will only be saved for final results.
    - folder_for_results [optional]: Directory to save optimization results to. "./results" by default
    - num_objectives [optional]: Number of objective (i.e. values returned by the objective function). 2 by default.
    - minimize_mask [optional]: A bool or bool array indicating which objectives need to be minmized. True by default
      If True, all objectives will be minimized, and maximized if False. When defined as an array, the order and size must match 
      the returns of obj_function. e.g. for vibration, distance_from_resonance = obj_function, it could be [True, False], to minimize 
      first and maximize second.
    - num_generations [optional]: Number of generations GA will iterative to. 10 by default.
    - num_populations [optional]: Number of populations/candidates analyzed in a single generation. 10 by default.
    - random_seed [optional]: 42 by default
    - Verbose [optional]: False by default

    Methods:
    - run(system: System): Optimizes a System class representation of a SysML system

    """

    
    def __init__(
            self, 
            system: System, 
            obj_function: Callable = default_obj_function,
            folder_for_results: str | Path = "./results",
            num_objectives: int = 2, 
            minimize_mask: list[bool] | bool = True, 
            num_generations: int = 10, 
            num_populations: int = 10, 
            random_seed: int | float = 42,
            verbose: bool = False
        ):
        
        self.system = system
        self.results_folder_path = Path(folder_for_results).resolve()
        self.results_folder_path.mkdir(parents=True, exist_ok=True)

        self.opt_history_path = self.results_folder_path / f"{self.system.name}_optimization_history.json"
        self.final_results_path = self.results_folder_path / f"{self.system.name}_final_results.json"
        self.optimized_system_path = self.results_folder_path / f"{self.system.name}_optimized.json"

        self.opt_history, self.final_results = [], []
        
        self.obj_function = obj_function
        self.num_objectives = num_objectives

        self.num_populations = num_populations
        self.num_generations = num_generations
        self.random_seed = random_seed
        self.verbose = verbose

        if isinstance(minimize_mask, bool):
            self.minimize_mask = [1] * num_objectives if minimize_mask else [-1] * num_objectives
        else: 
            if len(minimize_mask) != num_objectives:
                raise ValueError("Error: Length of minimize mask must be equal to the number of outputs of objective function")
            self.minimize_mask = [1 if m else -1 for m in minimize_mask]

        design_vars = self.system.get_design_vars()
        
        if not design_vars:
            raise ValueError("Failed to begin optimization. No design variables found.")
        
        self.algorithm = MixedVariableGA(pop_size=self.num_populations, survival=RankAndCrowdingSurvival())
        self.problem = VibrationOptimizationProblem(
            design_vars, self.system, self.obj_function, self.num_objectives, self.minimize_mask
            )


    def run(self):
        
        print("Optimization started...")
        minimize(
            self.problem, 
            self.algorithm, 
            ("n_gen", self.num_generations), 
            seed=self.random_seed, 
            verbose=self.verbose
            )
        
        self.opt_history = to_python(self.problem.results)
        self.final_results = self.opt_history[-self.num_populations:]

        with open(self.opt_history_path, "w") as f:
            json.dump(self.opt_history, f, indent=4)

        with open(self.final_results_path, "w") as f:
            json.dump(self.final_results, f, indent=4)

        print(f"Optimization finished. Results saved to {self.final_results_path}")

    
    def save_optimized_system(self, candidate_index: int):
        selected_candidate = self.opt_history[candidate_index]
        
        self.system.update(candidate=selected_candidate["design_vars"])
        self.system["optimization_results"] = selected_candidate["results"]

        with open(self.optimized_system_path, "w") as f:
            json.dump(self.system, f, indent=4)

        print(f"Optimized system saved to {self.optimized_system_path}")


class VibrationOptimizationProblem(ElementwiseProblem):
    def __init__(self, design_vars, system, obj_func, n_obj, minimize_mask):
        
        self.obj_func = obj_func
        self.system = system
        self.minimize_mask = minimize_mask
        
        self.i = 0
        self.results = []

        super().__init__(vars=design_vars, n_obj=n_obj)

    def _evaluate(self, x, out):
        
        self.system.update(candidate=x)
        full_result = self._validate_and_run_obj_func()

        candidate_name = f"{self.system.name}_candidate_{self.i}"
        self.i += 1

        self.results.append({
            "candidate": candidate_name,
            "design_vars": x,
            "results": full_result
        })

        results_for_GA = [v for v in full_result.values() if isinstance(v, (int, float))]
        if len(results_for_GA) != len(self.minimize_mask):
            raise ValueError("Mismatch between results from objective function and minimize_mask length.")
        out["F"] = [r * m for r, m in zip(results_for_GA, self.minimize_mask)]

    def _validate_and_run_obj_func(self, *args, **kwargs):
        result = self.obj_func(self.system, *args, **kwargs)
        if not isinstance(result, dict):
            raise TypeError(f"Objective function must return a dictionary, got {type(result).__name__}")
        return result




   

