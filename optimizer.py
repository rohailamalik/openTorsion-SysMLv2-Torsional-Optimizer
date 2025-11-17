from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.core.mixed import MixedVariableGA
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from parser import System, ValidationError
from pathlib import Path
from typing import Callable, Union
import json
from objective import default_obj_function
from utils import to_python


class Optimizer:
    """
    A multi-objective Genetic Algorithm optimizer for SysML system design optimization.
    
    Automatically identifies design variables in a System class representation,
    discretizes and assembles into an OpenTorsion Assembly, and optimizes using
    a custom objective function.

    Args:
        system: System class representation of a SysML system
        obj_function: Function taking a System and returning a dict of outputs.
            All objectives to be used in optimization must be returned as a list in "objectives" key.
            The rest will be logged to the results but not passed to algorithm.
        folder_for_results: Directory for optimization results (default: "./results")
        num_objectives: Number of optimization objectives (default: 2)
        minimize_mask: Bool or list[bool] indicating which objectives to minimize.
            True minimizes all, False maximizes all. List must match obj_function outputs.
            Example: [True, False] minimizes first, maximizes second.
        num_generations: GA iteration count (default: 10)
        num_populations: Candidates per generation (default: 10)
        random_seed: Random seed for reproducibility (default: 42)
        verbose: Enable detailed output (default: False)

    Methods:
        run(): Execute optimization and save results
        save_optimized_system(candidate_index): Save a specific candidate configuration
    """

    def __init__(
        self,
        system: System,
        obj_function: Callable = default_obj_function,
        folder_for_results: Union[str, Path] = "./results",
        num_objectives: int = 2,
        minimize_mask: Union[list[bool], bool] = True,
        num_generations: int = 10,
        num_populations: int = 10,
        random_seed: Union[int, float] = 42,
        verbose: bool = False
    ):

        if num_objectives < 1:
            raise ValueError("num_objectives must be at least 1")
        if num_generations < 1:
            raise ValueError("num_generations must be at least 1")
        if num_populations < 2:
            raise ValueError("num_populations must be at least 2")

        self.system = system
        self.obj_function = obj_function
        self.num_objectives = num_objectives
        self.num_populations = num_populations
        self.num_generations = num_generations
        self.random_seed = random_seed
        self.verbose = verbose

        self._setup_result_paths(folder_for_results)
        
        self.opt_history = []
        self.final_results = []

        self.minimize_mask = self._process_minimize_mask(minimize_mask, num_objectives)

        design_vars = system.get_design_vars()
        if not design_vars:
            raise ValueError("No design variables found in system. Cannot optimize.")

        self.algorithm = MixedVariableGA(
            pop_size=self.num_populations,
            survival=RankAndCrowdingSurvival()
        )

        self.problem = VibrationOptimizationProblem(
            design_vars=design_vars,
            system=self.system,
            obj_func=self.obj_function,
            n_obj=self.num_objectives,
            minimize_mask=self.minimize_mask
        )

    def _setup_result_paths(self, folder_for_results: Union[str, Path]) -> None:
        """Setup and create results directory and file paths."""
        self.results_folder_path = Path(folder_for_results).resolve()
        self.results_folder_path.mkdir(parents=True, exist_ok=True)

        system_name = self.system.name
        self.opt_history_path = self.results_folder_path / f"{system_name}_optimization_history.json"
        self.final_results_path = self.results_folder_path / f"{system_name}_final_results.json"
        self.optimized_system_path = self.results_folder_path / f"{system_name}_optimized.json"

    @staticmethod
    def _process_minimize_mask(
        minimize_mask: Union[list[bool], bool],
        num_objectives: int
    ) -> list[int]:
        """
        Convert minimize mask to internal format.
        Returns list of 1 (minimize) or -1 (maximize) for each objective.
        """
        if isinstance(minimize_mask, bool):
            return [1 if minimize_mask else -1] * num_objectives
        
        if len(minimize_mask) != num_objectives:
            raise ValueError(
                f"minimize_mask length ({len(minimize_mask)}) must match "
                f"num_objectives ({num_objectives})"
            )
        return [1 if m else -1 for m in minimize_mask]

    def run(self) -> None:
        """Execute the optimization and save results."""

        print(f"Starting optimization of '{self.system.name}'...")

        minimize(
            self.problem,
            self.algorithm,
            ("n_gen", self.num_generations),
            seed=self.random_seed,
            verbose=self.verbose
        )

        # convert to python types to ensure serialization
        self.opt_history = to_python(self.problem.results) 
        self.final_results = self.opt_history[-self.num_populations:]

        self._save_json(self.opt_history_path, self.opt_history)
        self._save_json(self.final_results_path, self.final_results)

        print(f"Optimization complete!")
        print(f"Full history saved to: {self.opt_history_path}")
        print(f"Final results saved to: {self.final_results_path}")

    def save_optimized_system(self, candidate_index: int) -> None:
        """
        Save a specific optimized candidate to file.
        
        Args:
            candidate_index: Index of candidate in optimization history
        """
        if not self.opt_history:
            raise RuntimeError("No optimization results available. Run optimization first.")
        
        if not 0 <= candidate_index < len(self.opt_history):
            raise IndexError(
                f"candidate_index {candidate_index} out of range "
                f"[0, {len(self.opt_history)-1}]"
            )

        selected_candidate = self.opt_history[candidate_index]
        
        self.system.update(candidate=selected_candidate["design_vars"])
        self.system["optimization_results"] = selected_candidate["results"]

        self._save_json(self.optimized_system_path, self.system)
        print(f"Optimized system saved to {self.optimized_system_path}")

    @staticmethod
    def _save_json(path: Path, data: dict) -> None:
        """Save data to JSON file with error handling."""
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
        except (IOError, TypeError) as e:
            print(f"Error saving to {path}: {e}")
            raise


class VibrationOptimizationProblem(ElementwiseProblem):
    """Problem definition for vibration optimization."""

    def __init__(
        self,
        design_vars: dict,
        system: System,
        obj_func: Callable,
        n_obj: int,
        minimize_mask: list[int]
    ):
        self.obj_func = obj_func
        self.n_obj = n_obj
        self.system = system
        self.minimize_mask = minimize_mask
        self.results = []
        self._eval_counter = 0

        super().__init__(vars=design_vars, n_obj=n_obj)

    def _evaluate(self, x: dict, out: dict) -> None:
        """
        Evaluate a single candidate solution.
        
        Args:
            x: Design variable values
            out: Output dictionary for pymoo (modified in-place)
        """

        self.system.update(candidate=x)

        results_for_GA, full_result = self._run_obj_func()

        candidate_name = f"{self.system.name}_candidate_{self._eval_counter}"
        self._eval_counter += 1

        self.results.append({
            "candidate": candidate_name,
            "design_vars": x,
            "results": full_result
        })

        out["F"] = [r * m for r, m in zip(results_for_GA, self.minimize_mask)]

    def _run_obj_func(self) -> tuple:

        try:
            full_result = self.obj_func(self.system)
        except ValidationError as e:
            full_result = {"objectives": f"Failed Validation: {str(e)}"}
            results_for_GA = [1e12] * self.n_obj # to prevent breaking optimization

        if not isinstance(full_result, dict):
                raise TypeError(f"Objective function must return a dict, got {type(full_result).__name__}")

        results_for_GA = full_result.get("objectives")  

        if not isinstance(results_for_GA, list):
            raise TypeError(
                "Objective function must return a dictionary with at least an 'objectives' key " \
                "with a list of numeric values of objectives")
        
        if len(results_for_GA) != self.n_obj:
            raise ValueError(
                f"Objective function returned {len(results_for_GA)} numeric values, "
                f"but expected {self.n_obj} objectives"
            )
        
        return results_for_GA, full_result
