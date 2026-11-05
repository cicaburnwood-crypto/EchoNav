"""Planar planning contract, confidence-aware cost maps, A* and Dijkstra."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
from typing import Callable, Protocol

import numpy as np

Cell = tuple[int, int]  # row, column


@dataclass(frozen=True)
class GridMap:
    """Positive cost per metre; +inf means blocked. Latest ego, forward up."""

    costs: np.ndarray
    extent_m: float = 10.0

    def __post_init__(self):
        costs = np.array(self.costs, dtype=np.float64, copy=True)
        if costs.ndim != 2 or costs.shape[0] != costs.shape[1] or not costs.size:
            raise ValueError("Costs must form a nonempty square grid")
        if not math.isfinite(self.extent_m) or self.extent_m <= 0:
            raise ValueError("Extent must be finite and positive")
        if np.isnan(costs).any() or (costs <= 0).any():
            raise ValueError("Costs must be positive finite numbers or +inf")
        costs.setflags(write=False)
        object.__setattr__(self, "costs", costs)

    @property
    def resolution_m(self) -> float:
        return self.extent_m / self.costs.shape[0]

    def metric_to_cell(self, x: float, z: float) -> Cell:
        half = self.extent_m / 2
        if (
            not math.isfinite(x)
            or not math.isfinite(z)
            or not (-half <= x < half and -half < z <= half)
        ):
            raise ValueError(
                "Goal/start outside local BEV; select an in-bounds subgoal"
            )
        return (
            math.floor((half - z) / self.resolution_m),
            math.floor((x + half) / self.resolution_m),
        )

    def cell_to_metric(self, cell: Cell) -> tuple[float, float]:
        r, c = cell
        if not self.contains(cell):
            raise ValueError("Cell outside map")
        return (
            (c + 0.5) * self.resolution_m - self.extent_m / 2,
            self.extent_m / 2 - (r + 0.5) * self.resolution_m,
        )

    def contains(self, cell: Cell) -> bool:
        return len(cell) == 2 and all(
            isinstance(v, (int, np.integer)) and 0 <= v < self.costs.shape[0]
            for v in cell
        )


@dataclass(frozen=True)
class PlanResult:
    status: str
    cells: tuple[Cell, ...] = ()
    cost: float | None = None
    expanded: int = 0


class PlanarPlanner(Protocol):
    """Any planar algorithm may implement this interface via a grid adapter."""

    def plan(self, grid: GridMap, start: Cell, goal: Cell) -> PlanResult: ...


def build_costmap(
    occupancy: np.ndarray,
    support: np.ndarray,
    confidence: np.ndarray,
    *,
    extent_m: float = 10.0,
    occupancy_threshold: float = 0.5,
    support_threshold: float = 0.5,
    confidence_weight: float = 1.0,
    robot_radius_m: float = 0.10,
) -> GridMap:
    arrays = [np.asarray(a, dtype=np.float64) for a in (occupancy, support, confidence)]
    if any(
        a.shape != arrays[0].shape
        or a.ndim != 2
        or not np.isfinite(a).all()
        or (a < 0).any()
        or (a > 1).any()
        for a in arrays
    ):
        raise ValueError(
            "Probability maps must have matching shapes and finite values in [0,1]"
        )
    if not all(
        math.isfinite(v) and 0 <= v <= 1
        for v in (occupancy_threshold, support_threshold)
    ):
        raise ValueError("Thresholds must be in [0,1]")
    if not all(
        math.isfinite(v) and v >= 0 for v in (confidence_weight, robot_radius_m)
    ):
        raise ValueError("Radius and confidence weight must be finite and nonnegative")
    occ, sup, conf = arrays
    grid = GridMap(np.ones_like(occ), extent_m)
    blocked = (occ >= occupancy_threshold) | (sup < support_threshold)
    # Inflate occupied AND unknown cells, and the map boundary. Do not erase
    # occupied cells at the ego origin or snap an invalid goal to free space.
    radius = math.ceil(robot_radius_m / grid.resolution_m)
    if radius:
        n = len(blocked)
        if radius >= n:
            blocked = np.ones_like(blocked)
        else:
            original = blocked
            padded = np.pad(original, radius, constant_values=True)
            blocked = original.copy()
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr * dr + dc * dc <= radius * radius:
                        blocked |= padded[
                            radius + dr : radius + dr + n, radius + dc : radius + dc + n
                        ]
    safe_score = np.clip((1.0 - occ) * conf, 1e-4, 1.0)
    costs = 1.0 - confidence_weight * np.log(safe_score)
    costs[blocked] = np.inf
    return GridMap(costs, extent_m)


class GridSearch:
    def __init__(self, *, heuristic: bool = True):
        self.heuristic = heuristic

    def plan(self, grid: GridMap, start: Cell, goal: Cell) -> PlanResult:
        if not grid.contains(start) or not grid.contains(goal):
            return PlanResult("out_of_bounds")
        if not np.isfinite(grid.costs[start]):
            return PlanResult("start_blocked")
        if not np.isfinite(grid.costs[goal]):
            return PlanResult("goal_blocked")
        minimum = float(grid.costs[np.isfinite(grid.costs)].min())

        def h(cell):
            dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
            return (
                (max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc))
                * grid.resolution_m
                * minimum
                if self.heuristic
                else 0.0
            )

        serial = itertools.count()
        queue = [(h(start), next(serial), 0.0, start)]
        distance, parent = {start: 0.0}, {}
        expanded = 0
        while queue:
            _, _, current, cell = heapq.heappop(queue)
            if current > distance[cell]:
                continue
            expanded += 1
            if cell == goal:
                path = [cell]
                while cell in parent:
                    cell = parent[cell]
                    path.append(cell)
                return PlanResult("ok", tuple(reversed(path)), current, expanded)
            r, c = cell
            for dr, dc in (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ):
                nxt = (r + dr, c + dc)
                if not grid.contains(nxt) or not np.isfinite(grid.costs[nxt]):
                    continue
                if (
                    dr
                    and dc
                    and (
                        not np.isfinite(grid.costs[r + dr, c])
                        or not np.isfinite(grid.costs[r, c + dc])
                    )
                ):
                    continue
                step = (
                    math.hypot(dr, dc)
                    * grid.resolution_m
                    * (grid.costs[cell] + grid.costs[nxt])
                    / 2
                )
                candidate = current + step
                if candidate < distance.get(nxt, math.inf):
                    distance[nxt], parent[nxt] = candidate, cell
                    heapq.heappush(
                        queue, (candidate + h(nxt), next(serial), candidate, nxt)
                    )
        return PlanResult("unreachable", expanded=expanded)


class AStarPlanner(GridSearch):
    def __init__(self):
        super().__init__(heuristic=True)


class DijkstraPlanner(GridSearch):
    def __init__(self):
        super().__init__(heuristic=False)


_PLANNERS: dict[str, Callable[[], PlanarPlanner]] = {
    "astar": AStarPlanner,
    "dijkstra": DijkstraPlanner,
}


def register_planner(name: str, factory: Callable[[], PlanarPlanner]) -> None:
    if not name or name in _PLANNERS or not callable(factory):
        raise ValueError("Use a new, nonempty backend name and a callable factory")
    _PLANNERS[name] = factory


def make_planner(name: str = "astar") -> PlanarPlanner:
    if name not in _PLANNERS:
        raise ValueError(
            f"Unknown planner {name!r}; registered backends: {sorted(_PLANNERS)}"
        )
    return _PLANNERS[name]()
