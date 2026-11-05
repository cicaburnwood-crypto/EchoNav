import numpy as np
import pytest
from echonav.planning import (
    GridMap,
    build_costmap,
    make_planner,
    register_planner,
    PlanResult,
)


@pytest.mark.parametrize("seed", range(12))
def test_astar_matches_dijkstra(seed):
    rng = np.random.default_rng(seed)
    costs = 1 + rng.random((25, 25)) * 5
    costs[rng.random((25, 25)) < 0.18] = np.inf
    costs[1, 1] = costs[23, 23] = 1
    grid = GridMap(costs)
    a = make_planner("astar").plan(grid, (1, 1), (23, 23))
    d = make_planner("dijkstra").plan(grid, (1, 1), (23, 23))
    assert a.status == d.status
    if a.status == "ok":
        assert a.cost == pytest.approx(d.cost, rel=1e-12)
        assert a.cells[0] == (1, 1) and a.cells[-1] == (23, 23)
        for r, c in a.cells:
            assert np.isfinite(costs[r, c])


@pytest.mark.parametrize("name", ["astar", "dijkstra"])
def test_corner_cutting_and_invalid_endpoints(name):
    grid = GridMap(np.array([[1, np.inf], [np.inf, 1]]))
    planner = make_planner(name)
    assert planner.plan(grid, (0, 0), (1, 1)).status == "unreachable"
    assert planner.plan(grid, (0, 1), (1, 1)).status == "start_blocked"
    assert planner.plan(grid, (0, 0), (0, 1)).status == "goal_blocked"
    assert planner.plan(grid, (-1, 0), (1, 1)).status == "out_of_bounds"
    assert planner.plan(grid, (0, 0), (0, 0)).cost == 0


def test_metric_roundtrip():
    g = GridMap(np.ones((256, 256)))
    for cell in ((0, 0), (255, 255), (128, 128), (10, 221)):
        assert g.metric_to_cell(*g.cell_to_metric(cell)) == cell
    assert g.metric_to_cell(0, 0) == (128, 128)
    for x, z in ((5, 0), (0, -5), (float("nan"), 0)):
        with pytest.raises(ValueError):
            g.metric_to_cell(x, z)


def test_unknown_inflation_and_costs():
    occ = np.zeros((20, 20))
    sup = np.ones_like(occ)
    conf = np.ones_like(occ)
    sup[8, 8] = 0
    occ[12, 12] = 1
    conf[5, 5] = 0
    grid = build_costmap(occ, sup, conf, extent_m=2, robot_radius_m=0.1)
    assert np.isinf(grid.costs[8, 9]) and np.isinf(grid.costs[12, 11])
    assert np.isinf(grid.costs[0]).all()
    assert grid.costs[5, 5] > grid.costs[5, 6]


def test_invalid_input_and_registry():
    for bad in (np.full((2, 2), np.nan), np.zeros((2, 2)), np.ones((2, 3))):
        with pytest.raises(ValueError):
            GridMap(bad)
    with pytest.raises(ValueError):
        build_costmap(np.zeros((2, 2)), np.ones((2, 2)), np.full((2, 2), 2.0))

    class Dummy:
        def plan(self, grid, start, goal):
            return PlanResult("custom_no_path")

    register_planner("test_custom", Dummy)
    assert make_planner("test_custom").plan(None, None, None).status == "custom_no_path"
    with pytest.raises(ValueError):
        register_planner("astar", Dummy)
    with pytest.raises(ValueError):
        make_planner("missing")
