import copy
import csv
import importlib
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

SIM_MODE = True

from a_star import AStarPlanner
from icp_matching import icp_matching as icp_match
    
Cell = Tuple[int, int]
Wall = Tuple[float, float, float, float]

def make_pose(x: float, y: float, theta: float) -> Dict[str, float]:
    """Build a pose dictionary using float values."""

    return {"x": float(x), "y": float(y), "theta": float(theta)}


def make_set(x: float, y: float, theta: float) -> Dict[str, float]:
    """Alias for make_pose used by the set-point controller."""

    return make_pose(x, y, theta)


def make_goal(cell: Iterable[int]) -> Dict[str, Cell]:
    """Normalise a goal cell into the expected dictionary shape."""

    cx, cy = cell
    return {"cell": (int(cx), int(cy))}


def cell_center(cell: Cell, cell_size: float) -> Tuple[float, float]:
    """Return the metric centre of a grid cell."""

    return cell[0] * cell_size + cell_size / 2, cell[1] * cell_size + cell_size / 2


def wrap(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return (angle + math.pi) % (2 * math.pi) - math.pi


def ang_diff(angle_a: float, angle_b: float) -> float:
    """Return the wrapped difference between two angles."""

    return (angle_a - angle_b + math.pi) % (2 * math.pi) - math.pi


def pose_to_cell(world: Dict, pose: Dict[str, float]) -> Cell:
    """Convert a continuous pose into row/column indices for the maze grid."""

    cell_size = world["cell_size_m"]
    return int(pose["x"] / cell_size), int(pose["y"] / cell_size)


def sample_perimeter(walls: List[Wall], step: float) -> Tuple[List[float], List[float]]:
    """Sample wall rectangles into obstacle coordinates compatible with A*."""

    obstacles_x: List[float] = []
    obstacles_y: List[float] = []

    for minx, maxx, miny, maxy in walls:
        x = minx
        while x <= maxx:
            obstacles_x.extend([x, x])
            obstacles_y.extend([miny, maxy])
            x += step

        y = miny
        while y <= maxy:
            obstacles_x.extend([minx, maxx])
            obstacles_y.extend([y, y])
            y += step

    return obstacles_x, obstacles_y

def ray_rect_hits(
    px: float,
    py: float,
    dx: float,
    dy: float,
    rect: Wall,
    eps: float,
) -> List[float]:
    """Return parametric hit distances from a ray to an axis-aligned rectangle."""

    minx, maxx, miny, maxy = rect
    hits: List[float] = []

    if abs(dx) > eps:
        for boundary in (minx, maxx):
            t = (boundary - px) / dx
            if t > 0:
                y = py + t * dy
                if miny <= y <= maxy:
                    hits.append(t)

    if abs(dy) > eps:
        for boundary in (miny, maxy):
            t = (boundary - py) / dy
            if t > 0:
                x = px + t * dx
                if minx <= x <= maxx:
                    hits.append(t)

    return hits


def ogm_idx(ogm: Dict, x: float, y: float) -> Tuple[int, int]:
    """Convert world coordinates into OGM indices."""

    return (
        int((x - ogm["minx"]) / ogm["res"]),
        int((y - ogm["miny"]) / ogm["res"]),
    )


def ogm_image(ogm: Dict) -> np.ndarray:
    """Convert log-odds into a greyscale image for plotting."""

    prob = 1 / (1 + np.exp(-ogm["grid"]))
    cfg = ogm["cfg"]
    img = np.full_like(prob, cfg["gray_unk"], dtype=float)
    img = np.where(prob <= cfg["prob_free_max"], cfg["gray_free"], img)
    return np.where(prob >= cfg["prob_occ_min"], cfg["gray_occ"], img)

class RobotInterface:
    """Interface for swapping between simulated and real robot IO."""

    def get_pose(self, state: "SimulationState") -> Dict[str, float]:
        raise NotImplementedError

    def get_scan(self, state: "SimulationState", pose: Dict[str, float]) -> Dict[str, List[float]]:
        raise NotImplementedError

    def apply_setpoint(
        self, state: "SimulationState", pose: Dict[str, float], setpoint: Dict[str, float]
    ) -> Dict[str, float]:
        raise NotImplementedError



THUMB_FIG: Optional[Figure] = None
_LAST_KEY = None


@dataclass
class SimulationState:
    """Container for simulation state, ready for frontier exploration extensions."""

    world: Dict[str, Any]
    entrance: Cell
    goal: Dict[str, Cell]
    path: List[Tuple[float, float]]
    sensor: Dict[str, Any]
    ogm: Dict[str, Any]
    viz: Dict[str, Any]
    logger: Dict[str, Any]
    pose: Dict[str, float]
    settings: Dict[str, Dict[str, Any]]
    icp_prev_pts: Optional[np.ndarray]
    icp_prev_pose: Optional[Dict[str, float]]
    step: int
    astar_pts: Tuple[List[float], List[float]]
    ctrl: Dict[str, float]
    planner: Optional[Dict[str, Any]]
    robot_iface: Optional['RobotInterface'] = None
    frontier_goal: Optional[Cell] = None
    frontier_candidates: List[Cell] = field(default_factory=list)
    frontier_distances: Dict[Cell, int] = field(default_factory=dict)

def _resample_path(
    path: List[Tuple[float, float]],
    ds: float,
    eq: float,
    eps: float,
) -> List[Tuple[float, float]]:
    """Down-sample a path so that points are spaced by ``ds`` within tolerance."""

    if len(path) < 2:
        return path[:]

    cleaned: List[Tuple[float, float]] = [path[0]]
    for px, py in path[1:]:
        if abs(px - cleaned[-1][0]) > eq or abs(py - cleaned[-1][1]) > eq:
            cleaned.append((px, py))

    if len(cleaned) < 2:
        return cleaned

    arc = [0.0]
    for i in range(1, len(cleaned)):
        dx = cleaned[i][0] - cleaned[i - 1][0]
        dy = cleaned[i][1] - cleaned[i - 1][1]
        arc.append(arc[-1] + math.hypot(dx, dy))

    total = arc[-1]
    if total < eps:
        return cleaned

    out: List[Tuple[float, float]] = []
    s = 0.0
    i = 0
    while s <= total and i < len(cleaned) - 1:
        while arc[i + 1] < s and i + 1 < len(arc) - 1:
            i += 1

        seg = arc[i + 1] - arc[i]
        if seg <= eps:
            out.append(cleaned[i])
            s += ds
            continue

        t = (s - arc[i]) / seg
        x = cleaned[i][0] * (1 - t) + cleaned[i + 1][0] * t
        y = cleaned[i][1] * (1 - t) + cleaned[i + 1][1] * t
        out.append((x, y))
        s += ds

    out.append(cleaned[-1])
    return out

def _update_cloud(
    viz: Dict,
    key: str,
    ax,
    pts: Optional[np.ndarray],
    style: Dict,
) -> None:
    """Update or create a scatter plot, if points are available."""

    if pts is None or not pts.size:
        return

    obj = viz[key]
    if obj is None:
        viz[key] = ax.scatter(pts[:, 0], pts[:, 1], **style)
    else:
        obj.set_offsets(pts)

def maze_base(wcfg: Dict[str, float], size: float, cell: float) -> Dict:
    """Create the basic maze structure shared by all generators."""

    return {
        "size_m": size,
        "cell_size_m": cell,
        "grid_size": int(round(size / cell)),
        "barriers": [],
        "walls": [],
        "w_half": wcfg["wall_half_thickness_m"],
        "border": wcfg["border_thickness_m"],
    }

def add_walls(maze: Dict) -> None:
    """Populate the thickened wall rectangles, including the border."""

    wall_half = maze["w_half"]
    walls = [
        (
            barrier["minx"],
            barrier["maxx"],
            barrier["y"] - wall_half,
            barrier["y"] + wall_half,
        )
        if barrier["orientation"] == "H"
        else (
            barrier["x"] - wall_half,
            barrier["x"] + wall_half,
            barrier["miny"],
            barrier["maxy"],
        )
        for barrier in maze["barriers"]
    ]

    size = maze["size_m"]
    border = maze["border"]
    walls += [
        (0, border, 0, size),
        (size - border, size, 0, size),
        (0, size, 0, border),
        (0, size, size - border, size),
    ]
    maze["walls"] = walls


def build_snake(wcfg: Dict[str, float], cfg: Dict[str, float]) -> Dict:
    """Create the deterministic snake maze."""

    maze = maze_base(wcfg, cfg["size_m"], cfg["cell_size_m"])
    gap = max(1, int(cfg["gap_cells"])) * cfg["cell_size_m"]

    for i in range(1, int(cfg["num_walls"]) + 1):
        y = i * cfg["cell_size_m"]
        if y >= cfg["size_m"]:
            break

        if i % 2:
            minx, maxx = 0.0, max(0.0, cfg["size_m"] - gap)
        else:
            minx, maxx = min(cfg["size_m"], gap), cfg["size_m"]

        maze["barriers"].append({"orientation": "H", "y": float(y), "minx": float(minx), "maxx": float(maxx)})

    add_walls(maze)
    return maze


def build_random(wcfg: Dict[str, float], cfg: Dict[str, float]) -> Dict:
    """Create a randomised maze using the supplied configuration."""

    maze = maze_base(wcfg, cfg["size_m"], cfg["cell_size_m"])
    if cfg.get("random_seed") is not None:
        random.seed(cfg["random_seed"])

    grid_size = maze["grid_size"]
    step = cfg["cell_size_m"]
    seg_min = max(1, int(cfg["segment_len_cells_min"]))
    seg_max = max(seg_min, int(cfg["segment_len_cells_max"]))

    for _ in range(int(cfg["random_wall_count"])):
        span = random.randint(seg_min, seg_max)
        if random.random() < cfg["orientation_bias"]:
            y = random.randint(1, grid_size - 1) * step
            minx = random.randint(0, grid_size - 1 - span) * step
            maze["barriers"].append({"orientation": "H", "y": float(y), "minx": float(minx), "maxx": float(minx + span * step)})
        else:
            x = random.randint(1, grid_size - 1) * step
            miny = random.randint(0, grid_size - 1 - span) * step
            maze["barriers"].append({"orientation": "V", "x": float(x), "miny": float(miny), "maxy": float(miny + span * step)})

    add_walls(maze)
    return maze

def _open_vertical(maze: Dict, r0: int, r1: int, col: int) -> bool:
    """True if two vertically adjacent cells are connected."""

    if abs(r1 - r0) != 1:
        return True

    y = (min(r0, r1) + 1) * maze["cell_size_m"]
    c0 = col * maze["cell_size_m"]
    c1 = (col + 1) * maze["cell_size_m"]

    for barrier in maze["barriers"]:
        blocked = barrier["orientation"] == "H" and abs(barrier["y"] - y) < 1e-9 and barrier["minx"] < c1 and barrier["maxx"] > c0
        if blocked:
            return False
    return True


def _open_horizontal(maze: Dict, c0: int, c1: int, row: int) -> bool:
    """True if two horizontally adjacent cells are connected."""

    if abs(c1 - c0) != 1:
        return True

    x = (min(c0, c1) + 1) * maze["cell_size_m"]
    r0 = row * maze["cell_size_m"]
    r1 = (row + 1) * maze["cell_size_m"]

    for barrier in maze["barriers"]:
        blocked = barrier["orientation"] == "V" and abs(barrier["x"] - x) < 1e-9 and barrier["miny"] < r1 and barrier["maxy"] > r0
        if blocked:
            return False
    return True


def neighbors(maze: Dict, row: int, col: int):
    """Yield all neighbouring maze cells that can be reached."""

    grid_size = maze["grid_size"]
    if col + 1 < grid_size and _open_horizontal(maze, col, col + 1, row):
        yield row, col + 1
    if col - 1 >= 0 and _open_horizontal(maze, col - 1, col, row):
        yield row, col - 1
    if row + 1 < grid_size and _open_vertical(maze, row, row + 1, col):
        yield row + 1, col
    if row - 1 >= 0 and _open_vertical(maze, row - 1, row, col):
        yield row - 1, col


def maze_path_exists(maze: Dict, start: Cell, goal: Cell) -> bool:
    """Simple DFS to confirm a path exists between two cells."""

    start_row, start_col = start[1], start[0]
    goal_row, goal_col = goal[1], goal[0]
    stack = [(start_row, start_col)]
    seen = {(start_row, start_col)}

    while stack:
        row, col = stack.pop()
        if (row, col) == (goal_row, goal_col):
            return True

        for next_row, next_col in neighbors(maze, row, col):
            if (next_row, next_col) not in seen:
                seen.add((next_row, next_col))
                stack.append((next_row, next_col))

    return False


def create_lidar(cfg: Dict[str, float]) -> Dict:
    """Pre-compute trigonometric lookup tables for the simulated lidar."""

    angles = np.linspace(0, 2 * math.pi, int(cfg["num_rays"]), endpoint=False)
    return {
        "cfg": cfg,
        "angles": angles,
        "cos": np.cos(angles),
        "sin": np.sin(angles),
    }


def scan(sensor: Dict, pose: Dict[str, float], world: Dict) -> Dict[str, List[float]]:
    """Simulate a lidar scan from the provided robot pose."""

    cfg = sensor["cfg"]
    cos_theta = math.cos(pose["theta"])
    sin_theta = math.sin(pose["theta"])
    ranges: List[float] = []

    for i, angle in enumerate(sensor["angles"]):
        dx = cos_theta * sensor["cos"][i] - sin_theta * sensor["sin"][i]
        dy = sin_theta * sensor["cos"][i] + cos_theta * sensor["sin"][i]
        best = cfg["max_range_m"]

        for rect in world["walls"]:
            hits = ray_rect_hits(pose["x"], pose["y"], dx, dy, rect, cfg["raycast_eps"])
            if hits:
                best = min(best, min(hits))

        clamped = best if best < cfg["max_range_m"] else cfg["max_range_m"]
        ranges.append(clamped)

    return {"angles": list(sensor["angles"]), "ranges": ranges}


def create_ogm(
    cfg: Dict[str, float],
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
) -> Dict:
    """Allocate an occupancy grid map covering the requested region."""

    res = cfg["xyreso_m"]
    height = int((maxy - miny) / res + cfg["size_eps"])
    width = int((maxx - minx) / res + cfg["size_eps"])
    grid = np.zeros((height, width), dtype=float)
    return {"cfg": cfg, "grid": grid, "minx": minx, "miny": miny, "res": res}


def update_ogm(
    ogm: Dict,
    scan_data: Dict[str, List[float]],
    pose: Dict[str, float],
) -> None:
    """Integrate a single lidar scan into the occupancy grid map."""

    grid = ogm["grid"]
    height, width = grid.shape
    cfg = ogm["cfg"]
    max_range = max(scan_data["ranges"]) if scan_data["ranges"] else 0.0

    for rel_angle, dist in zip(scan_data["angles"], scan_data["ranges"]):
        rng = min(dist, max_range)
        endx = pose["x"] + rng * math.cos(pose["theta"] + rel_angle)
        endy = pose["y"] + rng * math.sin(pose["theta"] + rel_angle)

        ix0, iy0 = ogm_idx(ogm, pose["x"], pose["y"])
        ix1, iy1 = ogm_idx(ogm, endx, endy)
        if not (0 <= ix0 < width and 0 <= iy0 < height):
            continue

        dx = abs(ix1 - ix0)
        dy = -abs(iy1 - iy0)
        sx = 1 if ix0 < ix1 else -1
        sy = 1 if iy0 < iy1 else -1
        err = dx + dy
        x, y = ix0, iy0
        cells: List[Tuple[int, int]] = []

        while True:
            cells.append((x, y))
            if x == ix1 and y == iy1:
                break

            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

        if not cells:
            continue

        hit = dist < max_range - cfg["hit_margin_m"]
        traversed = cells[:-1] if hit else cells
        for cx, cy in traversed:
            if 0 <= cx < width and 0 <= cy < height:
                grid[cy, cx] = np.clip(grid[cy, cx] + cfg["l_free"], cfg["l_min"], cfg["l_max"])

        if hit:
            end_cell_x, end_cell_y = cells[-1]
            if 0 <= end_cell_x < width and 0 <= end_cell_y < height:
                grid[end_cell_y, end_cell_x] = np.clip(grid[end_cell_y, end_cell_x] + cfg["l_occ"], cfg["l_min"], cfg["l_max"])


def create_planner(world: Dict, step: float, radius: float) -> Dict:
    """Create an A* planner wrapper with obstacle caches."""

    # Assume A* is available per request

    return {
        "world": world,
        "step": step,
        "radius": radius,
        "obstacles": ([], []),
        "cspace": ([], []),
    }


def run_astar(
    plan: Dict,
    start: Cell,
    goal: Cell,
    cfg: Dict[str, float],
) -> List[Tuple[float, float]]:
    """Run A* and resample the resulting path."""

    override = plan.pop("obstacles_override", None)
    if override is not None:
        ox, oy = override
    else:
        ox, oy = sample_perimeter(plan["world"]["walls"], cfg["sample_step_m"])
    plan["obstacles"] = (ox, oy)

    sx, sy = cell_center(start, plan["world"]["cell_size_m"])
    gx, gy = cell_center(goal, plan["world"]["cell_size_m"])

    try:
        astar = AStarPlanner(ox, oy, cfg["sample_step_m"], plan["radius"])
        rx, ry = astar.planning(sx, sy, gx, gy)
    except Exception as exc:  # pragma: no cover - external dependency
        raise RuntimeError(f"A* planning failed: {exc}") from exc

    if not rx or not ry:
        raise RuntimeError("A* returned empty path")

    try:
        xs: List[float] = []
        ys: List[float] = []
        for ix in range(astar.x_width):
            for iy in range(astar.y_width):
                if astar.obstacle_map[ix][iy]:
                    xs.append(astar.calc_grid_position(ix, astar.min_x))
                    ys.append(astar.calc_grid_position(iy, astar.min_y))
        plan["cspace"] = (xs, ys)
    except Exception:  # pragma: no cover - defensive
        plan["cspace"] = ([], [])

    path = list(zip(list(rx)[::-1], list(ry)[::-1]))
    return _resample_path(path, cfg["resample_ds_m"], cfg["equal_eps"], cfg["seg_eps"])


def fallback_path(world: Dict, start: Cell, goal: Cell) -> List[Tuple[float, float]]:
    """Generate a simple start-to-goal path used as a safe default."""

    start_pt = cell_center(start, world["cell_size_m"])
    goal_pt = cell_center(goal, world["cell_size_m"])
    if math.isclose(start_pt[0], goal_pt[0], abs_tol=1e-9) and math.isclose(start_pt[1], goal_pt[1], abs_tol=1e-9):
        return [start_pt]
    return [start_pt, goal_pt]


def initialise_navigation_path(
    planner: Dict,
    entrance: Cell,
    goal_cell: Cell,
    settings: Dict[str, Dict[str, Any]],
    mode: str,
) -> List[Tuple[float, float]]:
    """Construct the initial path depending on the selected navigation mode."""

    try:
        if mode == "UNKNOWN":
            return plan_unknown_world_initial(planner["world"], entrance, goal_cell)
        return run_astar(planner, entrance, goal_cell, settings["planning"])
    except RuntimeError as exc:
        log.warning("Initial planning failed: %s; using fallback path.", exc)
        return fallback_path(planner["world"], entrance, goal_cell)


def plan_unknown_world_initial(world: Dict, entrance: Cell, goal_cell: Cell) -> List[Tuple[float, float]]:
    """Fallback initial path for the unknown-world mode; override for custom logic."""

    return fallback_path(world, entrance, goal_cell)


def plan_unknown_world(
    state: "SimulationState",
    start_cell: Cell,
    goal_cell: Cell,
) -> List[Tuple[float, float]]:
    """Plan a path in an unknown world using the current occupancy map."""

    planner = create_planner(
        state.world,
        state.settings["planning"]["sample_step_m"],
        state.settings["robot"]["robot_radius_m"],
    )

    ogm_obstacles = ogm_obstacles_from_map(state.ogm)
    step = state.settings["planning"]["sample_step_m"]
    border = state.world.get("border", state.settings["world"].get("border_thickness_m", 0.0))
    if border > 0.0:
        size = state.world["size_m"]
        border_rects = [
            (0.0, border, 0.0, size),
            (size - border, size, 0.0, size),
            (0.0, size, 0.0, border),
            (0.0, size, size - border, size),
        ]
        border_samples = sample_perimeter(border_rects, step)
    else:
        border_samples = ([], [])

    if ogm_obstacles[0] and ogm_obstacles[1]:
        ox = list(border_samples[0]) + ogm_obstacles[0]
        oy = list(border_samples[1]) + ogm_obstacles[1]
        planner["obstacles_override"] = (ox, oy)
    elif border_samples[0] and border_samples[1]:
        planner["obstacles_override"] = border_samples

    try:
        path = run_astar(planner, start_cell, goal_cell, state.settings["planning"])
        state.astar_pts = planner["cspace"] if planner["cspace"] else planner["obstacles"]
        state.planner = planner
        return path
    except RuntimeError as exc:
        log.debug("Unknown-world planning fell back (%s)", exc)
        state.astar_pts = ogm_obstacles
        state.planner = planner
        return []


def ogm_obstacles_from_map(ogm: Dict) -> Tuple[List[float], List[float]]:
    """Extract obstacle samples from the occupancy grid for visualisation and planning."""

    if not ogm:
        return ([], [])

    grid = ogm["grid"]
    if grid.size == 0:
        return ([], [])

    cfg = ogm["cfg"]
    prob = 1 / (1 + np.exp(-grid))
    mask = prob >= cfg.get("prob_occ_min", 0.65)
    if not np.any(mask):
        return ([], [])

    res = ogm["res"]
    minx = ogm["minx"]
    miny = ogm["miny"]
    ys, xs = np.nonzero(mask)
    obs_x = (xs * res + minx).tolist()
    obs_y = (ys * res + miny).tolist()
    return obs_x, obs_y


def determine_goal_path(state: SimulationState) -> None:
    state.path = run_astar(state.planner, state.entrance, state.goal["cell"], state.settings["planning"])

def compute_setpoint(ctrl: Dict[str, float], path: List[Tuple[float, float]], pose: Dict[str, float],) -> Dict[str, float]:
    """
    Computes the setpoint for robot navigation based on the current pose and path.
    This function determines the next goal point along a path considering a lookahead distance.
    If no path is provided, it uses the current pose as the setpoint.
    Args:
        ctrl (Dict[str, float]): Control parameters dictionary containing 'lookahead_m' for path following
        path (List[Tuple[float, float]]): List of (x, y) waypoints defining the path to follow
        pose (Dict[str, float]): Current robot pose containing 'x', 'y', and 'theta' keys
    Returns:
        Dict[str, float]: Setpoint dictionary containing target x, y coordinates and orientation theta
    Notes:
        - Uses euclidean distance (hypot) to find closest point on path
        - Advances along path until reaching lookahead distance or path end
        - If path is empty, returns current pose as setpoint
    """
    
    if path:
        distances = [math.hypot(px - pose["x"], py - pose["y"]) for px, py in path]
        idx = int(np.argmin(distances))
        travel = 0.0
        while travel < ctrl["lookahead_m"] and idx < len(path) - 1:
            travel += math.hypot(path[idx + 1][0] - path[idx][0], path[idx + 1][1] - path[idx][1])
            idx += 1
        goal_x, goal_y = path[idx]
        print(f"Next path point at index {idx}, distance {distances[idx]:.2f} m")
    else:
        goal_x, goal_y = pose["x"], pose["y"]
    
    theta = pose["theta"]
    #print(f"Setpoint at ({goal_x:.2f}, {goal_y:.2f}), {math.degrees(theta):.1f}°")
    return make_set(goal_x, goal_y, theta)

def create_viz(size: float, cell: float, cfg: Dict[str, float], radius: float) -> Dict:
    """Create matplotlib figures used for live visualisation."""

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=cfg["main_figsize_in"], constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig)
    axes = {
        "map": fig.add_subplot(grid[0, 0]),
        "ogm": fig.add_subplot(grid[0, 1]),
        "path": fig.add_subplot(grid[1, 0]),
        "icp": fig.add_subplot(grid[1, 1]),
    }

    for axis in axes.values():
        axis.set_xlim(0, size)
        axis.set_ylim(0, size)
        axis.set_box_aspect(1)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True)

    xticks = np.arange(0, size + 1e-9, cell)
    for axis in (axes["map"], axes["ogm"]):
        axis.set_xticks(xticks)
        axis.set_yticks(xticks)

    axes["path"].set_title("A* C-space Obstacles")
    axes["icp"].set_title("ICP Clouds")

    return {
        "fig": fig,
        "axes": axes,
        "size": size,
        "cell": cell,
        "cfg": cfg,
        "radius": radius,
        "icp_prev": None,
        "icp_curr": None,
        "icp_tf": None,
    }

def render(
    viz: Dict,
    world: Dict,
    ogm: Dict,
    pose: Dict[str, float],
    scan_data: Dict[str, List[float]],
    goal: Dict[str, Cell],
    step: int,
    path: List[Tuple[float, float]],
    entrance: Cell,
    prev_pts: Optional[np.ndarray],
    curr_pts: Optional[np.ndarray],
    tf_pts: Optional[np.ndarray],
    astar_pts: Optional[Tuple[List[float], List[float]]],
    frontier_goal: Optional[Cell] = None,
    frontier_candidates: Optional[List[Cell]] = None,
) -> None:
    """Update all live visualisations for the current simulation step."""

    cfg = viz["cfg"]
    axes = viz["axes"]
    size = viz["size"]
    cell = viz["cell"]

    map_ax = axes["map"]
    ogm_ax = axes["ogm"]
    path_ax = axes["path"]
    icp_ax = axes["icp"]

    frontier_points: List[Tuple[float, float]] = []
    if frontier_candidates:
        for cell_coord in dict.fromkeys(frontier_candidates):
            frontier_points.append(cell_center(cell_coord, cell))

    frontier_goal_pt: Optional[Tuple[float, float]] = None
    if frontier_goal:
        frontier_goal_pt = cell_center(frontier_goal, cell)

    for axis in (map_ax, ogm_ax):
        axis.clear()
        axis.set_xlim(0, size)
        axis.set_ylim(0, size)
        axis.set_box_aspect(1)
        axis.set_aspect("equal", adjustable="box")
        ticks = np.arange(0, size + 1e-9, cell)
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
        axis.grid(True)

    for rect in world["walls"]:
        wall_patch = Rectangle((rect[0], rect[2]), rect[1] - rect[0], rect[3] - rect[2], facecolor="black")
        map_ax.add_patch(wall_patch)

    for cell_coord, color, label in ((entrance, "green", "Entrance"), (goal["cell"], "red", "Goal")):
        highlight_patch = Rectangle((cell_coord[0] * cell, cell_coord[1] * cell), cell, cell, facecolor=color, alpha=0.3, label=label)
        map_ax.add_patch(highlight_patch)

    map_ax.plot(pose["x"], pose["y"], "bo", markersize=8)
    robot_arrow_dx = cfg["robot_arrow_len_m"] * math.cos(pose["theta"])
    robot_arrow_dy = cfg["robot_arrow_len_m"] * math.sin(pose["theta"])
    map_ax.arrow(pose["x"], pose["y"], robot_arrow_dx, robot_arrow_dy, head_width=cfg["robot_arrow_head_m"][0], head_length=cfg["robot_arrow_head_m"][1], fc="red", ec="red")

    for angle, distance in zip(scan_data["angles"], scan_data["ranges"]):
        end_x = pose["x"] + distance * math.cos(pose["theta"] + angle)
        end_y = pose["y"] + distance * math.sin(pose["theta"] + angle)
        ray_x = [pose["x"], end_x]
        ray_y = [pose["y"], end_y]
        map_ax.plot(ray_x, ray_y, "r-", alpha=cfg["lidar_alpha"], linewidth=cfg["lidar_lw"])

    if path:
        map_ax.plot(*zip(*path), "g--", linewidth=2, alpha=0.8)

    if frontier_points:
        fx, fy = zip(*frontier_points)
        map_ax.scatter(fx, fy, c="#FF7F0E", marker="s", s=25, label="frontiers")

    if frontier_goal_pt:
        map_ax.scatter([frontier_goal_pt[0]], [frontier_goal_pt[1]], c="#FFD700", marker="*", s=80, label="frontier goal")

    handles, labels = map_ax.get_legend_handles_labels()
    if labels:
        legend_entries = dict(zip(labels, handles))
        map_ax.legend(legend_entries.values(), legend_entries.keys())

    map_ax.set_title(f"Ground Truth Maze — Step {step}")
    map_ax.set_xlabel("X (m)")
    map_ax.set_ylabel("Y (m)")

    ogm_img = ogm_image(ogm)
    ogm_ax.imshow(ogm_img, extent=(0, size, 0, size), origin="lower", vmin=0.0, vmax=1.0, cmap="gray")
    ogm_ax.plot(pose["x"], pose["y"], "bo", markersize=6, label="robot")
    ogm_arrow_dx = cfg["ogm_arrow_len_m"] * math.cos(pose["theta"])
    ogm_arrow_dy = cfg["ogm_arrow_len_m"] * math.sin(pose["theta"])
    ogm_ax.arrow(pose["x"], pose["y"], ogm_arrow_dx, ogm_arrow_dy, head_width=cfg["ogm_arrow_head_m"][0], head_length=cfg["ogm_arrow_head_m"][1], fc="blue", ec="blue")

    if path:
        ogm_ax.plot(*zip(*path), "g--", linewidth=2, alpha=0.8, label="planned path")

    if frontier_points:
        fx, fy = zip(*frontier_points)
        ogm_ax.scatter(fx, fy, c="#FF7F0E", marker="s", s=25, label="frontiers")

    if frontier_goal_pt:
        ogm_ax.scatter([frontier_goal_pt[0]], [frontier_goal_pt[1]], c="#FFD700", marker="*", s=80, label="frontier goal")
    ogm_ax.set_title("Occupancy Grid Map")
    ogm_ax.set_xlabel("X (m)")
    ogm_ax.set_ylabel("Y (m)")

    handles, labels = ogm_ax.get_legend_handles_labels()
    if labels:
        legend_entries = dict(zip(labels, handles))
        ogm_ax.legend(legend_entries.values(), legend_entries.keys())

    path_ax.clear()
    path_ax.set_xlim(0, size)
    path_ax.set_ylim(0, size)
    path_ax.set_box_aspect(1)
    path_ax.set_aspect("equal", adjustable="box")
    path_ax.grid(True)
    path_ax.set_title("A* C-space Obstacles")
    path_ax.set_xlabel("X (m)")
    path_ax.set_ylabel("Y (m)")

    if astar_pts and astar_pts[0]:
        path_ax.plot(astar_pts[0], astar_pts[1], "x", ms=4, alpha=0.8, color="#12B2B2", label="occupied cells")

    if path and len(path) > 3:
        path_ax.plot(*zip(*path), ".-", alpha=0.9, label="Goal path")

    if frontier_points:
        fx, fy = zip(*frontier_points)
        path_ax.scatter(fx, fy, c="#FF7F0E", marker="s", s=20, label="frontiers")

    if frontier_goal_pt:
        path_ax.scatter([frontier_goal_pt[0]], [frontier_goal_pt[1]], c="#FFD700", marker="*", s=60, label="frontier goal")

    path_ax.plot(pose["x"], pose["y"], "ro", ms=4, label="robot")
    radius_ring = Circle((pose["x"], pose["y"]), radius=viz["radius"], fill=False, linestyle="--", linewidth=1.2, color="#FF7F0E", label="robot_radius")
    path_ax.add_patch(radius_ring)

    handles, labels = path_ax.get_legend_handles_labels()
    if labels:
        legend_entries = dict(zip(labels, handles))
        path_ax.legend(legend_entries.values(), legend_entries.keys())

    icp_ax.set_xlim(0, size)
    icp_ax.set_ylim(0, size)
    icp_ax.set_box_aspect(1)
    icp_ax.set_aspect("equal", adjustable="box")
    icp_ax.grid(True)
    icp_ax.set_title("ICP Clouds (prev=gray, curr=blue, xform=green)")
    icp_ax.set_xlabel("X (m)")
    icp_ax.set_ylabel("Y (m)")

    _update_cloud(viz, "icp_prev", icp_ax, prev_pts, {"s": 3, "c": "#888888", "label": "prev"})
    _update_cloud(viz, "icp_curr", icp_ax, curr_pts, {"s": 3, "label": "curr"})
    _update_cloud(viz, "icp_tf", icp_ax, tf_pts, {"s": 3, "c": "#2ca02c", "label": "xformed"})

    handles, labels = icp_ax.get_legend_handles_labels()
    if labels:
        legend_entries = dict(zip(labels, handles))
        icp_ax.legend(legend_entries.values(), legend_entries.keys())

    plt.draw()
    plt.pause(cfg["pause_s"])
    viz["fig"].canvas.flush_events()

def create_logger(num_rays: int, cfg: Dict[str, str]) -> Dict:
    """Prepare CSV log files for pose and lidar measurements."""

    pose_path = Path(cfg["pose_csv"])
    lidar_path = Path(cfg["lidar_csv"])
    diag_path = Path(cfg.get("diag_csv", "diagnostic.csv"))

    for path_obj in (pose_path, lidar_path, diag_path):
        if path_obj.parent:
            path_obj.parent.mkdir(parents=True, exist_ok=True)
        try:
            if path_obj.exists():
                path_obj.unlink()
        except Exception:
            path_obj.write_text("")

    with pose_path.open("w", newline="") as handle:
        csv.writer(handle).writerow(["step", "x_m", "y_m", "theta_deg", "mode"])

    header = ["step"] + [
        item
        for idx in range(num_rays)
        for item in (f"angle_deg_{idx}", f"dist_m_{idx}")
    ]
    with lidar_path.open("w", newline="") as handle:
        csv.writer(handle).writerow(header)

    with diag_path.open("w", newline="") as handle:
        csv.writer(handle).writerow(
            [
                "step",
                "nav_mode",
                "pose_x_m",
                "pose_y_m",
                "pose_theta_deg",
                "frontier_goal_x",
                "frontier_goal_y",
                "frontier_goal_dist",
                "frontier_goal_cell",
                "frontier_candidate_count",
                "frontier_candidates",
                "path_length",
                "path_first_x",
                "path_first_y",
                "icp_pose_x",
                "icp_pose_y",
                "icp_pose_theta_deg",
                "icp_rmse",
                "icp_num_points",
            ]
        )

    return {"pose": pose_path, "lidar": lidar_path, "diag": diag_path, "rays": num_rays}

def propose_random(
    world_cfg: Dict[str, float],
    cfg: Dict[str, float],
    entrance: Cell,
    goal: Cell,
    viz_cfg: Dict[str, float],
) -> Tuple[List[int], int]:
    """Enumerate candidate random mazes and display thumbnails."""

    seeds: List[int] = []
    wall_sets: List[List[Wall]] = []
    start = cfg["seed_scan_start"]
    attempts = 0

    while len(seeds) < cfg["candidates_to_list"] and attempts < cfg["max_attempts_per_page"]:
        attempts += 1
        trial = cfg.copy()
        trial["random_seed"] = start
        try:
            maze = build_random(world_cfg, trial)
            if maze_path_exists(maze, entrance, goal):
                seeds.append(start)
                wall_sets.append(maze["walls"])
        except Exception as exc:
            log.debug("Seed %d generation failed: %s", start, exc)

        start += cfg["seed_scan_stride"]

    global THUMB_FIG
    if seeds:
        cols = len(seeds)
        width = viz_cfg["thumb_size_in"][0] * cols
        height = viz_cfg["thumb_size_in"][1]
        if THUMB_FIG is None or not plt.fignum_exists(THUMB_FIG.number):
            THUMB_FIG = plt.figure(num="Random maze candidates", figsize=(width, height))
        else:
            THUMB_FIG.clf()
            THUMB_FIG.set_size_inches(width, height, forward=True)

        axes = THUMB_FIG.subplots(1, cols, squeeze=False)[0]
        for axis, seed, wall_list in zip(axes, seeds, wall_sets):
            axis.set_xlim(0, cfg["size_m"])
            axis.set_ylim(0, cfg["size_m"])
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal")

            for rect in wall_list:
                wall_patch = Rectangle((rect[0], rect[2]), rect[1] - rect[0], rect[3] - rect[2], facecolor="black")
                axis.add_patch(wall_patch)

            step = cfg["cell_size_m"]
            for cell_coord, color in ((entrance, "green"), (goal, "red")):
                highlight_patch = Rectangle((cell_coord[0] * step, cell_coord[1] * step), step, step, facecolor=color, alpha=0.3)
                axis.add_patch(highlight_patch)

            axis.set_title(f"seed {seed}")

        THUMB_FIG.suptitle("Random maze candidates (path_exists=True)")
        THUMB_FIG.tight_layout()
        plt.draw()
        plt.pause(0.01)
        print("Candidate seeds:", ", ".join(map(str, seeds)))
    else:
        print("No valid random mazes found on this page.")

    return seeds, start

def simulate_step(
    current: Dict[str, float],
    setpoint: Dict[str, float],
    dt: float,
    vmax: float,
    k_ang: float,
    guard: float,
) -> Dict[str, float]:
    """Advance the simulated robot pose toward the set-point."""

    heading = math.atan2(setpoint["y"] - current["y"], setpoint["x"] - current["x"])
    omega = k_ang * wrap(heading - current["theta"])
    distance = math.hypot(setpoint["x"] - current["x"], setpoint["y"] - current["y"])
    velocity = min(vmax, max(0.0, distance / max(guard, dt)))

    new_x = current["x"] + velocity * math.cos(current["theta"]) * dt
    new_y = current["y"] + velocity * math.sin(current["theta"]) * dt
    new_theta = (current["theta"] + omega * dt) % (2 * math.pi)
    return make_pose(new_x, new_y, new_theta)


class SimulatedRobotInterface(RobotInterface):
    """Default adapter that mirrors the built-in simulator."""

    def get_pose(self, state: "SimulationState") -> Dict[str, float]:
        return state.pose

    def get_scan(self, state: "SimulationState", pose: Dict[str, float]) -> Dict[str, List[float]]:
        return scan(state.sensor, pose, state.world)

    def apply_setpoint(self, state: "SimulationState", pose: Dict[str, float], setpoint: Dict[str, float]) -> Dict[str, float]:
        if SIM_MODE:
            robot_cfg = state.settings["robot"]
            return simulate_step(
                pose,
                setpoint,
                robot_cfg["dt_s"],
                robot_cfg["v_max_mps"],
                robot_cfg["k_ang"],
                robot_cfg["dt_guard_s"],
            )
        return pose


def load_robot_interface(settings: Dict[str, Dict[str, Any]]) -> RobotInterface:
    module_name = settings.get("robot", {}).get("interface_module")
    module_name = module_name or os.environ.get("MAZE_ROBOT_INTERFACE")
    if module_name:
        try:
            module = importlib.import_module(module_name)
            factory = getattr(module, "create_robot_interface", None)
            if callable(factory):
                iface = factory(settings)
                if iface is not None:
                    missing = [name for name in ("get_pose", "get_scan", "apply_setpoint") if not hasattr(iface, name)]
                    if not missing:
                        return iface
                    log.warning(
                        "Robot interface %s missing required methods %s; falling back to simulation.",
                        module_name,
                        missing,
                    )
            else:
                log.warning(
                    "Robot interface module %s does not define create_robot_interface; falling back to simulation.",
                    module_name,
                )
        except Exception as exc:
            log.warning("Robot interface import failed (%s); using simulated interface.", exc)
    return SimulatedRobotInterface()


def icp_points(
    pose: Dict[str, float],
    scan_data: Dict[str, List[float]],
    cfg: Dict[str, float],
) -> np.ndarray:
    """Convert lidar hits to Cartesian points for ICP matching."""

    points = [
        (
            pose["x"] + dist * math.cos(pose["theta"] + angle),
            pose["y"] + dist * math.sin(pose["theta"] + angle),
        )
        for angle, dist in zip(scan_data["angles"], scan_data["ranges"])
        if 0.0 < dist < cfg["max_range_m"] - 1e-6 and math.isfinite(dist)
    ]
    if points:
        return np.array(points, dtype=float)
    return np.zeros((0, 2), dtype=float)


def icp_match_step(
    prev_pts: Optional[np.ndarray],
    curr_pts: Optional[np.ndarray],
    prev_pose: Optional[Dict[str, float]],
) -> Tuple[Optional[Dict[str, float]], Optional[float], int, Optional[np.ndarray]]:
    """Run a single ICP alignment step if data is available."""

    if (
        prev_pts is None
        or prev_pose is None
        or curr_pts is None
        or prev_pts.size == 0
        or curr_pts.size == 0
    ):
        return None, None, 0, None

    try:
        result = icp_match(prev_pts.T, curr_pts.T)
        if (
            isinstance(result, tuple)
            and len(result) == 3
            and all(isinstance(val, (int, float)) for val in result)
        ):
            yaw, tx, ty = float(result[0]), float(result[1]), float(result[2])
        elif isinstance(result, tuple) and len(result) >= 2:
            R, t = result[0], result[1]
            yaw = math.atan2(R[1, 0], R[0, 0])
            tx, ty = float(t[0]), float(t[1])
        else:
            return None, None, 0, None

        fused_x = prev_pose["x"] + tx
        fused_y = prev_pose["y"] + ty
        fused_theta = (prev_pose["theta"] + yaw) % (2 * math.pi)
        pose = make_pose(fused_x, fused_y, fused_theta)

        try:
            rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            transformed = prev_pts @ rotation.T + np.array([tx, ty])
            if transformed.size and curr_pts.size:
                d2 = ((transformed[:, None, :] - curr_pts[None, :, :]) ** 2).sum(axis=2)
                rmse = float(np.sqrt(d2.min(axis=1).mean()))
                return pose, rmse, prev_pts.shape[0], transformed
        except Exception:
            pass

        return pose, None, prev_pts.shape[0], None
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("ICP failed: %s", exc)
        return None, None, 0, None


def fuse_icp_pose(
    settings: Dict[str, Dict],
    pose: Dict[str, float],
    icp_pose: Optional[Dict[str, float]],
    rmse: Optional[float],
    n_pts: int,
) -> Optional[Dict[str, float]]:
    """Blend ICP and odometry poses if the estimate is trustworthy."""

    cfg = settings["icp_fusion"]
    if (
        not cfg["enabled"]
        or icp_pose is None
        or n_pts < cfg["min_points"]
        or rmse is None
        or rmse > cfg["max_rmse_m"]
    ):
        return None

    translation_delta = math.hypot(icp_pose["x"] - pose["x"], icp_pose["y"] - pose["y"])
    rotation_delta = abs(ang_diff(icp_pose["theta"], pose["theta"]))

    if translation_delta > cfg["max_trans_m"] or math.degrees(rotation_delta) > cfg["max_rot_deg"]:
        return None

    if (
        translation_delta < cfg["snap_trans_m"]
        and math.degrees(rotation_delta) < cfg["snap_rot_deg"]
    ):
        return icp_pose

    alpha = max(0.0, min(1.0, cfg["alpha"]))
    blended_x = (1 - alpha) * pose["x"] + alpha * icp_pose["x"]
    blended_y = (1 - alpha) * pose["y"] + alpha * icp_pose["y"]
    blended_theta = (pose["theta"] + alpha * ang_diff(icp_pose["theta"], pose["theta"])) % (2 * math.pi)
    return make_pose(blended_x, blended_y, blended_theta)

def ask_options(defaults: Dict[str, Dict]) -> Dict:
    """Prompt the user for the desired maze type."""

    app = defaults["app"].copy()
    #app["mode"] = "GOALSEEKING"

    while True:
        choice = input("Choose map: [R]ANDOM or [S]NAKE? ").strip().lower()
        if choice in {"", "r", "random"}:
            app["map_type"] = "RANDOM"
            return app
        if choice in {"s", "snake"}:
            app["map_type"] = "SNAKE"
            return app
        print("Please enter R or S.")


def choose_navigation_mode(settings: Dict[str, Dict[str, Any]]) -> str:
    """Ask the user whether the maze layout is known or unknown."""

    nav_cfg = settings.setdefault("navigation", {})
    while True:
        prompt = "Is the maze layout known? Y/N (default: Y): "
        choice = input(prompt).strip().lower()
        if choice in {"", "y", "yes"}:
            nav_cfg["mode"] = "KNOWN"
            return "KNOWN"
        if choice in {"n", "no"}:
            nav_cfg["mode"] = "UNKNOWN"
            return "UNKNOWN"
        print("Please enter Y or N.")


def build_world(settings: Dict[str, Dict], app: Dict) -> Tuple[Dict, Cell, Cell]:
    """Create the selected maze and return the entrance/goal cells."""

    entrance = tuple(app["entrance_cell"])
    if app["map_type"] == "SNAKE":
        world = build_snake(settings["world"], settings["snake_maze"])
        goal = (
            tuple(app["snake_goal_cell"])
            if app["snake_goal_cell"]
            else (world["grid_size"] - 1, world["grid_size"] - 1)
        )
        if not maze_path_exists(world, entrance, goal):
            raise RuntimeError("Snake maze has no valid path with current params.")
        return world, entrance, goal

    goal = tuple(app["random_goal_cell"])
    cfg = settings["random_maze"].copy()
    cfg["random_seed"] = None

    while cfg["random_seed"] is None:
        seeds, next_seed = propose_random(settings["world"], cfg, entrance, goal, settings["viz"])
        selection = input(f"Enter a seed from {seeds} to proceed (ENTER for next page): ").strip()
        if selection == "":
            cfg["seed_scan_start"] = next_seed
            continue
        try:
            value = int(selection)
            if value in seeds:
                cfg["random_seed"] = value
            else:
                print("[WARN] Seed not in current list.")
        except Exception as exc:
            print(f"[WARN] {exc}")

    global THUMB_FIG
    try:
        if THUMB_FIG is not None:
            plt.close(THUMB_FIG)
            THUMB_FIG = None
    except Exception:
        pass

    world = build_random(settings["world"], cfg)
    if not maze_path_exists(world, entrance, goal):
        raise RuntimeError("Selected random maze has no valid path; choose another seed.")

    return world, entrance, goal
