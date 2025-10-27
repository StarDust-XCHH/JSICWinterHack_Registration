"""
===============================================================================
Autonomous Mobile Robot Navigation App/Pipeline
Author: Jonathan Loo
Version: 1.0
Date: October 2025
===============================================================================
Purpose
--------
Implements a synchronous Sense→Think→Act control loop for autonomous maze navigation.
Each loop iteration reads the robot pose and LiDAR scan, refines pose via
ICP scan matching, updates the occupancy grid map (OGM), computes or updates a path
(A* or frontier-based), generates a lookahead setpoint, applies it to the simulated
robot, and visualises/logs the result.

Core Concept
-------------
Demonstrates a compact “SLAM” pipeline:
    ICP-aided localisation + OGM mapping + goal/frontier navigation
executed in real time within a single blocking loop.

Simulation vs Real Operation
----------------------------
- **SIMULATION (default):** 
  `apply_setpoint()` advances robot pose internally via unicycle kinematics.
- **REAL MODE:** 
  `apply_setpoint()` transmits setpoints to hardware; display updates only from
  robot-reported pose/scan data. Loop remains synchronous and blocking.

Main Loop Sequence
------------------
SENSE → (ICP) → FUSE → MAP → PLAN → ACT → LOG/VIZ

1) Pose & LiDAR acquisition  
2) ICP alignment and gated fusion  
3) Occupancy grid update  
4) Path planning (`determine_navigation_path()`)  
5) Setpoint computation (`compute_setpoint()`)  
6) Motion update (`apply_setpoint()`)  
7) Visualisation and CSV logging  

Modes
-----
- **KNOWN:** Preplanned A* path to fixed goal.  
- **UNKNOWN:** Frontier-based exploration until goal discovered.  
- **GOALSEEKING:** Path-following using lookahead setpoints.  

Termination
------------
Loop ends when the robot reaches the goal (`arrival_tol_m`) or user quits ('q').

Notes
-----
- All localisation, mapping, and control logic run in one synchronous loop.
- For real-robot use, implement:
      get_pose(), get_scan(), apply_setpoint()
- Candidates only modify `determine_frontier_path()` for the unknown-world task.
"""

from util import *

# -----------------------------------------------------------------------------
# This is the main simulation configuration
# -----------------------------------------------------------------------------
DEFAULTS: Dict[str, Dict] = {
    "world": {
        "wall_half_thickness_m": 0.005,
        "border_thickness_m": 0.01,
    },
    "snake_maze": {
        "size_m": 1.80,
        "cell_size_m": 0.45,
        "num_walls": 4,
        "gap_cells": 1,
    },
    "random_maze": {
        "size_m": 1.80,
        "cell_size_m": 0.45,
        "random_wall_count": 5,
        "random_seed": None,
        "candidates_to_list": 3,
        "seed_scan_start": 0,
        "seed_scan_stride": 1,
        "max_attempts_per_page": 10000,
        "segment_len_cells_min": 1,
        "segment_len_cells_max": 2,
        "orientation_bias": 0.5,
    },
    "planning": {
        "sample_step_m": 0.03,
        "resample_ds_m": 0.05,
        "equal_eps": 1e-6,
        "seg_eps": 1e-9,
    },
    "lidar": {
        "num_rays": 360,
        "max_range_m": 3.0,
        "raycast_eps": 1e-6,
    },
    "ogm": {
        "xyreso_m": 0.03,
        "l_free": -0.4,
        "l_occ": 0.85,
        "l_min": -4.0,
        "l_max": 4.0,
        "hit_margin_m": 1e-3,
        "prob_free_max": 0.35,
        "prob_occ_min": 0.65,
        "size_eps": 1e-9,
        "gray_free": 0.9,
        "gray_occ": 0.0,
        "gray_unk": 1.0,
    },

    "icp_fusion": {
        "enabled": True,
        "alpha": 0.1,
        "max_trans_m": 0.20,
        "max_rot_deg": 20.0,
        "min_points": 50,
        "max_rmse_m": 0.05,
        "snap_trans_m": 0.02,
        "snap_rot_deg": 2.0,
    },
    "viz": {
        "main_figsize_in": (14, 10),
        "robot_arrow_len_m": 0.05,
        "robot_arrow_head_m": (0.03, 0.03),
        "ogm_arrow_len_m": 0.05,
        "ogm_arrow_head_m": (0.03, 0.03),
        "lidar_alpha": 0.2,
        "lidar_lw": 0.5,
        "thumb_size_in": (3, 3),
        "pause_s": 0.01,
    },
    "logging": {
        "level": logging.INFO,
        "format": "[%(levelname)s] %(message)s",
        "pose_csv": "pose.csv",
        "lidar_csv": "lidar.csv",
    },
    "app": {
        "arrival_tolerance_m": 0.1,
        "mode": "GOALSEEKING",  # fixed mode
        "map_type": "RANDOM",  # RANDOM | SNAKE
        "entrance_cell": (0, 0),
        "snake_goal_cell": (3, 3),
        "random_goal_cell": (3, 3),
    },
    "robot": {
        "robot_radius_m": 0.15,
        "turn_angle_rad": math.radians(36),
        "k_ang": 10,
        "v_max_mps": 1.0,  # may be 0.35 for real robot
        "dt_s": 0.1,
        "dt_guard_s": 1e-3,
    },
    "setpoing_cfg": {
        "lookahead_m": 0.3,
    },    
}

def install_key_to_viz(viz: Dict) -> None:
    """Attach keyboard listeners for the live plot window."""
    def _on_key(event):
        globals()["_LAST_KEY"] = event.key
    viz["fig"].canvas.mpl_connect("key_press_event", _on_key)

logging.basicConfig(level=DEFAULTS["logging"]["level"], format=DEFAULTS["logging"]["format"])
log = logging.getLogger("maze_app")

# -----------------------------------------------------------------------------
# This is the main application loop
# -----------------------------------------------------------------------------

def main() -> None:

# -----------------------------------------------------------------------------
# The following is the initial setup including user input, maze world generation, entrance and goal "cell" coordinates,
# initial path planning (mainly for the known maze), lidar, occupancy grid map (OGM), visualisation and logging setup. 
# -----------------------------------------------------------------------------
    settings = copy.deepcopy(DEFAULTS)
    app = ask_options(settings)
    nav_mode = choose_navigation_mode(settings)

    world, entrance, goal_cell = build_world(settings, app)
    planner = create_planner(world, settings["planning"]["sample_step_m"], settings["robot"]["robot_radius_m"])
    path = initialise_navigation_path(planner, entrance, goal_cell, settings, nav_mode)
    sensor = create_lidar(settings["lidar"])
    ogm = create_ogm(settings["ogm"], 0.0, 0.0, world["size_m"], world["size_m"])
    viz = create_viz(world["size_m"], world["cell_size_m"], settings["viz"], settings["robot"]["robot_radius_m"])
    logger_dict = create_logger(settings["lidar"]["num_rays"], settings["logging"])
    start_x, start_y = cell_center(entrance, world["cell_size_m"])
    start_heading = math.atan2(path[1][1] - start_y, path[1][0] - start_x) if len(path) >= 2 else 0.0
    astar_pts = planner["cspace"] if planner["cspace"] else planner["obstacles"]

    state = SimulationState(
        world=world,
        entrance=entrance,
        goal=make_goal(goal_cell),
        path=path,
        sensor=sensor,
        ogm=ogm,
        viz=viz,
        logger=logger_dict,
        pose=make_pose(start_x, start_y, start_heading),
        settings=settings,
        icp_prev_pts=None,
        icp_prev_pose=None,
        step=0,
        astar_pts=astar_pts,
        ctrl=settings["setpoing_cfg"].copy(),
        planner=planner,
    )
    state.robot_iface = load_robot_interface(state.settings)

    install_key_to_viz(state.viz)

    while True:
        key = globals().get("_LAST_KEY", None)
        globals()["_LAST_KEY"] = None
        if key == "q":
            print("Quit requested.")
            break
# -----------------------------------------------------------------------------
# Interface to simulated robot data or real robot data
# For real robot data, simply load the real robot data via the load_robot_interface()
# -----------------------------------------------------------------------------
        robot = state.robot_iface
        if robot is None:
            robot = state.robot_iface = load_robot_interface(state.settings)

# -----------------------------------------------------------------------------
# Main navigation pipeline
# read robot (pose, lidar) --> ICP matching (pose estimation) --> pose fusion --> update OGM --> path planning --> setpoint control --> apply to robot --> map visualisation
# -----------------------------------------------------------------------------
        pose = robot.get_pose(state)
        state.pose = pose
        scan_data = robot.get_scan(state, pose)
        curr_pts = icp_points(pose, scan_data, state.settings["lidar"])
        state.icp_prev_pts, state.icp_prev_pose = curr_pts, pose
        icp_pose, rmse, n_pts, tf_pts = icp_match_step(state.icp_prev_pts, curr_pts, state.icp_prev_pose)
        pose = fuse_icp_pose(state.settings, pose, icp_pose, rmse, n_pts)
        state.pose = pose
        update_ogm(state.ogm, scan_data, pose)
        determine_navigation_path(state)
        setpoint = compute_setpoint(state.ctrl, state.path, pose)

        new_pose = robot.apply_setpoint(state, pose, setpoint)
        state.pose = new_pose
        state.step += 1
# -----------------------------------------------------------------------------
# Visualisation and Logging
# -----------------------------------------------------------------------------
        render(state.viz, state.world, state.ogm, pose, scan_data, state.goal, state.step, state.path, state.entrance, state.icp_prev_pts, curr_pts, tf_pts, state.astar_pts, state.frontier_goal, state.frontier_candidates)

        with state.logger["pose"].open("a", newline="") as handle:
            csv.writer(handle).writerow([state.step, new_pose["x"], new_pose["y"], math.degrees(new_pose["theta"]), state.settings["app"]["mode"]])

        nav_mode = state.settings.get("navigation", {}).get("mode", "KNOWN")
        if state.frontier_goal:
            fgx, fgy = cell_center(state.frontier_goal, state.world["cell_size_m"])
            fg_dist = math.hypot(fgx - new_pose["x"], fgy - new_pose["y"])
        else:
            fgx = fgy = fg_dist = float("nan")
        frontier_cells = ";".join(f"{cell[0]}:{cell[1]}" for cell in state.frontier_candidates) if state.frontier_candidates else ""
        path_length = len(state.path)
        if state.path:
            path_first_x, path_first_y = state.path[0]
        else:
            path_first_x = path_first_y = float("nan")

        diag_icp_x = diag_icp_y = diag_icp_theta = float("nan")
        diag_rmse = float("nan")
        diag_pts = 0
        diag_icp_x = icp_pose["x"]
        diag_icp_y = icp_pose["y"]
        diag_icp_theta = math.degrees(icp_pose["theta"])
        diag_rmse = rmse if rmse is not None else float("nan")
        diag_pts = n_pts

        with state.logger["diag"].open("a", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    state.step, nav_mode, new_pose["x"], new_pose["y"], math.degrees(new_pose["theta"]), fgx, fgy, fg_dist,
                    f"{state.frontier_goal[0]}:{state.frontier_goal[1]}" if state.frontier_goal else "", len(state.frontier_candidates),
                    frontier_cells, path_length, path_first_x, path_first_y, diag_icp_x, diag_icp_y, diag_icp_theta, diag_rmse, diag_pts,
                ]
            )
        row = [state.step]
        for angle, distance in zip(scan_data["angles"], scan_data["ranges"]):
            row.extend([math.degrees(angle), distance])

        with state.logger["lidar"].open("a", newline="") as handle:
            csv.writer(handle).writerow(row)

        icp_info = f" | icp_pose=({icp_pose['x']:.3f},{icp_pose['y']:.3f},{math.degrees(icp_pose['theta']):.1f}°)"
        
        log.info("Step %05d | Maze World = %s | pose=(%.2f,%.2f,%.1f°)%s | setpoint=(%.2f,%.2f,%.1f°)", state.step, state.settings.get("navigation", {}).get("mode", "KNOWN").upper(), new_pose["x"], new_pose["y"], math.degrees(new_pose["theta"]), icp_info, setpoint["x"], setpoint["y"], math.degrees(setpoint["theta"])) 

# -----------------------------------------------------------------------------
# Stopping condition
# -----------------------------------------------------------------------------
        goal_x, goal_y = cell_center(state.goal["cell"], state.world["cell_size_m"])
        if math.hypot(goal_x - pose["x"], goal_y - pose["y"]) <= state.settings["app"]["arrival_tolerance_m"]:
            print("Simulation complete: Robot reached the goal.")
            log.info("Reached goal; stopping.")
            plt.show(block=True)
            break

    print("Done.")
    plt.close("all")

# -----------------------------------------------------------------------------
# WinterHack 2025: Candidate Selection Challenge
# The following function is to be completed by candidates as part of the challenge.
# Candidates only modify the code within the designated section. Candidates should not
# change the function signature, docstring, or any code outside the designated section.
# -----------------------------------------------------------------------------

def determine_frontier_path(state: SimulationState) -> None:
    """
    Determines and sets the frontier path for robot navigation in an unknown environment.
    This function identifies the next frontier cell to explore and plans a path to it. If the current
    frontier goal matches the ultimate goal cell, it returns that cell. Otherwise, it detects new
    frontiers and selects the most promising one based on various criteria including heading alignment,
    forward progress towards the goal, and distances.
    
    Args:
        state (SimulationState): The current simulation state containing robot pose, world information,
                                goals, and other navigation parameters.
    Returns:
         None. Modifies the state in place by setting the `frontier_goal` and `path` attributes.
         `frontier_goal` is a cell representing the chosen frontier to explore.
         `path` is a list of cells representing the plan to reach the `frontier_goal`.

    The function is expected to perform the following key steps:
    1. Checks if current frontier matches the overall goal.
    2. If not, detects new frontiers and their distances 
    3. Select a frontier based on:
       - Heading alignment with robot's current orientation
       - Forward progress towards goal
       - Distance from robot
       - Proximity to ultimate goal
    3. Plans a path to the selected frontier. 
    4. Update:
        - state.frontier_goal with selected frontier
        - state.path with planned path to the selected frontier
    """

    if state.frontier_goal == state.goal["cell"]:
        print("Found matched frontier to goal")
        return state.frontier_goal
    
    else:
        frontiers, distances = detect_frontiers(state)
        goal_cell = state.goal["cell"]
    
        state.frontier_candidates = frontiers
        state.frontier_distances = distances

    #-----START: To be completed by candidate-----
    #-----START: Stable Frontier Selection with Local A* and Hysteresis-----

    def manhattan_distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def euclidean_distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def is_free_or_unknown(cell, state, grid_size):
        if not (0 <= cell[0] < grid_size and 0 <= cell[1] < grid_size):
            return False
        wx, wy = cell_center(cell, state.world["cell_size_m"])
        ix = int((wx - state.ogm["minx"]) / state.ogm["res"])
        iy = int((wy - state.ogm["miny"]) / state.ogm["res"])
        if not (0 <= ix < state.ogm["grid"].shape[1] and 0 <= iy < state.ogm["grid"].shape[0]):
            return False
        p = 1 / (1 + math.exp(-state.ogm["grid"][iy, ix]))
        return p <= state.ogm["cfg"].get("prob_free_max", 0.35) or p < 0.5

    def local_a_star(start, goal, state, grid_size, max_steps=20):
        """在局部范围内运行 A*，避免全局计算开销"""
        from heapq import heappush, heappop
        open_set = []
        heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: manhattan_distance(start, goal)}
        
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        
        while open_set and len(came_from) < max_steps:
            _, current = heappop(open_set)
            if current == goal or manhattan_distance(current, goal) <= 1:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return [start] + path if path else [start]
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if not is_free_or_unknown(neighbor, state, grid_size):
                    continue
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + manhattan_distance(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        return [start]  # fallback

    # 主逻辑
    start_cell = pose_to_cell(state.world, state.pose)
    goal_cell = state.goal["cell"]
    grid_size = state.world.get("grid_size", int(round(state.world["size_m"] / state.world["cell_size_m"])))

    # 初始化历史状态
    if not hasattr(state, 'last_frontier_goal'):
        state.last_frontier_goal = None
    if not hasattr(state, 'consecutive_same_goal'):
        state.consecutive_same_goal = 0

    # >>>>>>>>>> 新增：强制将 goal cell 视为 frontier（如果合理） <<<<<<<<<<
    goal_is_reachable = is_free_or_unknown(goal_cell, state, grid_size)
    goal_is_adjacent_to_known = False
    if goal_is_reachable:
        # 检查 goal_cell 是否与任何已知 free cell 相邻（即是否“可接触”）
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nb = (goal_cell[0] + dx, goal_cell[1] + dy)
            if (0 <= nb[0] < grid_size and 0 <= nb[1] < grid_size and
                is_free_or_unknown(nb, state, grid_size)):
                goal_is_adjacent_to_known = True
                break

    # 如果 goal 尚未到达，且可接触，则强制加入 frontier 候选
    enhanced_frontiers = list(frontiers)
    if goal_cell != start_cell and goal_is_reachable and goal_is_adjacent_to_known:
        if goal_cell not in enhanced_frontiers:
            enhanced_frontiers.append(goal_cell)
            # 为它估算一个合理的距离（比如 Manhattan）
            if goal_cell not in distances:
                distances[goal_cell] = manhattan_distance(start_cell, goal_cell)

    # 使用 enhanced_frontiers 替代 frontiers
    frontiers_to_use = enhanced_frontiers
    # >>>>>>>>>> END 新增逻辑 <<<<<<<<<<

    # 如果已有 frontiers（或增强后的），优先从中选；否则 fallback 到目标方向
    candidate_targets = []
    if frontiers_to_use:
        robot_theta = state.pose["theta"]
        heading_vec = (math.cos(robot_theta), math.sin(robot_theta))
        best_score = -float('inf')
        best_candidate = None

        for cell in frontiers_to_use:
            dist_to_robot = euclidean_distance(start_cell, cell)
            if dist_to_robot == 0:
                continue

            to_cell = (cell[0] - start_cell[0], cell[1] - start_cell[1])
            norm = math.hypot(to_cell[0], to_cell[1])
            align = 0
            if norm > 0:
                dir_vec = (to_cell[0]/norm, to_cell[1]/norm)
                align = dir_vec[0] * heading_vec[0] + dir_vec[1] * heading_vec[1]

            progress = manhattan_distance(start_cell, goal_cell) - manhattan_distance(cell, goal_cell)
            switch_penalty = -2.0 if (state.last_frontier_goal and cell != state.last_frontier_goal) else 0

            # >>>>>>>>>> 关键：给 goal_cell 极高优先级 <<<<<<<<<<
            goal_bonus = 1000.0 if cell == goal_cell else 0.0

            score = align * 1.0 + progress * 0.5 - dist_to_robot * 0.1 + switch_penalty + goal_bonus
            if score > best_score:
                best_score = score
                best_candidate = cell

        if best_candidate:
            candidate_targets.append(best_candidate)


    # 如果没有 frontiers 或选不出，就朝 goal 方向局部探索
    if not candidate_targets:
        # 在 goal 方向找一个 nearby free cell（within 3 steps）
        ideal_dir = (goal_cell[0] - start_cell[0], goal_cell[1] - start_cell[1])
        norm = max(math.hypot(*ideal_dir), 1)
        unit_dir = (ideal_dir[0]/norm, ideal_dir[1]/norm)
        # 尝试沿方向走 1~3 步
        for d in [1, 2, 3]:
            guess = (int(start_cell[0] + d * unit_dir[0]), int(start_cell[1] + d * unit_dir[1]))
            if is_free_or_unknown(guess, state, grid_size):
                candidate_targets.append(guess)
                break
        if not candidate_targets:
            candidate_targets.append(start_cell)  # 最后兜底

    next_target = candidate_targets[0]

    # 防止震荡：如果新目标与上次不同，但距离很近，且方向相反，优先保持原目标
    if (state.last_frontier_goal and 
        next_target != state.last_frontier_goal and
        euclidean_distance(next_target, state.last_frontier_goal) < 2):
        # 检查是否“回头”：从 last -> current -> next 是否 U-turn
        vec1 = (start_cell[0] - state.last_frontier_goal[0], start_cell[1] - state.last_frontier_goal[1])
        vec2 = (next_target[0] - start_cell[0], next_target[1] - start_cell[1])
        if vec1[0]*vec2[0] + vec1[1]*vec2[1] < -0.5:  # 夹角 > 120度
            next_target = state.last_frontier_goal
            state.consecutive_same_goal += 1
        else:
            state.consecutive_same_goal = 0
    else:
        if next_target == state.last_frontier_goal:
            state.consecutive_same_goal += 1
        else:
            state.consecutive_same_goal = 0

    # 如果连续多次选同一个目标，可以稍微“信任”它，即使暂时被挡
    state.last_frontier_goal = next_target
    best_frontier_cell = next_target

    #-----END: Stable Frontier Selection-----
    #-----END: To be completed by candidate-----
    state.frontier_goal = best_frontier_cell
    start_cell = pose_to_cell(state.world, state.pose)
    state.path = plan_unknown_world(state, start_cell, state.frontier_goal)
    return


def detect_frontiers(state: SimulationState) -> Tuple[List[Cell], Dict[Cell, int]]:
    """
    Detect frontier cells in an occupancy grid map using a breadth-first search from the robot pose.
    Parameters
    ----------
    state : SimulationState
        The simulation state object providing the world and map information required for frontier
        detection. Expected fields and structure:
          - state.settings: a dict; navigation mode is read from
            state.settings.get("navigation", {}).get("mode", "KNOWN"). Mode must be the string
            "UNKNOWN" (case-insensitive) for frontier detection to run; otherwise the function
            returns ([], {}).
          - state.ogm: a dict describing the occupancy grid map with keys:
              - "grid": 2D numpy array (float) of log-odds or similar values. The code converts this to
                probabilities using the logistic/sigmoid function: p = 1 / (1 + exp(-grid)).
              - "cfg": a dict of optional configuration thresholds:
                  - "prob_free_max" (float, default 0.35) — cells with p <= prob_free_max are treated as free.
                  - "prob_occ_min"  (float, default 0.65) — cells with p >= prob_occ_min are treated as occupied.
              - "minx", "miny" (float) — origin of the occupancy grid in world coordinates.
              - "res" (float) — grid resolution (meters per grid cell).
          - state.world: a dict with world/grid parameters:
              - "cell_size_m" (float) — cell size used by pose_to_cell / cell_center.
              - either "grid_size" (int) or "size_m" (float). If "grid_size" not present, an integer grid
                size is computed as round(size_m / cell_size_m). grid_size must be > 0.
          - state.pose: robot pose used as the BFS start, converted to a starting grid cell using
            pose_to_cell(state.world, state.pose).
    Returns
    -------
    Tuple[List[Cell], Dict[Cell, int]]
        - frontier_cells: list of Cell (tuples of ints, e.g. (cx, cy)) that are reachable free cells
          adjacent (4-connected) to at least one "unknown" cell. The list is sorted by descending
          distance (farthest reachable first) and then by the cell coordinates as a tie-breaker.
        - frontier_distances: dict mapping each returned frontier cell to its integer Manhattan-style
          distance (number of 4-connected steps) from the start cell discovered by the BFS.
    Behavior and details
    --------------------
    - Early exits:
        - If navigation mode is not "UNKNOWN" (after uppercasing), returns ([], {}).
        - If state.ogm is missing or ogm["grid"] is empty, returns ([], {}).
        - If grid_size <= 0, returns ([], {}).
        - If the start cell (pose_to_cell(state.world, state.pose)) classifies as "occupied",
          returns ([], {}).
    - Cell classification:
        - The inner classification converts a cell index to world coordinates using cell_center(cell, cell_size),
          converts to occupancy grid indices (ix, iy) with (wx - minx)/res and (wy - miny)/res,
          and returns:
            - "occupied" if (ix, iy) is out of the occupancy-grid bounds or p >= prob_occ_min
            - "free"     if p <= prob_free_max
            - "unknown"  otherwise (probability between the free and occupied thresholds)
        - Occupancy probabilities are obtained with a sigmoid applied to the raw grid values.
    - Search and frontier definition:
        - Performs a BFS (4-connected neighbors) starting from the robot cell, exploring only cells
          classified as "free" and bounded by the provided grid_size.
        - A frontier cell is any reachable free cell that has at least one 4-connected neighbor
          classified as "unknown".
        - Only reachable free cells are considered when forming frontiers; occupied or out-of-bounds
          neighbors block traversal.
    - Output ordering and contents:
        - frontier_cells is sorted by (-distance, cell) so that cells farther from the start appear first.
        - frontier_distances contains distances only for those cells present in frontier_cells.
    Complexity
    ----------
    - Time: O(V) where V is the number of free cells visited by the BFS (bounded by grid_size^2 in worst-case).
      Each visited cell checks up to four neighbors and classification uses constant-time operations (array access).
    - Space: O(V) for the BFS queue and the distances mapping.
    Notes
    -----
    - This function relies on helper functions/constructs not defined here: pose_to_cell(world, pose)
      and cell_center(cell, cell_size). The type alias Cell is assumed to be a 2-tuple of ints.
    - The exact numeric behavior depends on how the occupancy grid (ogm["grid"]) stores values
      (log-odds or other); this function treats those values as inputs to a sigmoid to obtain a
      probability in [0, 1].
    - The thresholds prob_free_max and prob_occ_min are inclusive as implemented (<= free and >= occ).
    """

    from collections import deque

    def classify(cell: Cell) -> str:
        cx, cy = cell
        wx, wy = cell_center(cell, cell_size)
        ix = int((wx - minx) / res)
        iy = int((wy - miny) / res)
        if not (0 <= ix < width and 0 <= iy < height):
            return "occupied"
        p = prob[iy, ix]
        if p >= occ_thresh:
            return "occupied"
        if p <= free_thresh:
            return "free"
        return "unknown"
    
    mode = state.settings.get("navigation", {}).get("mode", "KNOWN").upper()
    if mode != "UNKNOWN":
        return [], {}

    ogm = state.ogm
    if not ogm or ogm["grid"].size == 0:
        return [], {}

    grid = ogm["grid"]
    cfg = ogm["cfg"]
    prob = 1 / (1 + np.exp(-grid))
    free_thresh = cfg.get("prob_free_max", 0.35)
    occ_thresh = cfg.get("prob_occ_min", 0.65)

    cell_size = state.world["cell_size_m"]
    grid_size = state.world.get("grid_size", int(round(state.world["size_m"] / cell_size)))
    if grid_size <= 0:
        return [], {}

    width = grid.shape[1]
    height = grid.shape[0]
    minx = ogm["minx"]
    miny = ogm["miny"]
    res = ogm["res"]

    start_cell = pose_to_cell(state.world, state.pose)
    if classify(start_cell) == "occupied":
        return [], {}

    queue: "deque[Cell]" = deque([start_cell])
    distances: Dict[Cell, int] = {start_cell: 0}

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < grid_size and 0 <= ny < grid_size):
                continue
            cell = (nx, ny)
            if cell in distances:
                continue
            if classify(cell) != "free":
                continue
            distances[cell] = distances[(cx, cy)] + 1
            queue.append(cell)

    frontier_cells: List[Cell] = []
    for cell, dist in distances.items():
        cx, cy = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (cx + dx, cy + dy)
            if not (0 <= nb[0] < grid_size and 0 <= nb[1] < grid_size):
                continue
            if classify(nb) == "unknown":
                frontier_cells.append(cell)
                break

    if not frontier_cells:
        return [], {}

    frontier_cells.sort(key=lambda cell: (-distances[cell], cell))
    frontier_distances = {cell: distances[cell] for cell in frontier_cells}
    return frontier_cells, frontier_distances


def determine_navigation_path(state: SimulationState) -> None:
    """
    Determines the navigation path to the goal cell based on the current simulation state.
    If the navigation mode is set to "UNKNOWN", computes a path to the frontier using
    `determine_frontier_path`. Otherwise, assumes the world is known and the path to the
    goal cell has already been determined during initialization.
    Args:
        state (SimulationState): The current simulation state containing settings and goal information.
    Returns:
        None
    """

    mode = state.settings.get("navigation", {}).get("mode", "KNOWN").upper()

    if mode == "UNKNOWN":
        determine_frontier_path(state)
        return
    else:
        #------------------------------
        # Known world: path to the goal cell already determined at initialisation
        #------------------------------
        if not state.path:
            determine_goal_path(state)
        return

if __name__ == "__main__":
    main()
