import random
import time
import os
import heapq
from collections import deque

def get_user_input(prompt, default, input_type=int, min_val=None, max_val=None):
    """Get user input with validation and default values"""
    while True:
        try:
            user_input = input(f"{prompt} [{default}]: ").strip()
            if not user_input:  # If empty, use default
                return default
            
            value = input_type(user_input)
            
            # Validate range if specified
            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}.")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be at most {max_val}.")
                continue
                
            return value
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__}.")

def setup_parameters():
    """Get all parameters from user input"""
    print("\n" + "="*50)
    print("ACO PATHFINDING PARAMETER SETUP")
    print("="*50)
    print("Enter values or press Enter to use defaults.\n")
    
    # Grid parameters
    rows = get_user_input("Enter number of rows", 7, int, 3, 20)
    cols = get_user_input("Enter number of columns", 8, int, 3, 20)
    grid_size = rows * cols
    
    obstacle_ratio = get_user_input("Enter obstacle ratio (0.0-0.5)", 0.2, float, 0.0, 0.5)
    obstacle_count = int(grid_size * obstacle_ratio)
    
    # Start and goal positions
    print("\nStart and goal positions (0-indexed):")
    start_x = get_user_input("Start X position", 1, int, 0, rows-1)
    start_y = get_user_input("Start Y position", 0, int, 0, cols-1)
    goal_x = get_user_input("Goal X position", rows-2, int, 0, rows-1)
    goal_y = get_user_input("Goal Y position", cols-1, int, 0, cols-1)
    
    # Check if start and goal are the same
    while (start_x, start_y) == (goal_x, goal_y):
        print("Start and goal cannot be the same position. Please re-enter goal position.")
        goal_x = get_user_input("Goal X position", rows-1, int, 0, rows-1)
        goal_y = get_user_input("Goal Y position", cols-1, int, 0, cols-1)
    
    # ACO parameters
    print("\nACO algorithm parameters:")
    num_ants = get_user_input("Number of ants", 10, int, 1, 100)
    iterations = get_user_input("Number of iterations", 20, int, 1, 100)
    
    # Advanced parameters
    print("\nAdvanced parameters (press Enter for defaults):")
    alpha_min = get_user_input("Pheromone importance (min)", 0.5, float, 0.1, 5.0)
    alpha_max = get_user_input("Pheromone importance (max)", 3.0, float, alpha_min, 5.0)
    
    beta_min = get_user_input("Heuristic importance (min)", 1.0, float, 0.1, 10.0)
    beta_max = get_user_input("Heuristic importance (max)", 5.0, float, beta_min, 10.0)
    
    evap_min = get_user_input("Evaporation rate (min)", 0.3, float, 0.1, 0.9)
    evap_max = get_user_input("Evaporation rate (max)", 0.8, float, evap_min, 0.9)
    
    q_min = get_user_input("Pheromone deposit amount (min)", 50, float, 1.0, 500.0)
    q_max = get_user_input("Pheromone deposit amount (max)", 200, float, q_min, 500.0)
    
    hybrid_weight = get_user_input("Hybrid weight (ACO-Dijkstra balance, 0.0-1.0)", 0.5, float, 0.0, 1.0)
    
    max_backtrack = get_user_input("Maximum backtracking steps", 5, int, 0, 20)
    
    # Animation speed
    animation_speed = get_user_input("Animation delay (seconds)", 0.2, float, 0.0, 5.0)
    
    # Create parameter dictionary
    params = {
        "rows": rows,
        "cols": cols,
        "obstacle_ratio": obstacle_ratio,
        "obstacle_count": obstacle_count,
        "start": (start_x, start_y),
        "goal": (goal_x, goal_y),
        "alpha_range": (alpha_min, alpha_max),
        "beta_range": (beta_min, beta_max),
        "evaporation_range": (evap_min, evap_max),
        "Q_range": (q_min, q_max),
        "num_ants": num_ants,
        "iterations": iterations,
        "hybrid_weight": hybrid_weight,
        "max_backtrack_steps": max_backtrack,
        "animation_speed": animation_speed
    }
    
    # Print summary
    print("\n" + "="*50)
    print("PARAMETER SUMMARY")
    print("="*50)
    print(f"Grid Size: {rows}x{cols} with {obstacle_count} obstacles ({obstacle_ratio:.1%})")
    print(f"Start: ({start_x},{start_y}), Goal: ({goal_x},{goal_y})")
    print(f"Ants: {num_ants}, Iterations: {iterations}")
    print(f"Hybrid Weight: {hybrid_weight}, Max Backtrack: {max_backtrack}")
    print("="*50)
    
    confirm = input("\nPress Enter to start or 'r' to reenter parameters: ").lower()
    if confirm == 'r':
        return setup_parameters()
    
    return params

# ANSI colors - These work on most terminals
class Color:
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Regular colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

def clear_screen():
    """Clear the terminal screen based on OS"""
    os.system('cls' if os.name == 'nt' else 'clear')

def generate_grid(params):
    """Generate a new grid with random obstacles"""
    rows, cols = params["rows"], params["cols"]
    obstacle_count = params["obstacle_count"]
    start, goal = params["start"], params["goal"]
    
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    obstacles = set()
    while len(obstacles) < obstacle_count:
        x = random.randint(0, rows - 1)
        y = random.randint(0, cols - 1)
        if (x, y) != start and (x, y) != goal:
            obstacles.add((x, y))
            grid[x][y] = 1
    return grid

def get_neighbors(pos, grid, rows, cols):
    """Get valid neighboring cells (not obstacles)"""
    x, y = pos
    moves = [(0,1), (1,0), (0,-1), (-1,0)]  # Right, Down, Left, Up
    result = []
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
            result.append((nx, ny))
    return result

def heuristic(a, b):
    """Manhattan distance heuristic"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def dijkstra(grid, start, goal, rows, cols):
    """A* algorithm for shortest path finding"""
    # Priority queue for A*
    queue = [(0, start)]
    cost_so_far = {start: 0}
    came_from = {start: None}
    
    while queue:
        current_cost, current = heapq.heappop(queue)
        
        if current == goal:
            break
            
        for next_cell in get_neighbors(current, grid, rows, cols):
            new_cost = cost_so_far[current] + 1
            
            if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                cost_so_far[next_cell] = new_cost
                priority = new_cost + heuristic(goal, next_cell)
                heapq.heappush(queue, (priority, next_cell))
                came_from[next_cell] = current
    
    # Reconstruct path
    if goal not in came_from:
        return None
        
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    
    return path

def choose_next(current, visited, grid, pheromones, alpha, beta, dijkstra_path, dijkstra_influence, dead_ends, params):
    """Choose next cell for an ant to move to"""
    goal = params["goal"]
    rows, cols = params["rows"], params["cols"]
    
    neighbors = get_neighbors(current, grid, rows, cols)
    probabilities = []
    total = 0
    
    # Filter out known dead ends from neighbors
    valid_neighbors = [n for n in neighbors if n not in dead_ends]
    if not valid_neighbors and neighbors:
        # If all neighbors are marked as dead ends but we have neighbors,
        # we're forced to try one (might be our only way forward)
        valid_neighbors = neighbors
    
    for n in valid_neighbors:
        if n in visited:
            continue
            
        # Combine pheromone information with heuristic distance
        pher = pheromones[n[0]][n[1]] ** alpha
        dist = max(0.1, heuristic(n, goal))  # Avoid division by zero
        
        # Add Dijkstra influence if this cell is on the Dijkstra path
        dijkstra_bonus = 1.0
        if dijkstra_path and n in dijkstra_path:
            dijkstra_bonus += dijkstra_influence
            
        score = pher * (1.0 / dist) ** beta * dijkstra_bonus
        probabilities.append((n, score))
        total += score
        
    if not probabilities:
        return None  # No valid moves, will trigger backtracking
        
    # Probabilistic selection
    r = random.uniform(0, total)
    upto = 0
    for n, score in probabilities:
        upto += score
        if upto >= r:
            return n
            
    return probabilities[-1][0]  # Fallback to last option

def run_ant_with_backtracking(grid, pheromones, alpha, beta, dijkstra_path, dijkstra_influence, params, dead_ends):
    """Run an ant with backtracking capability to avoid dead ends"""
    start = params["start"]
    goal = params["goal"]
    max_backtrack_steps = params["max_backtrack_steps"]
    
    path = [start]
    visited = set([start])
    current = start
    backtrack_count = 0
    
    while current != goal:
        next_cell = choose_next(current, visited, grid, pheromones, alpha, beta, 
                               dijkstra_path, dijkstra_influence, dead_ends, params)
        
        if next_cell is None:
            # Dead end reached, backtrack if possible
            if len(path) > 1 and backtrack_count < max_backtrack_steps:
                # Mark current position as a dead end
                dead_ends.add(current)
                # Remove current position from path
                path.pop()
                # Set current to previous position
                current = path[-1]
                backtrack_count += 1
                continue
            else:
                # Too much backtracking or can't backtrack further
                return None
        
        # Reset backtrack count when moving forward
        backtrack_count = 0
        path.append(next_cell)
        visited.add(next_cell)
        current = next_cell
        
    return path

def update_pheromones(paths, pheromones, evaporation, Q, rows, cols):
    """Update pheromones based on ant paths"""
    # Evaporate pheromones on all cells
    for i in range(rows):
        for j in range(cols):
            pheromones[i][j] *= (1 - evaporation)

    # Add new pheromones based on paths - shorter paths get more pheromones
    for path in paths:
        if not path:
            continue
        contribution = Q / len(path)
        for cell in path:
            pheromones[cell[0]][cell[1]] += contribution

def update_parameters(success_rate, generation, total_generations, current_params, param_ranges):
    """Adaptively tune parameters based on algorithm progress"""
    # Unpack current parameters
    alpha, beta, evaporation, Q = current_params
    alpha_range, beta_range, evaporation_range, Q_range = param_ranges
    
    # Calculate progress percentage
    progress = generation / total_generations
    
    # Adjust parameters based on success rate and progress
    if success_rate < 0.3:  # Poor performance
        # Increase exploration (lower alpha, higher beta)
        alpha = max(alpha_range[0], alpha * 0.9)
        beta = min(beta_range[1], beta * 1.1)
        # Increase evaporation to forget bad paths
        evaporation = min(evaporation_range[1], evaporation * 1.1)
        # Increase pheromone deposit for successful paths
        Q = min(Q_range[1], Q * 1.1)
    elif success_rate > 0.7:  # Good performance
        # Increase exploitation (higher alpha, lower beta)
        alpha = min(alpha_range[1], alpha * 1.1)
        beta = max(beta_range[0], beta * 0.95)
        # Decrease evaporation to strengthen good paths
        evaporation = max(evaporation_range[0], evaporation * 0.95)
        
    # Late-stage fine-tuning: as we approach the end of iterations
    if progress > 0.7:
        # Focus more on exploitation in later stages
        alpha = min(alpha_range[1], alpha * 1.05)
        evaporation = max(evaporation_range[0], evaporation * 0.9)
        
    return alpha, beta, evaporation, Q

def genetic_crossover(paths):
    """Apply genetic algorithm crossover to combine good paths"""
    if len(paths) < 2:
        return paths
        
    # Sort paths by length (shorter is better)
    sorted_paths = sorted(paths, key=lambda p: len(p) if p else float('inf'))
    
    # Take the best paths for crossover
    best_paths = sorted_paths[:max(2, len(sorted_paths)//2)]
    new_paths = best_paths.copy()  # Keep the best paths
    
    # Create new paths through crossover
    for i in range(len(best_paths)):
        for j in range(i+1, len(best_paths)):
            path1 = best_paths[i]
            path2 = best_paths[j]
            
            if not path1 or not path2:
                continue
                
            # Find common points between paths
            common_points = set(path1) & set(path2)
            
            if len(common_points) > 2:  # Need at least one common point besides start and goal
                # Choose a random common point for crossover
                common_point = random.choice(list(common_points - {path1[0], path1[-1]}))
                
                # Find indices of common point in both paths
                idx1 = path1.index(common_point)
                idx2 = path2.index(common_point)
                
                # Create new path by crossover
                new_path = path1[:idx1] + path2[idx2:]
                
                # Ensure no duplicates and path is valid
                if len(set(new_path)) == len(new_path):
                    new_paths.append(new_path)
    
    return new_paths

def print_grid_with_path(grid, path=None, pheromones=None, dead_ends=None, params=None):
    """Print a grid with colored cells and a box for each cell"""
    rows, cols = len(grid), len(grid[0])
    start, goal = params["start"], params["goal"]

    # Print the column numbers as a header
    print("  ", end="")
    for j in range(cols):
        print(f" {j} ", end="")
    print("\n  ", end="")
    for j in range(cols):
        print("---", end="")
    print()

    for i in range(rows):
        # Print row number
        print(f"{i}|", end="")

        for j in range(cols):
            # Choose the appropriate character and color based on cell type
            if (i, j) == start:
                # Start position - Blue
                cell_str = " S "
                color = Color.BG_BLUE + Color.WHITE + Color.BOLD
            elif (i, j) == goal:
                # Goal position - Green
                cell_str = " G "
                color = Color.BG_GREEN + Color.BLACK + Color.BOLD
            elif grid[i][j] == 1:
                # Obstacle - Red
                cell_str = " # "
                color = Color.BG_RED + Color.WHITE
            elif dead_ends and (i, j) in dead_ends:
                # Dead end - Black with X
                cell_str = " X "
                color = Color.BG_BLACK + Color.WHITE
            elif path and (i, j) in path:
                # Path - Yellow with asterisk
                cell_str = " * "
                color = Color.BG_YELLOW + Color.BLACK + Color.BOLD
            elif pheromones:
                # Color based on pheromone level
                max_pher = max(max(row) for row in pheromones)
                pher_level = pheromones[i][j]
                pher_ratio = pher_level / max_pher if max_pher > 0 else 0

                if pher_ratio > 0.7:
                    # High pheromone - Purple
                    cell_str = f"{pher_level:.1f}"[-3:]
                    color = Color.BG_MAGENTA + Color.WHITE
                elif pher_ratio > 0.4:
                    # Medium pheromone - Cyan
                    cell_str = f"{pher_level:.1f}"[-3:]
                    color = Color.BG_CYAN + Color.BLACK
                else:
                    # Low pheromone - White
                    cell_str = "   "
                    color = Color.BG_WHITE + Color.BLACK
            else:
                # Empty cell - White
                cell_str = "   "
                color = Color.BG_WHITE + Color.BLACK

            # Print the colored cell
            print(f"{color}{cell_str}{Color.RESET}", end="")

        print()  # End of row

    print()  # Empty line after grid

def print_legend():
    """Print color legend"""
    print("\n" + "="*30)
    print("COLOR LEGEND:")
    print(f"{Color.BG_BLUE}{Color.WHITE}{Color.BOLD} S {Color.RESET} Start Position")
    print(f"{Color.BG_GREEN}{Color.BLACK}{Color.BOLD} G {Color.RESET} Goal Position")
    print(f"{Color.BG_RED}{Color.WHITE} # {Color.RESET} Obstacle")
    print(f"{Color.BG_YELLOW}{Color.BLACK}{Color.BOLD} * {Color.RESET} Path")
    print(f"{Color.BG_BLACK}{Color.WHITE} X {Color.RESET} Dead End")
    print(f"{Color.BG_MAGENTA}{Color.WHITE}   {Color.RESET} High Pheromone")
    print(f"{Color.BG_CYAN}{Color.BLACK}   {Color.RESET} Medium Pheromone")
    print(f"{Color.BG_WHITE}{Color.BLACK}   {Color.RESET} Low/No Pheromone")
    print("="*30)

def print_stats(iteration, iterations, paths=None, best_path=None, params=None):
    """Print stats about the current iteration"""
    print(f"\n{Color.BOLD}ITERATION {iteration}/{iterations}{Color.RESET}")

    if paths:
        valid_paths = [p for p in paths if p]
        if isinstance(params, dict) and 'num_ants' in params:
            print(f"Ants finding valid paths: {len(valid_paths)}/{params['num_ants']}")
        else:
            print(f"Ants finding valid paths: {len(valid_paths)}")

    if best_path:
        print(f"{Color.GREEN}Best path length: {len(best_path)}{Color.RESET}")
    else:
        print(f"{Color.RED}No valid path found yet{Color.RESET}")
        
    if params:
        # Check if params is a tuple (alpha, beta, evaporation, Q) or a dictionary
        if isinstance(params, tuple) and len(params) == 4:
            alpha, beta, evaporation, Q = params
            print(f"\nCurrent Parameters:")
            print(f"Alpha: {alpha:.2f} | Beta: {beta:.2f}")
            print(f"Evaporation: {evaporation:.2f} | Q: {Q:.1f}")
        
    print("-"*30)

def animate_path(grid, path, dead_ends=None, params=None):
    """Animate the path step by step"""
    if not path:
        return
        
    step_delay = params.get("animation_speed", 0.2)

    print(f"{Color.CYAN}{Color.BOLD}PATH ANIMATION{Color.RESET}")
    for i in range(len(path) + 1):
        clear_screen()
        print(f"{Color.CYAN}{Color.BOLD}STEP {i}/{len(path)}{Color.RESET}")
        partial_path = path[:i]
        print_grid_with_path(grid, partial_path, dead_ends=dead_ends, params=params)
        time.sleep(step_delay)

def run_aco():
    """Run the ACO algorithm with backtracking, hybrid approach, and parameter tuning"""
    # Get parameters from user
    params = setup_parameters()
    
    # Initialize grid and pheromones
    grid = generate_grid(params)
    rows, cols = params["rows"], params["cols"]
    pheromones = [[1.0 for _ in range(cols)] for _ in range(rows)]
    dead_ends = set()

    start = params["start"]
    goal = params["goal"]
    iterations = params["iterations"]
    num_ants = params["num_ants"]
    hybrid_weight = params["hybrid_weight"]
    
    # Parameter ranges
    param_ranges = (
        params["alpha_range"],
        params["beta_range"],
        params["evaporation_range"],
        params["Q_range"]
    )

    # Initial parameter values - start at middle of each range
    alpha = sum(params["alpha_range"]) / 2
    beta = sum(params["beta_range"]) / 2
    evaporation = sum(params["evaporation_range"]) / 2
    Q = sum(params["Q_range"]) / 2
    
    # Calculate Dijkstra's path once for the grid
    dijkstra_path = dijkstra(grid, start, goal, rows, cols)
    dijkstra_influence = hybrid_weight
    
    if dijkstra_path:
        print(f"\n{Color.CYAN}Dijkstra's path found with length: {len(dijkstra_path)}{Color.RESET}")
    else:
        print(f"\n{Color.RED}Dijkstra's algorithm could not find a path!{Color.RESET}")
        dijkstra_influence = 0  # Don't use Dijkstra influence if no path found

    clear_screen()
    print(f"{Color.CYAN}{Color.BOLD}ANT COLONY OPTIMIZATION PATHFINDING{Color.RESET}")
    print("\nInitial Grid:")
    print_grid_with_path(grid, params=params)
    print_legend()

    best_path = None
    best_path_length = float('inf')
    
    # Use Dijkstra's path as initial best if available
    if dijkstra_path:
        best_path = dijkstra_path
        best_path_length = len(dijkstra_path)

    # Wait before starting iterations
    time.sleep(2)

    for it in range(iterations):
        clear_screen()
        print(f"{Color.CYAN}{Color.BOLD}ACO - ITERATION {it+1}/{iterations}{Color.RESET}")

        # Run all ants with backtracking
        all_paths = []
        for _ in range(num_ants):
            path = run_ant_with_backtracking(
                grid, pheromones, alpha, beta, dijkstra_path, 
                dijkstra_influence, params, dead_ends
            )
            all_paths.append(path)

            # Update best path if a better one is found
            if path and (best_path is None or len(path) < best_path_length):
                best_path = path
                best_path_length = len(path)
                
        # Apply genetic algorithm crossover to combine good paths
        if it > iterations // 2:  # Only in later iterations
            combined_paths = genetic_crossover([p for p in all_paths if p])
            all_paths.extend(combined_paths)
            
            # Check if genetic algorithm produced better paths
            for path in combined_paths:
                if path and len(path) < best_path_length:
                    best_path = path
                    best_path_length = len(path)

        # Calculate success rate for parameter tuning
        valid_paths = [p for p in all_paths if p]
        success_rate = len(valid_paths) / num_ants if num_ants > 0 else 0
        
        # Update parameters based on success rate
        alpha, beta, evaporation, Q = update_parameters(
            success_rate, it, iterations, (alpha, beta, evaporation, Q), param_ranges
        )

        # Update pheromones based on all valid paths
        update_pheromones(valid_paths, pheromones, evaporation, Q, rows, cols)
        
        # Collect dead ends from this iteration
        for path in all_paths:
            if not path:  # Failed path might have identified dead ends
                # Update dead_ends (simulated - in real implementation this would come from the ants)
                potential_dead_ends = [
                    (x, y) for x in range(rows) for y in range(cols)
                    if grid[x][y] == 0 and len(get_neighbors((x, y), grid, rows, cols)) <= 1 
                    and (x, y) != start and (x, y) != goal
                ]
                dead_ends.update(potential_dead_ends)

        # Show current state
        print_stats(it+1, iterations, all_paths, best_path, (alpha, beta, evaporation, Q))
        print("\nCurrent Grid with Pheromones:")
        print_grid_with_path(grid, None, pheromones, dead_ends, params)

        if best_path:
            print("\nCurrent Best Path:")
            print_grid_with_path(grid, best_path, None, dead_ends, params)

        time.sleep(0.5)

    # Final result
    clear_screen()
    print(f"{Color.GREEN}{Color.BOLD}FINAL RESULT{Color.RESET}")
    if best_path:
        print(f"Best path found with length: {len(best_path)}")
        print_grid_with_path(grid, best_path, None, dead_ends, params)
        animate_path(grid, best_path, dead_ends, params)
    else:
        print(f"{Color.RED}No path was found!{Color.RESET}")
        print_grid_with_path(grid, params=params)

if __name__ == "__main__":
    try:
        # Try to clear screen - if it fails, terminal might not support ANSI codes
        clear_screen()

        print(f"{Color.BOLD}ACO Pathfinding Algorithm{Color.RESET}")
        print("Features:")
        print("- User-configurable parameters")
        print("- Backtracking to escape dead ends")
        print("- Hybrid approach with Dijkstra's algorithm")
        print("- Adaptive parameter tuning")
        print("- Genetic algorithm crossover")
        print("\nStarting simulation...")
        time.sleep(1)

        # Set random seed for reproducibility (optional)
        random_seed = get_user_input("Enter random seed (0 for random)", 42, int, 0)
        if random_seed > 0:
            random.seed(random_seed)
        
        run_aco()
        
        # Ask if user wants to run again
        while input("\nRun again? (y/n): ").lower().strip() == 'y':
            run_aco()
            
        print("\nThank you for using the ACO Pathfinding Simulator!")

    except Exception as e:
        # Fallback mode if colors aren't supported
        print("\nError: Your terminal might not support ANSI color codes.")
        print(f"Error details: {e}")
        print("\nTry running in a different terminal that supports ANSI colors.")