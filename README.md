# ACO Pathfinding Algorithm

An implementation of Ant Colony Optimization (ACO) for pathfinding in 2D grids with obstacles, featuring backtracking, hybrid ACO-Dijkstra approach, and adaptive parameter tuning.

## Features

- **Customizable Grid Generation**:
  - Adjustable grid size (rows x columns)
  - Configurable obstacle ratio
  - Custom start and goal positions

- **Advanced ACO Implementation**:
  - Multiple ants with individual paths
  - Pheromone trail system
  - Dynamic parameter adaptation
  - Backtracking to escape dead ends

- **Hybrid Approach**:
  - Combines ACO with Dijkstra's algorithm
  - Adjustable hybrid weight parameter

- **Genetic Algorithm Integration**:
  - Path crossover in later iterations
  - Combines features of successful paths

- **Visualization**:
  - Color-coded grid display
  - Path animation
  - Real-time parameter and statistics display

## Requirements

- Python 3.x
- Terminal with ANSI color support (most modern terminals)

## Usage

1. Run the script:
   ```bash
   python aco_pathfinding.py

Example Output:

ACO PATHFINDING PARAMETER SETUP
==================================================
Enter values or press Enter to use defaults.

 - Grid Size: 7x8 with 11 obstacles (20.0%)
 - Start: (1,0), Goal: (5,7)
 - Ants: 10, Iterations: 20
 - Hybrid Weight: 0.5, Max Backtrack: 5

FINAL RESULT
Best path found with length: 11
 ```bash
  0  1  2  3  4  5  6  7  
---------------------------
0|   |   |   |   |   |   |   |   |
1| S | * |   |   | # |   |   |   |
2| * |   | # |   |   |   | # |   |
3| * |   |   | # |   |   |   |   |
4| * | # |   |   |   | # |   |   |
5| * | * | * | * | * | * | * | G |
6|   |   |   |   | # |   |   |   |

License
This project is open-source and available for free use.

Notes
 - For best results, use a terminal that supports ANSI color codes
 - Larger grids may require more iterations/ants
 - The hybrid approach helps when pure ACO struggles
 - Parameters can significantly affect performance
