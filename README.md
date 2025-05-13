# Python Code for Evaluation of experiments
## Robot workspace exploration using collision detection and isolation
_This repository contains Python code developed as supplementary material for the bachelor's thesis._
### 1.1 Input Data

The primary input consists of 3D coordinate points (x, y, z) representing the centers of spheres. This data is sourced from CSV files.

Data are for those combinations of measurements: 
* Box placement `right, mid, left`
* Searching strategies `End-effector sweep, FUll-arm sweep`

Each row in these CSV files contains the following 'x', 'y', and 'z' columns for the coordinates.

### 2.1. Core Parameters
Key parameters:

* **Collision box definition:**
    * `box_center`: A list defining the [x, y, z] coordinates of the center of the rectangular collision box. .
    * `box_dimensions`: A list defining the [width, height, depth] of the bounding box.
* **Sphere Definition:**
    * `fixed_sphere_diameter`: A floating-point value for the diameter of all spheres.
