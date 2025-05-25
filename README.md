# Robot workspace exploration using collision detection and isolation
## Python Code and Measured Data from Thesis
_This repository contains Python code developed as supplementary material for the bachelor's thesis._
### 1 Measured Data
Measured values used for calculations and evaluation in Chapter 4.
* `accuracy_experiments_logs` Data from _Evaluation of contact localization accuracy_ Section 4.1.
* `exploration_strategies_logs` Data from _Evaluation of workspace exploration_ Section 4.2.

Data are stored in CSV files.

### 2 Used Code
Folder structure:

```
scripts/
  ├── full_arm_sweep.py
  ├── lawn_mover_search.py
  ├── optimization.py
  ├── place_object.py
  └── place_task_space.py
```

`scripts/`: This is where your executable Python scripts are placed.

* `full_arm_sweep.py`: Script for performing Full Arm sweep Section 3.8.2.

* `lawn_mover_search.py`: Script for performing End-Effector search Section 3.8.2.

* `optimization.py`: Algorithm for Collision detection and Isolation Sections 3.3 - 3.7.

* `place_object.py`: Script to place a collision object visualization into environment.

* `place_task_space.py`: Script to place a task space visualization into environment.

### 3 Supplementary Code for Evaluation  

This supplementary code, presented in Jupyter notebooks, shows the visualization and evaluation of collision results from both exploration strategies Section 4.2.

* `evaluation_collision_0.06.ipynb` : This notebook visualizes and evaluates collision data when the exploration strategy was performed with a 6 cm diameter.
* `evaluation_collision_0.16.ipynb` : This notebook visualizes and evaluates collision data when the exploration strategy was performed with a 16 cm diameter.


## Prerequisites
All experiments were carried out using the KUKA LBR iiwa 7 R800, a
7-degrees-of-freedom (7-DoF) collaborative robotic arm. The robot was controlled via
a workstation equipped with an AMD Ryzen 5 7535U processor (2.90 GHz base clock
speed) and 16.0 GB of installed RAM. This workstation ran Ubuntu 20.04 LTS and Robot
Operating System (ROS), which communicated with the KUKA Sunrise cabinet.
### Software
- Python 3.x
- ROS
- Robotics Toolbox for Python

### Robot Model
-  KUKA LBR iiwa 7 R800, 7-degrees-of-freedom (7-DoF) collaborative robotic arm

# Errata
### Equation Formatting Correction:

In Equation (3.6), the force components `F_x`, `F_y`, and `F_z` were incorrectly bolded. These are scalar components and should appear in normal font.

**Corrected Equation (3.6):**

$$
\mathbf{F}_{0} = 
\begin{bmatrix}
F_{x} & F_{y} & F_{z}
\end{bmatrix}^{\intercal}
$$
