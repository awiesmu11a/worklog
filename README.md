# ECE299 worklog

The goal is to integrate the [IPC (Incremental Potential Contact)](https://ipc-sim.github.io/) model with existing [PBD (Position Based Dynamics)](https://github.com/ucsdarclab/ARCParticleSim) model for simulation of basic surgical task (grabbing etc).

## Fall 2022

Studied relevant papers given by Fei. Started studying the paper on IPC and PBD along with going through and playing around with the existing scripts. I was completely new to using C++ and working on designing the simulation for the first this process took most of the time.

## Spring 2023

Integrated a brute force algorithm of IPC in along the PBD script as shown below:
```cpp
\\ Initialized the wound and a cylinder as a tool
\\ Loop {
\\ Run PBD (existing code) to estimate the next pose
\\ Run the IPC implementation written
void PBDSolver::IPC() {

    //Energy evaluate

    Eigen::VectorXd temp_pose(vertexList.size() * 3);
    Eigen::VectorXd temp_pred(vertexList.size() * 3);
    Eigen::VectorXd energy(1);
    Eigen::VectorXd temp_energy(1);
    energy(0) = 0;
    for (int i = 0; i < vertexList.size(); ++i) {
        auto& vertex = vertexList[i];
        temp_pose(3 * i) = vertex->position(0);
        temp_pose((3 * i) + 1) = vertex->position(1);
        temp_pose((3 * i) + 2) = vertex->position(2);
        Eigen::VectorXd temp = vertex->position + (timeStep * vertex->velocity) + (timeStep * timeStep * vertex->force);
        temp_pred(3 * i) = temp(0);
        temp_pred((3 * i) + 1) = temp(1);
        temp_pred((3 * i) + 2) = temp(2);
    }
    energy += ((temp_pose - temp_pred).transpose() * (temp_pose - temp_pred));
    energy *= 0.5;
    double energyVal = energy(0);
    double temp_energyVal = energy(0);
    double barrier = 0;
    double threshold_dist = 0.01;
    double kappa = 1;
    
    for (int i = 0; i < (tetMeshList.size() - 1); ++i) {
        auto& tool = tetMeshList[i];
        auto& wound = tetMeshList[i + 1];
        double temp_dist;
        for (int j = 0; j < tool->V.rows(); ++j) {
            for (int k = 0; k < wound->V.rows(); ++k) {
                double diff_x = temp_pose((3 * j)) - temp_pose((3 * k) + wound->offset);
                double diff_y = temp_pose((3 * j) + 1) - temp_pose((3 * k) + 1 + wound->offset);
                double diff_z = temp_pose((3 * j) + 2) - temp_pose((3 * k) + 2 + wound->offset);
                temp_dist = std::sqrt((diff_x * diff_x) + (diff_y * diff_y) + (diff_z * diff_z));
                double temp = 0;
                if (temp_dist < threshold_dist) {
                    temp = std::log(temp_dist / threshold_dist);
                    temp *= (temp_dist - threshold_dist);
                    temp *= (temp_dist - threshold_dist);
                    temp *= (-1);
                }
                barrier += temp;
            }
        }        
    }
    barrier *= kappa;
    energyVal += barrier;
    spdlog::info("Energy at each timeStep is {}", energyVal);

    //Update step

    double alpha;
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Identity((vertexList.size() * 3), (vertexList.size() * 3));
    hessian += hessian;
    Eigen::MatrixXd barrier_hessian = Eigen::MatrixXd::Zero((vertexList.size() * 3), (vertexList.size() * 3));
    for (int i = 0; i < (tetMeshList.size() - 1); ++i) {
        auto& tool = tetMeshList[i];
        auto& wound = tetMeshList[i + 1];
        double temp_dist;
        for (int j = 0; j < tool->V.rows(); ++j) {
            for (int k = 0; k < wound->V.rows(); ++k) {
                double diff_x = temp_pose((3 * j)) - temp_pose((3 * k) + wound->offset);
                double diff_y = temp_pose((3 * j) + 1) - temp_pose((3 * k) + 1 + wound->offset);
                double diff_z = temp_pose((3 * j) + 2) - temp_pose((3 * k) + 2 + wound->offset);
                temp_dist = std::sqrt((diff_x * diff_x) + (diff_y * diff_y) + (diff_z * diff_z));

                Eigen::MatrixXd d_hessian = Eigen::MatrixXd::Zero((vertexList.size() * 3), (vertexList.size() * 3));
                Eigen::VectorXd d_grad = Eigen::VectorXd::Zero(vertexList.size() * 3);

                double temp;
                
                if (temp_dist < threshold_dist) {
                    spdlog::info("Distance between points is {}", temp_dist);
                    double temp_frac = (temp_dist / threshold_dist);
                    double temp_diff = (temp_dist - threshold_dist);
                    double temp_squared = (temp_diff) * (temp_diff);
                    
                    double bd_hessian = (2 * std::log(temp_frac));
                    bd_hessian += ((4 * (temp_diff / temp_dist)) - (temp_squared / (temp_dist * temp_dist)));
                    bd_hessian *= -1; 
                    
                    double bd_grad = (2 * temp_diff * std::log(temp_frac));
                    bd_grad += (temp_squared / temp_dist);
                    bd_grad *= -1;

                    for (int n = 0; n < 3; ++n) {
                        d_grad((3 * j) + n) = ((temp_pose((3 * j) + n)) / temp_dist);
                        d_hessian.row((3 * j) + n)[(3 * j) + n] = (1 / temp_dist);
                        d_grad((3 * j) + k) = ((temp_pose((3 * j) + n)) / temp_dist);
                        d_hessian.row((3 * k) + n)[(3 * k) + n] = (1 / temp_dist);
                    }
                    barrier_hessian += (bd_hessian * (d_grad * d_grad.transpose()));
                    barrier_hessian += (bd_grad * d_hessian);
                }
            }
        }          
    }
    barrier_hessian *= kappa;
    hessian += barrier_hessian;
    
    Eigen::MatrixXd gradient = hessian * temp_pose;
    Eigen::MatrixXd p = hessian.completeOrthogonalDecomposition().pseudoInverse();
    p *= gradient;
    alpha = 1;
    temp_pred = temp_pose + (alpha * p);     
    
    Eigen::VectorXd temp_center(3);
    for (int i = 0; i < tetMeshList[0]->V.rows(); ++i) {
        Eigen::VectorXd temp(3);
        temp(0) = temp_pred((3 * i) + 0);
        temp(1) = temp_pred((3 * i) + 1);
        temp(2) = temp_pred((3 * i) + 2);
        temp_center += temp;
    }
    temp_center /= vertexList.size();
    std::cout<<temp_center<<std::endl;
}
```

Above implementaion compiled but did not give results on simulation.
Probable errors might:
- High runtime as the algorithm implemented was of complexity O(n^3).
- Problem might be occuring while calculating the inverse of the Hessain i.e. Hessian matrix might be singular matrix.
One approach to debug would be to run the IPC algorithm when objects are about collide.

Latest update: Was trying to import a pre-existing rendered tool to the simulation
Problem: Did not find a tool which has appropriate mesh size to import and run the simulation.

### Latest
Now I decided to start from basic problem and see if above implementation is correct or not. Will implement the IPC on simpler objects like a plane and point. 

### Current
Debugging above script
