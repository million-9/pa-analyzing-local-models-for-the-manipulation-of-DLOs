# Analyzing Local Models for the Manipulation of Deformable Linear Objects (DLOs)

This repository contains the code and methodology for a project thesis exploring the manipulation of **Deformable Linear Objects (DLOs)** such as cables, using **local linear deformation models** and adaptive control strategies. The system is implemented and tested in the **SOFA simulation framework**, a powerful open-source platform for physical simulations.

## 🧠 Project Objective

The main objective is to investigate the feasibility and performance of **locally estimated Jacobian models** for planning and executing shape control of cables in constrained environments—without requiring prior knowledge of physical parameters.

## 🛠️ Features

- **Local Deformation Model:** Approximates small deformations of a DLO using gripper movement and pseudo-inverse Jacobian estimation.
- **Path Planning:**
  - **Basic algorithm** for contact-free environments.
  - **Advanced algorithm** for complex routing with obstacle avoidance and intermediate goals.
- **Adaptive Control:** Periodic retraining of local models and data regulation based on hyperparameters `n_max` and `t_max`.
- **Simulation Scenarios:**
  - Task A: Cable manipulation without obstacles.
  - Task B: Manipulation with a single contact.
  - Task C: Multi-contact routing with up to 9 intermediate steps.

## 🧪 Implementation

- **Language:** Python
- **Simulation Tool:** [SOFA Framework](https://www.sofa-framework.org/)
- The cable and gripper models are simulated in 3D space with realistic contact interactions and controlled using a data-driven planning loop.

## 📊 Evaluation

- Error reduction is measured using **Mean Squared Error (MSE)** between current and target cable configurations.
- The performance across varying task complexity shows that the local model is accurate and adaptable when retrained regularly.

## 📖 Documentation

The full simulation environment and theoretical background are based on the [SOFA framework](https://www.sofa-framework.org). Refer to their documentation for:
- Installation
- Scene creation
- Plugin usage for deformable objects

## 📄 License

This repository is intended for academic and research purposes only. Please cite appropriately when using this work.

---

**Author**: Mohamed Musthafa Palamadathil Kozhisseri  
**Supervisor**: Georg Rabenstein, M.Sc.  
**Institution**: Chair of Automatic Control, FAU Erlangen-Nürnberg  
**Date**: April 2025

