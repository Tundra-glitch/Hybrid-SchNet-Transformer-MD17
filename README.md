Hybrid SchNet + Multi-Head Transformer for Molecular Energy and Force Prediction
A PyTorch implementation of a hybrid Graph Neural Network and Transformer architecture for predicting molecular energies and atomic forces on the Revised MD17 dataset.​
￼
Project Overview
This project develops a hybrid deep learning model that combines the local geometric representation power of SchNet with the global context modeling capability of a Multi-Head Transformer.​
The model is trained on the Revised MD17 molecular dynamics dataset to predict:​
• Molecular potential energy​
• Atomic forces​
The ultimate goal is to build an accurate neural network potential suitable for molecular simulations and AI-driven computational chemistry.​
￼
Motivation
Accurate prediction of molecular energies and forces is fundamental to:​
• Molecular dynamics (MD)​
• Density Functional Theory (DFT) acceleration​
• Potential Energy Surface (PES) approximation​
• Materials discovery​
• Drug discovery​
• AI for Science​
Graph Neural Networks such as SchNet have demonstrated excellent performance for learning molecular representations from atomic structures.​
Transformers have shown remarkable success in modeling long-range dependencies in many scientific domains.​
This project investigates whether combining these two approaches can improve molecular property prediction while maintaining a clean, modular, and reproducible implementation.​
￼
Model Architecture
Atomic Numbers + 3D Coordinates
                │
                ▼
        Molecular Graph
                │
                ▼
          SchNet Encoder
                │
                ▼
    Multi-Head Transformer
                │
                ▼
        Global Pooling
                │
                ▼
        Regression Head
                │
         ┌──────┴──────┐
         ▼             ▼
     Energy        Atomic Forces
Atomic forces are obtained through automatic differentiation of the predicted energy with respect to atomic coordinates.​
￼
Features
• Hybrid SchNet + Transformer architecture​
• Multi-head self-attention​
• Energy prediction​
• Force prediction using automatic differentiation​
• Modular PyTorch implementation​
• Reproducible training pipeline​
• Easy-to-read, well-documented code​
• Professional project structure​
• Visualization tools for model evaluation​
￼
Dataset
Revised MD17​
The Revised MD17 dataset contains high-accuracy quantum chemistry molecular dynamics trajectories generated using Density Functional Theory (DFT).​
Each sample contains:​
• Atomic numbers​
• 3D Cartesian coordinates​
• Molecular energy​
• Atomic forces​
￼
Technologies
• Python 3.11​
• PyTorch 2.2+​
• PyTorch Geometric​
• NumPy​
• pandas​
• matplotlib​
• tqdm​
￼
Repository Structure
Hybrid-SchNet-Transformer-MD17/

README.md
requirements.txt
environment.yml
.gitignore
LICENSE

config.py
dataset.py
model.py
transformer.py
loss.py
trainer.py
evaluate.py
predict.py
visualize.py
utils.py
train.py
test.py

notebook.ipynb
￼
Training Pipeline
Load Dataset
      │
      ▼
DataLoader
      │
      ▼
Hybrid Model
      │
      ▼
Energy Prediction
      │
      ▼
Automatic Differentiation
      │
      ▼
Force Prediction
      │
      ▼
Loss Computation
      │
      ▼
Backpropagation
      │
      ▼
Model Update
￼
Planned Experiments
• Baseline SchNet​
• Hybrid SchNet + Transformer​
• Energy prediction benchmark​
• Force prediction benchmark​
• Attention visualization​
• Model comparison​
• Ablation studies​
￼
Future Improvements
Potential future extensions include:​
• Equivariant Graph Neural Networks​
• PaiNN​
• NequIP-inspired architectures​
• E(3)-equivariant Transformers​
• Diffusion models for molecular generation​
• Large-scale molecular foundation models​
￼
Learning Objectives
This repository is designed to demonstrate practical experience in:​
• Graph Neural Networks​
• Molecular Machine Learning​
• Scientific Deep Learning​
• Transformer Architectures​
• Neural Network Potentials​
• AI for Computational Chemistry​
• Reproducible Machine Learning Research​
￼
Project Status
🚧 In Development​
This repository is being built from the ground up with an emphasis on clean software engineering, modular design, scientific correctness, and reproducibility.​
Development follows an incremental workflow:​
1. Project architecture​
2. Dataset implementation​
3. Baseline SchNet​
4. Transformer integration​
5. Energy and force prediction​
6. Evaluation and visualization​
7. Documentation and benchmarking​
￼
License
This project is released under the MIT License.
