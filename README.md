# 🧬 Breast Cancer ML Pipeline with MLRun

A containerized CI/CD Machine Learning pipeline for the Breast Cancer dataset, powered by MLRun. This project demonstrates data ingestion, model training, hyperparameter tuning, and model deployment using Kubernetes and Nuclio.

## 📌 Overview

This repository showcases how to automate and orchestrate ML workflows using MLRun, enabling retraining and deployment pipelines for a Random Forest classifier on the Breast Cancer dataset.

## 📚 Dataset

The Breast Cancer dataset is loaded using:
```python
from sklearn.datasets import load_breast_cancer
```
It contains features computed from digitized images of breast masses and is commonly used for binary classification tasks.

## ⚙️ Installation Guide

### ✅ Prerequisites
- Python 3.9
- Docker Desktop (with Kubernetes enabled)
- Helm
- Kubernetes Cluster (local or remote)

### 🛠️ Setup Instructions

1. **Enable Kubernetes in Docker Desktop**
   - Ensure Kubernetes is enabled via Docker Desktop settings.

2. **Install Helm**
   - Official guide: [Install Helm](https://helm.sh/docs/intro/install/)

3. **Create a Kubernetes namespace**
   ```bash
   kubectl create namespace mlrun
   ```

4. **Add the MLRun CE Helm repository**
   ```bash
   helm repo add mlrun-ce https://mlrun.github.io/ce
   helm repo update
   ```

5. **Create Docker Registry Secret**
   ```bash
   kubectl --namespace mlrun create secret docker-registry registry-credentials \
     --docker-server=https://index.docker.io/v1/ \
     --docker-username=your_docker_username \
     --docker-password=your_docker_password \
     --docker-email=your_email
   ```

6. **Deploy MLRun CE**
   ```bash
   helm --namespace mlrun install mlrun-ce --wait --timeout 1800s \
     --set global.registry.url=index.docker.io/your_docker_username \
     --set global.registry.secretName=registry-credentials \
     --set kube-prometheus-stack.enabled=false \
     mlrun-ce/mlrun-ce
   ```

7. **Access MLRun UI**
   ```
   http://localhost:30040
   ```

## 🧪 Pipeline Components

### 📁 data_prep.py
- Loads and structures the Breast Cancer dataset into a pandas DataFrame.
- Saves the dataset as an MLRun artifact for downstream pipeline stages.

### 📁 trainer.py
- Splits the dataset into training and testing sets (90% train, 10% test).
- Trains a RandomForestClassifier.
- Wraps the training logic with apply_mlrun() for automatic experiment tracking.

### 📁 serving.py
- Implements a custom model class inheriting from mlrun.serving.V2ModelServer.
- Defines:
  - load() for model initialization.
  - predict() for real-time inference.
- Prepares the model for serving via Nuclio.

### 📁 workflow.py
Defines the entire ML pipeline using MLRun's @dsl.pipeline decorator.

#### 🔹 Step 1: Data Ingestion
- Loads the Breast Cancer dataset.
- Logs it as an artifact using MLRun.

#### 🔹 Step 2: Model Training
- Explores hyperparameter combinations:
  - n_estimators
  - max_depth
- Selects the best model based on accuracy using selector.

#### 🔹 Step 3: Model Deployment
- Deploys the selected model to a Nuclio function using:
  ```python
  mlrun.deploy_function(...)
  ```

#### 🔹 Step 4: Workflow Execution
- Executes the full pipeline with:
  ```python
  project.run()
  ```

## 🖼️ Artifacts (Expected Output)
(Note: You can add screenshots of these outputs to the repo.)
- 📊 DataFrame preview from data_prep.py
- 📉 Confusion matrix from trainer.py
- 🌟 Feature importance chart
- 🧩 Workflow DAG graph

## 🐳 Docker Image
This pipeline is containerized and can be pulled from Docker Hub:
```bash
hsinghal11/processor-breast-cancer-dataset-jovyan-serving
```
This image includes:
- Pre-installed MLRun and Jupyter
- All necessary project scripts and dependencies

## ✅ Code Quality
- Modular and readable Python code
- Clean structure following best practices
- Fully compatible with MLRun UI for visual tracking

## 🤝 Contributing
Contributions are welcome! Please fork the repo, make improvements, and open a pull request.

## 📄 License
MIT License

## 👨‍💻 Maintainer
Harshit Singhal
DockerHub: hsinghal11
