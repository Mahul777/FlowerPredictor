
---

# 🌸 FlowerPredictor: MLOps Major Project

Welcome to **MLOps Project 6**. This repository demonstrates a production-grade machine learning pipeline. The core focus of this project is mastering **CircleCI** for cloud-native automation, replacing traditional tools like Jenkins to deploy a containerized application to **Google Kubernetes Engine (GKE)**.

## 📖 Introduction

This project marks a major milestone in our MLOps journey. While previous projects focused on complex data engineering (like the 200,000-row Australia Weather dataset), this project uses a streamlined dataset to prioritize **Deployment Infrastructure**.

The goal is to move from "learning to drive in a parking lot" (local testing) to "driving a massive truck on the highway" (scalable cloud deployment).

---

## 🛠️ The Technology Stack

| Tool | Icon | Purpose |
| --- | --- | --- |
| **CircleCI** | ⭕ | Cloud-hosted automation for building and testing code on every push. |
| **GitLab CI** | 🦊 | Manages the step-by-step "pipeline" journey from laptop to server. |
| **GitHub Actions** | 🐙 | Automates tasks like unit testing directly within the repository. |
| **Docker** | 🐳 | Containerizes the application for consistent environments. |
| **GCP (GKE)** | ☁️ | Orchestrates the deployment using Google Kubernetes Engine. |

---

## 🏗️ Project Workflow

The project follows a modular, step-by-step professional engineering cycle:

1. **Project Setup:** Environment configuration () and directory structuring (`src`, `pipelines`, `artifacts`).
2. **Jupyter Testing:** Prototyping data processing and model logic in blocks. 
3. **Modularity:** Refactoring code into **Classes and Objects** for production readiness.
4. **UI Development:** Building a user-friendly interface using **Flask and HTML**.
5. **GCP Infrastructure:** * Setting up **GKE Clusters**.
* Configuring **Artifact Registry (GCR)**.
* Managing IAM service account JSON keys.


6. **CI/CD Pipeline:** Automating the flow from GitHub → CircleCI → GCR → GKE.

---

## 🔄 How CircleCI Works (The Connection)

The moment you push code to the `main` branch, the following automated sequence triggers:

1. **Trigger:** CircleCI detects the update via GitHub SCM.
2. **Checkout:** CircleCI pulls the latest code into a cloud container.
3. **Build & Push:** A Docker image is created and pushed to **Google Container Registry (GCR)**.
4. **Deploy:** CircleCI communicates with **GKE** to roll out the new image to the Kubernetes cluster.

---

## 🥊 Comparison: CircleCI vs. Jenkins

Why choose CircleCI for this project?

| Feature | ⭕ CircleCI | 🦊 Jenkins |
| --- | --- | --- |
| **Hosting** | Cloud-Based (SaaS) | Local/Self-Managed |
| **Setup** | Fast (Mostly UI clicks) | Complex (Manual installation) |
| **Scalability** | Native & Automated | Manual Scaling |
| **Customization** | Standardized | Infinite (3,000+ plugins) |
| **Typical User** | Startups (Spotify, Pinterest) | Big Tech (Netflix, Uber) |

---

## 🚀 Getting Started

### Prerequisites

* Google Cloud Platform Account
* CircleCI Account linked to GitHub
* Python 3.8+

### Installation

1. **Clone the repo:**
```bash
git clone https://github.com/Mahul777/FlowerPredictor.git

```


2. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run Locally:**
```bash
python app.py

```



---

## 📜 License

Distributed under the MIT License.

## 📧 Contact

**Mahul** - [Your GitHub Profile](https://www.google.com/search?q=https://github.com/Mahul777)

Project Link: [https://github.com/Mahul777/FlowerPredictor](https://www.google.com/search?q=https://github.com/Mahul777/FlowerPredictor)

---

Would you like me to help you write the specific `config.yml` file for CircleCI to match this workflow?

```
## Pipeline
![alt text](image.png)

## UI 
![alt text](image-1.png)

## Clustering
![alt text](image-2.png)

```
