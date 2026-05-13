# Machine Learning Assignment 2: Regularization

This project demonstrates two powerful regularization techniques to combat overfitting in Machine Learning models: **Sparse Representation (L1 Regularization)** and **Multi-Task Learning (MTL)**.

The project features a complete Python machine learning pipeline that generates an interactive, aesthetic Web Dashboard to showcase the results, analysis, and educational materials.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python3 -m pip install -r requirements.txt
```

### 2. Generate the Models and Visualizations
First, run the main orchestrator. This script will prepare the dataset, intentionally overfit a baseline model, apply both regularization techniques, and generate all necessary visualization charts into the `dashboard/assets/` directory.
```bash
python3 main.py
```

### 3. Launch the Interactive Dashboard
Run the server script to start a local web server and automatically open the interactive dashboard in your browser.
```bash
python3 run_server.py
```

---

## 📁 Active Project Structure

This assignment has been heavily refactored into a modular, production-ready pipeline:

```text
Sparse_Representation_Assignment/
├── main.py                       # 🧠 Main Pipeline: Trains models and generates assets
├── sparse_model.py               # 🎯 Sparse Representation (L1) implementation
├── multitask_model.py            # 🔀 Multi-Task Learning implementation
├── visualizations.py             # 📊 Chart generation logic
├── run_server.py                 # 🌐 Web server to host the dashboard
├── dashboard.html                # 🎨 The Interactive Web Dashboard (Frontend)
├── dashboard/assets/             # 🖼️ Generated charts (created by main.py)
├── peer_teaching_material.md     # 📚 Raw educational content
├── requirements.txt              # 📦 Python dependencies
└── README.md                     # 📖 This file
```

---

## 🎯 Assignment Requirements Coverage

| Requirement | Implementation | Status |
|-------------|---------|---------|
| **1. Peer Teaching Material** | Available in the **Dashboard's "Peer Teaching Material" tab** and `peer_teaching_material.md` | ✅ Complete |
| **2. Overfitting Implementation** | Handled in `main.py` by injecting 200 noise features into a small dataset. | ✅ Complete |
| **3. Apply Assigned Technique** | Demonstrated in `sparse_model.py` and `multitask_model.py`. | ✅ Complete |
| **4. Comparison & Interactive Analysis** | Presented visually in the **Dashboard's "Results & Analysis" tab** with interactive prediction tools. | ✅ Complete |

---

## 🌟 Key Features of the Dashboard

- **Interactive Prediction Demos:** Users can adjust input features via sliders and watch the model dynamically predict outcomes in real-time, demonstrating how Sparse Representation zeros out noise and how MTL shares representations.
- **Implementation Models:** Displays the exact Scikit-Learn Python code used to build the architectures.
- **Rich Visualizations:** Automatically generated comprehensive charts comparing baseline vs. regularized models.
- **Educational Guides:** Beautifully formatted explanations of the Bias-Variance tradeoff and the mechanics of both regularization techniques.

---

**Group Members**: [Etsubdink Zebre], [Betel Nigusu]  
**Assigned Techniques**: Sparse Representation & Multi-Task Learning  
**Course**: Machine Learning - Assignment 2
