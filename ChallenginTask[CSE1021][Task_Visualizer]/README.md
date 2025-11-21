# Task Progress Visualizer 📊

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.28.0-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

*A simple yet powerful task management application for tracking progress and visualizing productivity*

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Screenshots](#screenshots)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

**Task Progress Visualizer** is a web-based application designed to help individuals and teams manage tasks efficiently. Built with Python and Streamlit, it provides an intuitive interface for organizing tasks, tracking completion rates, and visualizing productivity patterns through interactive charts.

### Problem Statement

In today's fast-paced environment, people struggle to:
- Keep track of multiple tasks across different categories
- Monitor progress toward goals
- Identify productivity patterns
- Manage deadlines effectively

### Solution

This application provides:
- ✅ Simple task creation and management
- 📊 Real-time progress tracking with visual analytics
- 🎨 Interactive charts for insights
- 📥 Data export capabilities
- 🔍 Smart filtering and organization

---

## ✨ Features

### Core Functionality

#### 1. Task Management Module
- **Create Tasks** - Add tasks with title, description, category, priority, and due date
- **Update Status** - Mark tasks as Pending, In Progress, or Completed
- **Delete Tasks** - Remove unwanted tasks
- **Filter Tasks** - Filter by status, category, and priority
- **Deadline Tracking** - Monitor overdue and upcoming tasks

#### 2. Visualization Module
- **Status Distribution** - Pie chart showing task completion breakdown
- **Category Analysis** - Bar chart displaying tasks by category
- **Priority Overview** - Bar chart showing task priority distribution
- **Completion Trend** - Line chart tracking tasks completed over time

#### 3. Report Generation Module
- **CSV Export** - Download tasks in CSV format for Excel
- **JSON Export** - Export data in JSON format for integration
- **Text Reports** - Generate summary reports with statistics

### Additional Features
- 📈 Real-time dashboard with key metrics
- 🎯 Progress bar showing overall completion rate
- ⚠️ Overdue task indicators
- 🔄 Automatic data persistence
- 📱 Responsive design

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Core programming language |
| **Streamlit** | 1.28.0 | Web framework for UI |
| **SQLite** | 3.x | Database for data persistence |
| **Pandas** | 2.0.3 | Data manipulation and analysis |
| **Matplotlib** | 3.7.2 | Data visualization |
| **Seaborn** | 0.12.2 | Enhanced visualizations |
| **NumPy** | 1.24.3 | Numerical operations |


---

## 💻 Installation

### Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/)
- **pip** (Python package manager) - Usually comes with Python
- **Git** (optional) - For cloning the repository

### Step-by-Step Installation

#### Method 1: Clone from GitHub

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/task-progress-visualizer.git

# 2. Navigate to project directory
cd task-progress-visualizer

# 3. Create virtual environment
python -m venv venv

# 4. Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run the application
streamlit run app.py

After installation, you should see:

text

You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
Open your browser and navigate to http://localhost:8501

)
📂 Project Structure
task-progress-visualizer/
│
├── 📄 app.py                    # Main application (Streamlit UI)
├── 📄 database.py               # Database operations (CRUD)
├── 📄 visualizations.py         # Chart generation (Matplotlib)
├── 📄 reports.py                # Export functionality
├── 📄 config.py                 # Configuration settings
├── 📄 utils.py                  # Helper functions
├── 📄 requirements.txt          # Python dependencies
│
├── 📄 README.md                 # Project documentation (this file)
├── 📄 statement.md              # Problem statement
│
├── 📁 docs/                     # Documentation folder
│   ├── diagrams/
│   │   ├── usecase.png
│   │   ├── workflow.png
│   │   ├── sequence.png
│   │   ├── class.png
│   │   └── er_diagram.png
│   └── screenshots/
│       ├── dashboard.png
│       ├── add_task.png
│       └── analytics.png
│
├── 📁 venv/                     # Virtual environment (ignored in git)
└── 🗄️ tasks.db                  # SQLite database (auto-created)
---

## 🏗️ System Architecture
