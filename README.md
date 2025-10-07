# Genetic Algorithm 2D Optimization App
This project is a Streamlit web application for visualizing and experimenting with genetic algorithms (GA) on various 2D mathematical functions. It allows users to configure GA parameters, run optimizations, and visualize the search process in 3D.

## Features

- **Interactive UI:** Configure population size, number of epochs, selection, crossover, and mutation methods.
- **Multiple Functions:** Optimize Rastrigin, Hypersphere, Rosenbrock, and Styblinski-Tang functions.
- **Customizable Operators:** Choose from several crossover and mutation strategies.
- **3D Visualization:** See the GA's progress and best individuals on a 3D surface plot.
- **Report Generation:** Export results as CSV files for further analysis.

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── run.sh                  # Script to launch the app
├── core_algorithm/         # Core GA logic and data types
│   ├── __init__.py
│   ├── data_types.py
│   ├── genetic_algorigm.py
│   └── target_function.py
└── reports/                # Generated CSV reports (created at runtime)
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

1. Clone the repository.
2. Install dependencies:
```sh
pip install -r requirements.txt
```

### Running the App

Start the Streamlit app with:

```sh
bash run.sh
```

The app will open in your browser.

## Usage

1. Set GA parameters in the sidebar.
2. Click **Run** to start the optimization.
3. View the 3D plot and best individual found.
4. Optionally, generate a CSV report of the results.

## Core Modules

- target_function.py: Defines target functions and their global minima.
- genetic_algorigm.py: Implements GA operators (selection, crossover, mutation).
- data_types.py: Contains type definitions and enums for configuration.

## License

This project is for educational and research purposes.

---

**Authors:** Adam Gruszczyński, Tomasz Wójcicki, Wojciech Jakubiec

**Contact:** adam.01748.g@gmail.com, tomasz.wojcick@gmail.com, jakubiecwojtek@gmail.com

---

For more details, see the code in app.py and the core_algorithm directory.
