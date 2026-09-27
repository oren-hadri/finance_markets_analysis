
This project will investigate finance markets relationships, dependencies, 
and correlations using main 2 approaches: complex network analysis and machine learning techniques.
The goal is to provide insights into market trends, inform investment strategies, evaluate risks
and maybe even predict market future direction.

Preliminary Steps:
connect to a financial data source and retrieve historical metrics.
Ingest the data into a suitable format for analysis (raw price -> normalized returns).

Complex Networks:
build a financial network based on correlations between assets.
analyze the network's topology, including degree distribution, clustering coefficient, and path lengths.
calculate centrality measures (degree, betweenness, closeness) to identify influential assets.
detect communities within the network using algorithms like Louvain or Girvan-Newman.
detect risk contagion patterns by simulating shocks to the network and observing how they propagate.

Advanced Machine Learning:
apply Graph Neural Networks (GNNs) to the financial network to predict future returns or risks.
Apply techniques like Graph Convolutional Networks (GCNs) or Graph Attention Networks (GATs) 
to leverage the graph structure for improved predictions.
Apply knowledge from previous complex network analysis to inform the design of the GNN models, 
such as incorporating centrality measures or community structures as features.
compare the performance of GNN models with traditional machine learning models (e.g., Random Forest, XGBoost)
compare different methods - using complex network analysis, using macroeconomic indicators
- to evaluate their predictive power and robustness.

Code:
    python env 3.13.9
Libraries:
    - pandas
    - numpy
    - networkx
    - scikit-learn
    - torch
    - torch-geometric
    - matplotlib
    - seaborn
    - jupyter

Project structure:
- data/: contains raw and processed financial data.
- notebooks/: contains Jupyter notebooks for data exploration, analysis, and visualization.
- src/: contains Python scripts for data processing, network analysis, and machine learning models.
  - network_analysis/: contains scripts for building and analyzing financial networks.
    - graph_construction.py: script for constructing the financial network based on asset correlations.
    - network_metrics.py: script for calculating network metrics and centrality measures.
    - community_detection.py: script for detecting communities within the financial network.
    - risk_contagion.py: script for simulating shocks and analyzing risk contagion patterns.
    - visualization.py: script for visualizing the financial network and its properties.
  - machine_learning/: contains scripts for implementing and training machine learning models, including GNNs.
    - gnn_models.py: script for defining and training Graph Neural Network models.
    - traditional_models.py: script for implementing traditional machine learning models (e.g., Random Forest, XGBoost).
    - evaluation.py: script for evaluating model performance using metrics like accuracy, precision, recall, and F1-score.
  - utils/: contains utility functions for data preprocessing, visualization, and evaluation.
  - data_downloader/: contains scripts for downloading and ingesting financial data.
- tests/: contains unit tests for the codebase.
- README.md: this file, providing an overview of the project and its structure.
- requirements.txt: lists the required Python packages and their versions for the project.
- setup.py: script for setting up the project environment and dependencies.
- .gitignore: specifies files and directories to be ignored by Git version control.
- LICENSE: contains the license information for the project.
- docs/: contains documentation for the project, including methodology, results, and references.
- results/: contains output files, visualizations, and reports generated from the analysis.
- config/: contains configuration files for the project, such as settings for data sources, model parameters, and analysis options.
- scripts/: contains utility scripts for automating tasks, such as data retrieval, preprocessing, and model evaluation.
- logs/: contains log files generated during the execution of scripts and models, useful for debugging and tracking progress.
