# Project Summary: Financial Correlation Networks & Macro Dynamics Analysis

## 1. Core Code and Script Files Developed

* **`generate_global_metrics.py`**: Iterates through rolling-window edge list files, computes network topology metrics for each timeframe, and saves the output to `global_network_metrics.csv`.
* **Comprehensive Visualization Script**: Merges the global network metrics with normalized daily asset returns to generate a multi-panel visual timeline from late 2022 through 2026.


* **`visualize_financial_networks()`**: Loops through all time windows, detects communities using the Louvain algorithm, and exports structural network graphs (saved as `network_viz_*.png` / `.jpg`).

---

## 2. Generated Outputs and Data Files

* **Macro Network Dataset (`global_network_metrics.csv`)**: Consolidates node counts, edge counts (`Edges_Count`), network density (`Density`), modularity (`Modularity`), community counts (`Num_Communities`), and clustering coefficients (`Clustering_Coefficient`) across all time windows.
* **Comprehensive Multi-Panel Plot (`comprehensive_global_metrics_2.jpg`)**: A 5-panel time-series visualization tracking:


* Edge count evolution.


* Network density.


* Modularity and community count dynamics.


* Clustering coefficients.


* Normalized cumulative performance comparison between Bitcoin (BTC) and the S&P 500 (SPX).




* **Individual Network Graphs**: Visual snapshots for each temporal window highlighting asset clustering and community structures.

---

## 3. Key Insights and Conclusions

* **Market Regime Shifts:** The network metrics successfully capture structural regime changes, alternating between high-correlation crisis phases (surging density and edge counts with collapsing modularity) and normal, decentralized sectoral regimes.


* **Crypto vs. Traditional Equities:** While the S&P 500 exhibits smooth, long-term upward growth, Bitcoin displays sharp volatility that tightly correlates with spikes in network density—demonstrating that major crypto rallies coincide with broad systemic market coupling.


* **Predictive Structural Foundation:** Topology metrics (such as density and modularity shifts) offer robust macro-indicators of market stress, serving as an ideal feature store for future Machine Learning and Graph Neural Network (GNN) integration.