from pathlib import Path

import networkx as nx
import pandas as pd
import scipy
from matplotlib import pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data/normalized"


def generate_graph_date():

  data_file = DEFAULT_OUTPUT_DIR / "merged_returns.csv"

  # Load the generated log returns dataset
  returns_df = pd.read_csv(data_file, index_col=0, parse_dates=True)

  window_size = 60
  step_size = 60  # You can change step size (e.g., every 20 days or every 1 day)

  for start_idx in range(0, len(returns_df) - window_size + 1, step_size):
    end_idx = start_idx + window_size

    # Slice the rolling window
    window_returns = returns_df.iloc[start_idx:end_idx]

    # Get the date label for saving files (using the last date in the window)
    window_date = window_returns.index[-1].strftime("%Y-%m-%d")

    # Compute correlation matrix for this window
    corr_matrix = window_returns.corr()

    # Build graph
    threshold = 0.4
    G = nx.Graph()
    for i in range(len(corr_matrix.columns)):
      for j in range(i + 1, len(corr_matrix.columns)):
        weight = corr_matrix.iloc[i, j]
        if weight >= threshold:
          G.add_edge(
            corr_matrix.columns[i], corr_matrix.columns[j], weight=weight
          )

    # Export edges for this specific window
    if len(G.edges) > 0:
      edges_df = nx.to_pandas_edgelist(G)
      edges_df.to_csv(
        PROJECT_DIR / "data/graph" / f"edges_window_{window_date}.csv",
        index=False,
      )

  # Create the nodes table with basic info (e.g., asset identifier)
  #nodes_df = pd.DataFrame({"id": list(G.nodes())})
  #nodes_df.to_csv("cytoscape_nodes.csv", index=False)

  print("Graph data generated successfully!")

def analyze_graph(G):
  # Assume we have monthly returns or we slice the data into monthly windows
  # For example, iterating over monthly edge files or running over rolling windows:

  timeseries_metrics = []

  # If you have multiple monthly edge files or want to loop through rolling windows:
  # Let's assume we loop through files saved for each month:
  for file_path in sorted(
          (PROJECT_DIR / "data/graph").glob("edges_window_*.csv")
  ):
    month_name = file_path.stem.replace("edges_window_", "")
    edges_df = pd.read_csv(file_path)

    # Build the graph for this specific month
    G = nx.from_pandas_edgelist(
      edges_df, source="source", target="target", edge_attr="weight"
    )

    # Skip empty graphs
    if len(G.nodes) == 0:
      continue

    # Calculate metrics for this specific month
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G, weight="weight")
    pagerank_cent = nx.pagerank(G, weight="weight")

    # Store metrics per asset, including the month/time column
    for node in G.nodes():
      timeseries_metrics.append({
        "Month": month_name,
        "Asset": node,
        "Degree_Centrality": degree_cent[node],
        "Betweenness_Centrality": betweenness_cent[node],
        "PageRank": pagerank_cent[node],
      })

  # Combine everything into a grand time-series DataFrame
  ts_metrics_df = pd.DataFrame(timeseries_metrics)

  # Save the timeline metrics (ideal for tracking how asset centrality changes over 4 years!)
  ts_metrics_df.to_csv(
    PROJECT_DIR / "data/graph" / "nodes_centrality_timeseries.csv", index=False
  )
  print(
    "Time-series node metrics calculated and saved successfully for all"
    " months!"
  )

def generate_global_metrics():
  edge_files = sorted(list((PROJECT_DIR/ "data/graph").glob("edges_window_*.csv")))

  if not edge_files:
    print("No edge files found. Please generate the graph data first.")
    return

  global_metrics = []

  for file_path in edge_files:
    window_date = file_path.stem.replace("edges_window_", "")
    edges_df = pd.read_csv(file_path)

    if edges_df.empty:
      continue

    # Build graph from edges
    G = nx.from_pandas_edgelist(
        edges_df, source="source", target="target", edge_attr="weight"
    )

    if len(G.nodes) == 0:
      continue

    # 1. Global Network Metrics
    density = nx.density(G)
    clustering = nx.average_clustering(G, weight="weight")

    # Handle shortest path length safely (only if graph or its largest component is connected)
    if nx.is_connected(G):
      path_length = nx.average_shortest_path_length(G, weight="weight")
    else:
      largest_cc = max(nx.connected_components(G), key=len)
      subgraph = G.subgraph(largest_cc)
      path_length = (
          nx.average_shortest_path_length(subgraph, weight="weight")
          if len(subgraph) > 1
          else 0.0
      )

    # 2. Community Detection using NetworkX built-in Greedy Modularity
    try:
      communities = nx.community.greedy_modularity_communities(
          G, weight="weight"
      )
      num_communities = len(communities)
      # Calculate modularity using built-in function
      modularity = nx.community.modularity(
          G, communities, weight="weight"
      )
    except Exception:
      num_communities = 1
      modularity = 0.0

    # Store metrics for this window
    global_metrics.append({
        "Date": window_date,
        "Nodes_Count": len(G.nodes),
        "Edges_Count": len(G.edges),
        "Density": density,
        "Clustering_Coefficient": clustering,
        "Avg_Path_Length": path_length,
        "Modularity": modularity,
        "Num_Communities": num_communities,
    })

  # Save to CSV
  output_file = PROJECT_DIR / "data/graph" / "global_network_metrics.csv"
  pd.DataFrame(global_metrics).to_csv(output_file, index=False)
  print(f"Global network metrics successfully saved to: {output_file}")

def plot_all_global_metrics():
  METRICS_FILE = PROJECT_DIR / "data/graph/global_network_metrics.csv"

  if not METRICS_FILE.exists():
    print(f"Metrics file not found at: {METRICS_FILE}")
    return

  df = pd.read_csv(METRICS_FILE)
  df["Date"] = pd.to_datetime(df["Date"])
  df = df.sort_values("Date")

  # Create a figure with four network metrics and a normalized market comparison.
  fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 17), sharex=True)

  # 1. Edges and Density
  axes[0].plot(
      df["Date"],
      df["Edges_Count"],
      color="tab:blue",
      marker="o",
      label="Edges Count",
  )
  axes[0].set_ylabel("Edges", color="tab:blue")
  axes[0].tick_params(axis="y", labelcolor="tab:blue")
  axes[0].grid(True, linestyle="--", alpha=0.6)
  axes[0].set_title(
      "Financial Correlation Network - Comprehensive Global Metrics"
  )

  # 2. Network Density
  axes[1].plot(
      df["Date"], df["Density"], color="tab:red", marker="s", label="Density"
  )
  axes[1].set_ylabel("Density", color="tab:red")
  axes[1].tick_params(axis="y", labelcolor="tab:red")
  axes[1].grid(True, linestyle="--", alpha=0.6)

  # 3. Modularity and Number of Communities
  ax3_twin = axes[2].twinx()
  axes[2].plot(
      df["Date"],
      df["Modularity"],
      color="tab:green",
      marker="^",
      label="Modularity",
  )
  ax3_twin.plot(
      df["Date"],
      df["Num_Communities"],
      color="tab:purple",
      linestyle="--",
      marker="x",
      label="Communities",
  )
  axes[2].set_ylabel("Modularity", color="tab:green")
  ax3_twin.set_ylabel("Communities Count", color="tab:purple")
  axes[2].tick_params(axis="y", labelcolor="tab:green")
  ax3_twin.tick_params(axis="y", labelcolor="tab:purple")
  axes[2].grid(True, linestyle="--", alpha=0.6)

  # 4. Clustering Coefficient and Path Length
  axes[3].plot(
      df["Date"],
      df["Clustering_Coefficient"],
      color="tab:orange",
      marker="d",
      label="Clustering",
  )
  axes[3].set_xlabel("Date")
  axes[3].set_ylabel("Clustering Coeff", color="tab:orange")
  axes[3].tick_params(axis="y", labelcolor="tab:orange")
  axes[3].grid(True, linestyle="--", alpha=0.6)

  # 5. Bitcoin and S&P 500, each rebased to 100 at its first available price.
  levels_file = PROJECT_DIR / "data/normalized/merged_levels.csv"
  if levels_file.exists():
    levels_df = pd.read_csv(levels_file, parse_dates=["date"]).sort_values("date")
    market_columns = {
        "crypto__BTC": "Bitcoin",
        "market_indices__SPX": "S&P 500",
    }
    for column, label in market_columns.items():
      if column not in levels_df.columns:
        continue
      prices = pd.to_numeric(levels_df[column], errors="coerce")
      first_valid = prices.dropna()
      if first_valid.empty or first_valid.iloc[0] == 0:
        continue
      normalized_prices = prices / first_valid.iloc[0] * 100
      axes[4].plot(levels_df["date"], normalized_prices, label=label)

    if axes[4].lines:
      axes[4].legend()
    else:
      axes[4].text(
          0.5, 0.5, "Bitcoin and S&P 500 price data unavailable",
          ha="center", va="center", transform=axes[4].transAxes,
      )
  else:
    axes[4].text(
        0.5, 0.5, f"Price data file not found: {levels_file}",
        ha="center", va="center", transform=axes[4].transAxes,
    )
  axes[4].set_ylabel("Index (base = 100)")
  axes[4].set_xlabel("Date")
  axes[4].grid(True, linestyle="--", alpha=0.6)
  axes[4].set_title("Bitcoin and S&P 500 (Normalized)")

  plt.tight_layout()

  output_path = (
      PROJECT_DIR / "data/graph/visualizations/comprehensive_global_metrics.png"
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(output_path, dpi=300)
  plt.close()
  print(f"Comprehensive metrics plot saved successfully to: {output_path}")


if __name__ == '__main__':
    generate_graph_date()
    analyze_graph(None)  # Pass None since we are reading from files
    generate_global_metrics()
    plot_all_global_metrics()
