from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities

PROJECT_DIR = Path(__file__).resolve().parents[2]
GRAPH_DIR = PROJECT_DIR / "data/graph"
VIZ_DIR = PROJECT_DIR / "data/graph/visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def visualize_financial_networks():
  edge_files = sorted(list(GRAPH_DIR.glob("edges_window_*.csv")))

  if not edge_files:
    print("No edge files found. Please generate the graph data first.")
    return

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

    # Detect communities using NetworkX built-in Louvain algorithm
    communities = louvain_communities(G, weight="weight")
    node_to_community = {}
    for comm_id, comm in enumerate(communities):
      for node in comm:
        node_to_community[node] = comm_id

    # Set up layout and figure
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, weight="weight")

    # Map colors to communities
    node_colors = [node_to_community.get(node, 0) for node in G.nodes()]
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]

    # Draw components
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=600,
        node_color=node_colors,
        cmap=plt.cm.tab10,
        alpha=0.85,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.3,
        width=[w * 2 for w in edge_weights],
        edge_color="gray",
    )
    nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif")

    plt.title(
        f"Financial Correlation Network - {window_date}\nCommunities:"
        f" {len(communities)} | Edges: {len(G.edges)}",
        fontsize=14,
    )
    plt.axis("off")

    # Save visualization
    output_file = VIZ_DIR / f"network_viz_{window_date}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

  print(f"All network visualizations saved successfully in: {VIZ_DIR}")


if __name__ == "__main__":
  visualize_financial_networks()