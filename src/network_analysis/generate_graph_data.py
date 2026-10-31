from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data/normalized"

# Options: 'all', 'crypto_only', 'non_crypto'
GROUPS = ["all", "crypto_only", "non_crypto"]

WINDOW_SIZE = 30
STEP_SIZE = 30
THRESHOLD = 0.4


def graph_dir(group):
    # Each group has its own folder, so results do not overwrite each other
    folder = PROJECT_DIR / "data/graph" / group
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def filter_assets(columns, group):
    is_crypto = columns.str.startswith("crypto_")
    if group == "crypto_only":
        return columns[is_crypto]
    if group == "non_crypto":
        return columns[~is_crypto]
    return columns


def build_graph(corr_matrix, threshold):
    names = list(corr_matrix.columns)
    values = corr_matrix.to_numpy()

    G = nx.Graph()
    # Keep all assets as nodes, even if they have no edges
    G.add_nodes_from(names)

    rows, cols = np.triu_indices(len(names), k=1)
    for i, j in zip(rows, cols):
        corr = values[i, j]
        # A NaN correlation fails this check, so no edge is created
        if abs(corr) >= threshold:
            G.add_edge(
                names[i],
                names[j],
                weight=abs(corr),
                corr=corr,
                # Strong correlation = short distance
                distance=np.sqrt(max(0.0, 2 * (1 - abs(corr)))),
            )
    return G


def generate_graph_date(group, window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                        threshold=THRESHOLD):
    data_file = DEFAULT_OUTPUT_DIR / "merged_returns.csv"
    returns_df = pd.read_csv(data_file, index_col=0, parse_dates=True)

    # Choose the assets first, then compute the correlations
    returns_df = returns_df[filter_assets(returns_df.columns, group)]

    folder = graph_dir(group)

    for start_idx in range(0, len(returns_df) - window_size + 1, step_size):
        window_returns = returns_df.iloc[start_idx:start_idx + window_size]

        # The last date of the window is used as the file name
        window_date = window_returns.index[-1].strftime("%Y-%m-%d")

        corr_matrix = window_returns.corr()
        G = build_graph(corr_matrix, threshold)

        # Save the node list, so assets without edges are not lost
        pd.DataFrame({"id": list(G.nodes())}).to_csv(
            folder / f"nodes_window_{window_date}.csv", index=False
        )

        if G.number_of_edges() > 0:
            nx.to_pandas_edgelist(G).to_csv(
                folder / f"edges_window_{window_date}.csv", index=False
            )

    print(f"[{group}] Graph data generated successfully!")


def load_window_graph(folder, window_date):
    nodes = pd.read_csv(folder / f"nodes_window_{window_date}.csv")["id"]

    edge_file = folder / f"edges_window_{window_date}.csv"
    if edge_file.exists():
        edges_df = pd.read_csv(edge_file)
        G = nx.from_pandas_edgelist(
            edges_df, source="source", target="target",
            edge_attr=["weight", "distance"],
        )
    else:
        G = nx.Graph()

    G.add_nodes_from(nodes)
    return G


def window_dates(folder):
    files = sorted(folder.glob("nodes_window_*.csv"))
    return [f.stem.replace("nodes_window_", "") for f in files]


def analyze_graph(group):
    folder = graph_dir(group)
    rows = []

    for window_date in window_dates(folder):
        G = load_window_graph(folder, window_date)

        if len(G.nodes) == 0:
            continue

        degree_cent = nx.degree_centrality(G)
        # Betweenness uses shortest paths, so it needs distance
        betweenness_cent = nx.betweenness_centrality(G, weight="distance")
        # PageRank needs strength, so it uses weight
        pagerank_cent = nx.pagerank(G, weight="weight")

        for node in G.nodes():
            rows.append({
                "Month": window_date,
                "Asset": node,
                "Degree_Centrality": degree_cent[node],
                "Betweenness_Centrality": betweenness_cent[node],
                "PageRank": pagerank_cent[node],
            })

    pd.DataFrame(rows).to_csv(folder / "nodes_centrality_timeseries.csv", index=False)
    print(f"[{group}] Node metrics saved.")


def generate_global_metrics(group):
    folder = graph_dir(group)
    dates = window_dates(folder)

    if not dates:
        print(f"[{group}] No graph files found. Run generate_graph_date first.")
        return

    global_metrics = []

    for window_date in dates:
        G = load_window_graph(folder, window_date)

        if len(G.nodes) == 0:
            continue

        density = nx.density(G)
        clustering = nx.average_clustering(G, weight="weight")

        # Path length is measured on the largest connected part only
        largest_cc = max(nx.connected_components(G), key=len)
        largest_fraction = len(largest_cc) / len(G.nodes)
        subgraph = G.subgraph(largest_cc)
        if len(subgraph) > 1:
            path_length = nx.average_shortest_path_length(subgraph, weight="distance")
        else:
            path_length = 0.0

        # Communities are found only on nodes that have at least one edge
        connected_nodes = [n for n in G.nodes() if G.degree(n) > 0]
        if len(connected_nodes) > 0:
            core = G.subgraph(connected_nodes)
            communities = nx.community.greedy_modularity_communities(core, weight="weight")
            num_communities = len(communities)
            modularity = nx.community.modularity(core, communities, weight="weight")
        else:
            num_communities = 0
            modularity = 0.0

        global_metrics.append({
            "Date": window_date,
            "Nodes_Count": len(G.nodes),
            "Edges_Count": len(G.edges),
            "Density": density,
            "Clustering_Coefficient": clustering,
            "Avg_Path_Length": path_length,
            "Largest_Component_Fraction": largest_fraction,
            "Modularity": modularity,
            "Num_Communities": num_communities,
        })

    output_file = folder / "global_network_metrics.csv"
    pd.DataFrame(global_metrics).to_csv(output_file, index=False)
    print(f"[{group}] Global metrics saved to: {output_file}")


def plot_all_global_metrics(group):
    metrics_file = graph_dir(group) / "global_network_metrics.csv"

    if not metrics_file.exists():
        print(f"Metrics file not found at: {metrics_file}")
        return

    df = pd.read_csv(metrics_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 17), sharex=True)

    # 1. Edges
    axes[0].plot(df["Date"], df["Edges_Count"], color="tab:blue", marker="o")
    axes[0].set_ylabel("Edges", color="tab:blue")
    axes[0].tick_params(axis="y", labelcolor="tab:blue")
    axes[0].set_title(f"Financial Correlation Network ({group}) - Global Metrics")

    # 2. Density
    axes[1].plot(df["Date"], df["Density"], color="tab:red", marker="s")
    axes[1].set_ylabel("Density", color="tab:red")
    axes[1].tick_params(axis="y", labelcolor="tab:red")

    # 3. Modularity and number of communities
    ax3_twin = axes[2].twinx()
    axes[2].plot(df["Date"], df["Modularity"], color="tab:green", marker="^")
    ax3_twin.plot(df["Date"], df["Num_Communities"], color="tab:purple",
                  linestyle="--", marker="x")
    axes[2].set_ylabel("Modularity", color="tab:green")
    ax3_twin.set_ylabel("Communities Count", color="tab:purple")
    axes[2].tick_params(axis="y", labelcolor="tab:green")
    ax3_twin.tick_params(axis="y", labelcolor="tab:purple")

    # 4. Clustering and average path length
    ax4_twin = axes[3].twinx()
    axes[3].plot(df["Date"], df["Clustering_Coefficient"], color="tab:orange", marker="d")
    ax4_twin.plot(df["Date"], df["Avg_Path_Length"], color="tab:brown",
                  linestyle="--", marker="x")
    axes[3].set_ylabel("Clustering Coeff", color="tab:orange")
    ax4_twin.set_ylabel("Avg Path Length", color="tab:brown")
    axes[3].tick_params(axis="y", labelcolor="tab:orange")
    ax4_twin.tick_params(axis="y", labelcolor="tab:brown")

    # 5. Bitcoin and S&P 500, both start at 100
    levels_file = DEFAULT_OUTPUT_DIR / "merged_levels.csv"
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
            axes[4].plot(levels_df["date"], prices / first_valid.iloc[0] * 100, label=label)

        if axes[4].lines:
            axes[4].legend()
        else:
            axes[4].text(0.5, 0.5, "Bitcoin and S&P 500 price data unavailable",
                         ha="center", va="center", transform=axes[4].transAxes)
    else:
        axes[4].text(0.5, 0.5, f"Price data file not found: {levels_file}",
                     ha="center", va="center", transform=axes[4].transAxes)
    axes[4].set_ylabel("Index (base = 100)")
    axes[4].set_xlabel("Date")
    axes[4].set_title("Bitcoin and S&P 500 (Normalized)")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    output_path = PROJECT_DIR / f"data/visualizations/{group}_comprehensive_global_metrics.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved to: {output_path}")


def load_group_averages(group):
    csv_file = graph_dir(group) / "nodes_centrality_timeseries.csv"
    if not csv_file.exists():
        print(f"File not found: {csv_file}")
        return None

    df = pd.read_csv(csv_file)

    # Remove indices and macro data, we only compare real assets
    df = df[~df["Asset"].str.startswith(("market_indices__", "macro_indicators__"))]

    # Every asset is in every window (isolated nodes are kept), so a simple mean is fine
    return df.groupby("Asset")[["Degree_Centrality", "PageRank", "Betweenness_Centrality"]].mean()


def compare_local_centrality_insights(top_n=3):
    # Each market is analyzed in its own network
    tradfi = load_group_averages("non_crypto")
    crypto = load_group_averages("crypto_only")
    if tradfi is None or crypto is None:
        return

    # 1. Degree: average level and spread
    # Degree is divided by (n-1) of each network, so compare it with care
    _, p = stats.levene(tradfi["Degree_Centrality"], crypto["Degree_Centrality"])
    print("1. Degree")
    print("   Mean: TradFi=%.4f, Crypto=%.4f" % (
        tradfi["Degree_Centrality"].mean(), crypto["Degree_Centrality"].mean()))
    print("   Std : TradFi=%.4f, Crypto=%.4f (Levene p=%.2f -> %s)" % (
        tradfi["Degree_Centrality"].std(),
        crypto["Degree_Centrality"].std(),
        p,
        "similar spread" if p >= 0.05 else "different spread",
    ))

    # 2. Top drivers and 3. Top bridges
    for title, col in [("2. Top PageRank", "PageRank"),
                       ("3. Top Betweenness", "Betweenness_Centrality")]:
        print(f"\n{title}:")
        print("  TradFi:", ", ".join(tradfi[col].nlargest(top_n).index))
        print("  Crypto:", ", ".join(crypto[col].nlargest(top_n).index))
        print(f"  Mean: TradFi={tradfi[col].mean():.4f}, Crypto={crypto[col].mean():.4f}")


if __name__ == '__main__':
    recalculate = True
    if recalculate:
        for group in GROUPS:
            generate_graph_date(group)
            analyze_graph(group)
            generate_global_metrics(group)

    for group in GROUPS:
        plot_all_global_metrics(group)

    compare_local_centrality_insights()