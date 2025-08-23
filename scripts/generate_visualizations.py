#!/usr/bin/env python3
"""
Generate visualization examples for documentation using synthetic data.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def generate_sentiment_evolution_chart():
    """Generate a sentiment evolution chart."""
    # Create synthetic data
    dates = pd.date_range(start='2024-06-01', end='2024-12-15', freq='W')

    # Multiple contacts with different patterns
    data = {
        'Alice': np.concatenate([
            np.random.normal(0.3, 0.05, 10),  # Start neutral
            np.random.normal(0.5, 0.05, 10),  # Improve
            np.random.normal(0.7, 0.05, len(dates)-20)  # Stabilize positive
        ])[:len(dates)],
        'Bob': 0.6 + 0.1 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 0.05, len(dates)),
        'Carol': np.concatenate([
            np.random.normal(0.7, 0.05, 15),  # Start positive
            np.random.normal(0.4, 0.05, len(dates)-15)  # Decline
        ])[:len(dates)]
    }

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for contact, scores in data.items():
        ax.plot(dates, scores, marker='o', markersize=4, label=contact, linewidth=2)

    # Add trend line for Alice
    z = np.polyfit(range(len(dates)), data['Alice'], 1)
    p = np.poly1d(z)
    ax.plot(dates, p(range(len(dates))), "--", alpha=0.8, color='gray', label='Trend (Alice)')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sentiment Score', fontsize=12)
    ax.set_title('Sentiment Evolution Over Time', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.annotate('Relationship improving',
                xy=(dates[15], data['Alice'][15]),
                xytext=(dates[15], 0.8),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                fontsize=10, color='green')

    plt.tight_layout()
    plt.savefig('assets/sentiment_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_communication_heatmap():
    """Generate a communication frequency heatmap."""
    # Days and hours
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hours = [f"{h:02d}:00" for h in range(24)]

    # Generate synthetic data with realistic patterns
    data = np.zeros((7, 24))

    # Weekday patterns (work hours + evening)
    for d in range(5):  # Mon-Fri
        # Morning spike
        data[d, 8:10] = np.random.randint(5, 15, 2)
        # Lunch time
        data[d, 12:14] = np.random.randint(8, 20, 2)
        # Evening spike
        data[d, 18:22] = np.random.randint(15, 30, 4)

    # Weekend patterns
    for d in range(5, 7):  # Sat-Sun
        # Later morning
        data[d, 10:13] = np.random.randint(10, 25, 3)
        # Afternoon
        data[d, 14:17] = np.random.randint(8, 18, 3)
        # Evening
        data[d, 19:23] = np.random.randint(20, 35, 4)

    # Add some noise
    data += np.random.randint(0, 3, (7, 24))

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))

    # Select peak hours for display
    peak_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    data_subset = data[:, peak_hours]
    hours_subset = [hours[h] for h in peak_hours]

    sns.heatmap(data_subset,
                xticklabels=hours_subset,
                yticklabels=days,
                cmap='YlOrRd',
                annot=True,
                fmt='g',
                cbar_kws={'label': 'Message Count'},
                vmin=0,
                vmax=35)

    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Day of Week', fontsize=12)
    ax.set_title('Communication Frequency Heatmap', fontsize=16, fontweight='bold')

    # Highlight peak times
    ax.add_patch(plt.Rectangle((10, 5), 4, 2, fill=False, edgecolor='blue', lw=3))
    ax.text(12, 7.3, 'Peak Times', fontsize=10, ha='center', color='blue', fontweight='bold')

    plt.tight_layout()
    plt.savefig('assets/cadence_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_network_graph():
    """Generate a social network visualization."""
    # Create a network
    G = nx.Graph()

    # Add nodes (contacts)
    contacts = ['You', 'Alice', 'Bob', 'Carol', 'David', 'Emma', 'Frank', 'Grace', 'Henry']
    G.add_nodes_from(contacts)

    # Add edges with weights (message counts)
    edges = [
        ('You', 'Alice', 120),
        ('You', 'Bob', 95),
        ('You', 'Carol', 80),
        ('You', 'David', 45),
        ('Alice', 'Bob', 30),
        ('Alice', 'Carol', 25),
        ('Bob', 'David', 20),
        ('Carol', 'Emma', 35),
        ('David', 'Frank', 15),
        ('Emma', 'Grace', 40),
        ('Frank', 'Henry', 25),
        ('Grace', 'Henry', 30),
        ('You', 'Emma', 60),
        ('You', 'Grace', 40),
    ]

    for u, v, weight in edges:
        G.add_edge(u, v, weight=weight)

    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw edges with varying thickness
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(weights)
    edge_widths = [5 * w / max_weight for w in weights]

    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')

    # Draw nodes with sizes based on degree centrality
    centrality = nx.degree_centrality(G)
    node_sizes = [3000 * centrality[node] for node in G.nodes()]

    # Color nodes by community
    node_colors = ['#ff6b6b' if node == 'You' else
                   '#4ecdc4' if node in ['Alice', 'Bob', 'Carol'] else
                   '#95e1d3' if node in ['David', 'Emma'] else
                   '#a8e6cf' for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    # Add title and legend
    ax.set_title('Social Network Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff6b6b', label='You'),
        Patch(facecolor='#4ecdc4', label='Close Friends'),
        Patch(facecolor='#95e1d3', label='Colleagues'),
        Patch(facecolor='#a8e6cf', label='Extended Network')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True)

    # Add annotations
    ax.text(0.5, -0.1, 'Node size = Centrality | Edge width = Message volume',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig('assets/network_graph.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_response_time_distribution():
    """Generate response time distribution chart."""
    # Generate synthetic response times (in minutes)
    # Mix of quick responses and longer delays
    quick_responses = np.random.exponential(5, 500)  # Quick responses
    normal_responses = np.random.normal(30, 15, 300)  # Normal responses
    slow_responses = np.random.exponential(60, 100) + 60  # Slow responses

    all_responses = np.concatenate([quick_responses, normal_responses, slow_responses])
    all_responses = all_responses[all_responses > 0]  # Remove negative values
    all_responses = all_responses[all_responses < 180]  # Cap at 3 hours

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    ax1.hist(all_responses, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.median(all_responses), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(all_responses):.1f} min')
    ax1.axvline(np.mean(all_responses), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_responses):.1f} min')
    ax1.set_xlabel('Response Time (minutes)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Response Time Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Box plot by time of day
    # Create synthetic data for different times
    times = ['Morning\n(6-12)', 'Afternoon\n(12-18)', 'Evening\n(18-24)', 'Night\n(0-6)']
    time_data = [
        np.random.exponential(8, 200),  # Morning - quick
        np.random.normal(25, 10, 200),  # Afternoon - moderate
        np.random.normal(15, 8, 200),   # Evening - fairly quick
        np.random.exponential(45, 200) + 20  # Night - slow
    ]

    # Clean data
    time_data = [d[d > 0] for d in time_data]
    time_data = [d[d < 120] for d in time_data]

    bp = ax2.boxplot(time_data, labels=times, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
    ax2.set_ylabel('Response Time (minutes)', fontsize=12)
    ax2.set_title('Response Time by Time of Day', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('assets/response_time_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Generate all visualizations."""
    print("Generating visualizations...")

    # Ensure assets directory exists
    Path('assets').mkdir(exist_ok=True)

    # Generate each visualization
    print("- Sentiment evolution chart")
    generate_sentiment_evolution_chart()

    print("- Communication heatmap")
    generate_communication_heatmap()

    print("- Network graph")
    generate_network_graph()

    print("- Response time distribution")
    generate_response_time_distribution()

    print("\nVisualizations generated in assets/ directory")


if __name__ == "__main__":
    main()
