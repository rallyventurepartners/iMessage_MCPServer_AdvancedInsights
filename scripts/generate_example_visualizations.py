#!/usr/bin/env python3
"""
Generate example visualization images for the README documentation.

This script creates sample charts that demonstrate the visualization
capabilities of the enhanced tools without requiring actual iMessage data.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import os
from pathlib import Path

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
output_dir = Path(__file__).parent.parent / "docs" / "assets"
output_dir.mkdir(parents=True, exist_ok=True)


def generate_example_heatmap():
    """Generate an example communication heatmap."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Generate sample data
    np.random.seed(42)
    heatmap_data = np.zeros((7, 24))
    
    # Simulate realistic communication patterns
    for day in range(7):
        for hour in range(24):
            # Lower activity at night, higher during day
            base_activity = 5 if 0 <= hour <= 6 else 15 if 7 <= hour <= 22 else 3
            
            # Weekday vs weekend patterns
            if day < 5:  # Weekdays
                if 9 <= hour <= 17:  # Work hours
                    base_activity *= 0.7
                elif 18 <= hour <= 22:  # Evening
                    base_activity *= 1.3
            else:  # Weekend
                if 10 <= hour <= 23:
                    base_activity *= 1.2
            
            heatmap_data[day][hour] = base_activity + np.random.normal(0, 3)
    
    # Ensure non-negative values
    heatmap_data = np.maximum(heatmap_data, 0)
    
    # Create heatmap
    sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        cbar_kws={"label": "Message Count"},
        yticklabels=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
        xticklabels=[f"{h:02d}" for h in range(24)],
        ax=ax,
        vmin=0,
        vmax=30
    )
    
    ax.set_title("Communication Heatmap by Day and Hour", fontsize=16, pad=20)
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Day of Week", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "example_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated example_heatmap.png")


def generate_example_balance_chart():
    """Generate an example message balance chart."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Generate sample data for 30 days
    days = 30
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Simulate sent/received messages
    np.random.seed(42)
    sent = np.random.poisson(15, days) + np.random.randint(0, 10, days)
    received = np.random.poisson(12, days) + np.random.randint(0, 10, days)
    
    # Stacked area chart
    ax1.fill_between(range(days), sent, alpha=0.7, label="Sent", color='steelblue')
    ax1.fill_between(range(days), sent, sent + received, alpha=0.7, label="Received", color='coral')
    ax1.set_ylabel("Messages", fontsize=12)
    ax1.set_title("Message Flow (Sent vs Received)", fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(sent + received) * 1.1)
    
    # Balance ratio
    balance = sent / (sent + received)
    ax2.plot(range(days), balance, linewidth=2, color='purple')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(range(days), 0.5, balance, 
                    where=balance > 0.5, 
                    alpha=0.3, color='steelblue', label='You initiate more')
    ax2.fill_between(range(days), balance, 0.5,
                    where=balance < 0.5,
                    alpha=0.3, color='coral', label='They initiate more')
    ax2.set_ylabel("Send Ratio", fontsize=12)
    ax2.set_xlabel("Days", fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    ax2.set_xticks(range(0, days, 5))
    ax2.set_xticklabels([f"Day {i}" for i in range(0, days, 5)], rotation=45)
    
    plt.suptitle("Communication Balance Analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "example_balance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated example_balance.png")


def generate_example_dashboard():
    """Generate an example relationship dashboard."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Generate sample data
    np.random.seed(42)
    periods = 30
    dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
    
    # 1. Message volume over time
    ax1 = fig.add_subplot(gs[0, :])
    volume = np.random.poisson(20, periods) + np.random.randint(-5, 10, periods)
    volume = np.maximum(volume, 5)  # Ensure positive
    
    ax1.plot(range(periods), volume, linewidth=2, marker='o', markersize=4, color='darkblue')
    ax1.fill_between(range(periods), volume, alpha=0.3, color='skyblue')
    ax1.set_title("Message Volume Trend", fontsize=14)
    ax1.set_ylabel("Messages", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, periods)
    
    # 2. Communication balance
    ax2 = fig.add_subplot(gs[1, 0])
    balance = 0.5 + np.cumsum(np.random.normal(0, 0.02, periods))
    balance = np.clip(balance, 0.2, 0.8)  # Keep within reasonable bounds
    
    ax2.plot(range(periods), balance, linewidth=2, color='purple')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(range(periods), 0.5, balance, alpha=0.3, color='purple')
    ax2.set_title("Communication Balance", fontsize=14)
    ax2.set_ylabel("Your Message Ratio", fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # 3. Average message length
    ax3 = fig.add_subplot(gs[1, 1])
    avg_length = 50 + np.cumsum(np.random.normal(0, 2, periods))
    avg_length = np.maximum(avg_length, 20)  # Ensure reasonable minimum
    
    ax3.plot(range(periods), avg_length, linewidth=2, color='green', marker='s', markersize=4)
    ax3.set_title("Average Message Length", fontsize=14)
    ax3.set_ylabel("Characters", fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. Engagement score
    ax4 = fig.add_subplot(gs[2, :])
    engagement = 0.7 + np.cumsum(np.random.normal(0, 0.01, periods))
    engagement = np.clip(engagement, 0.3, 0.95)
    
    ax4.plot(range(periods), engagement, linewidth=3, marker='o', markersize=6, color='darkgreen')
    ax4.fill_between(range(periods), engagement, alpha=0.3, color='lightgreen')
    ax4.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Healthy threshold')
    ax4.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Warning threshold')
    ax4.set_title("Relationship Engagement Score Over Time", fontsize=14)
    ax4.set_ylabel("Engagement Score (0-1)", fontsize=12)
    ax4.set_xlabel("Days", fontsize=12)
    ax4.set_ylim(0, 1)
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    
    # Format x-axes
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim(-1, periods)
        if ax in [ax2, ax3]:
            ax.set_xticklabels([])
        else:
            ax.set_xticks(range(0, periods, 5))
            ax.set_xticklabels([f"Day {i}" for i in range(0, periods, 5)])
    
    plt.suptitle("Relationship Analytics Dashboard - 30 days", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "example_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated example_dashboard.png")


def update_existing_sentiment_chart():
    """Update the sentiment evolution chart with better styling."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate sample sentiment data
    np.random.seed(42)
    weeks = 12
    dates = pd.date_range(end=datetime.now(), periods=weeks, freq='W')
    
    # Simulate sentiment scores with trend
    base_sentiment = 0.6
    sentiment_scores = []
    for i in range(weeks):
        # Add some trend and noise
        trend = 0.002 * i  # Slight upward trend
        noise = np.random.normal(0, 0.1)
        score = base_sentiment + trend + noise
        sentiment_scores.append(np.clip(score, -1, 1))
    
    # Plot sentiment line
    ax.plot(range(weeks), sentiment_scores, linewidth=3, marker='o', 
            markersize=8, color='darkblue', label='Sentiment Score')
    
    # Add rolling average
    window = 3
    rolling_avg = pd.Series(sentiment_scores).rolling(window=window, center=True).mean()
    ax.plot(range(weeks), rolling_avg, linewidth=2, alpha=0.7, 
            linestyle='--', color='lightblue', label=f'{window}-week average')
    
    # Color bands for sentiment ranges
    ax.axhspan(0.5, 1.0, alpha=0.1, color='green', label='Positive')
    ax.axhspan(-0.5, 0.5, alpha=0.1, color='gray')
    ax.axhspan(-1.0, -0.5, alpha=0.1, color='red')
    
    # Formatting
    ax.set_xlabel('Weeks', fontsize=12)
    ax.set_ylabel('Sentiment Score', fontsize=12)
    ax.set_title('Sentiment Evolution Over Time', fontsize=16, pad=20)
    ax.set_ylim(-1, 1)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Set x-axis labels
    ax.set_xticks(range(weeks))
    ax.set_xticklabels([f'W{i+1}' for i in range(weeks)], rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Updated sentiment_evolution.png")


def main():
    """Generate all example visualizations."""
    print("Generating example visualizations...")
    
    # Generate new charts
    generate_example_heatmap()
    generate_example_balance_chart()
    generate_example_dashboard()
    
    # Update existing chart style
    update_existing_sentiment_chart()
    
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    for file in ["example_heatmap.png", "example_balance.png", "example_dashboard.png", "sentiment_evolution.png"]:
        if (output_dir / file).exists():
            print(f"  ✓ {file}")


if __name__ == "__main__":
    main()