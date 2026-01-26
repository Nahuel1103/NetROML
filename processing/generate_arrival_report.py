"""
Generate a comprehensive report of the arrival/departure model functionality.
Creates visualizations and statistics to understand how clients are distributed over time.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path to import the model
sys.path.append(str(Path(__file__).parent.parent))
from processing.arrival_departure_model import ArrivalDepartureModel

def generate_report():
    """Generate comprehensive report with visualizations"""
    
    # Create model with specific parameters
    print("=" * 80)
    print("ARRIVAL/DEPARTURE MODEL REPORT")
    print("=" * 80)
    
    model = ArrivalDepartureModel(
        arrival_rate=3.0,      # Average 3 clients per timestep
        mean_duration=15.0,    # Average 15 timesteps duration
        total_timesteps=100,
        random_seed=314
    )
    
    # Generate events
    print("\n[1] Generating arrival/departure events...")
    events = model.simulate_all_events()
    
    # Get statistics
    stats = model.get_statistics()
    
    print(f"\n[2] Model Statistics:")
    print(f"    - Total timesteps: {stats['total_timesteps']}")
    print(f"    - Total clients generated: {stats['total_clients']}")
    print(f"    - Mean duration: {stats['mean_duration']:.2f} timesteps")
    print(f"    - Std duration: {stats['std_duration']:.2f} timesteps")
    print(f"    - Min duration: {stats['min_duration']:.0f} timesteps")
    print(f"    - Max duration: {stats['max_duration']:.0f} timesteps")
    print(f"    - Mean occupancy: {stats['mean_occupancy']:.2f} clients/timestep")
    print(f"    - Max occupancy: {stats['max_occupancy']:.0f} clients")
    
    # Theoretical expected occupancy (Little's Law)
    theoretical_occupancy = model.arrival_rate * model.mean_duration
    print(f"    - Theoretical occupancy (λ×μ): {theoretical_occupancy:.2f} clients")
    
    # Create visualizations
    print("\n[3] Creating visualizations...")
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Occupancy Grid (Gantt chart style)
    ax1 = plt.subplot(3, 3, (1, 4))
    plot_occupancy_grid(model, events, ax1)
    
    # 2. Occupancy over time
    ax2 = plt.subplot(3, 3, 2)
    plot_occupancy_timeline(model, ax2)
    
    # 3. Arrivals per timestep
    ax3 = plt.subplot(3, 3, 3)
    plot_arrivals_histogram(model, events, ax3)
    
    # 4. Duration distribution
    ax4 = plt.subplot(3, 3, 5)
    plot_duration_distribution(events, ax4)
    
    # 5. Arrivals and Departures over time
    ax5 = plt.subplot(3, 3, 6)
    plot_arrivals_departures(model, ax5)
    
    # 6. Client lifetime heatmap
    ax6 = plt.subplot(3, 3, (7, 9))
    plot_client_lifetime_heatmap(model, events, ax6)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent.parent / "docs" / "arrival_departure_report.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[4] Saved visualization to: {output_path}")
    
    # Generate markdown report
    generate_markdown_report(model, stats, events, output_path)
    
    print("\n" + "=" * 80)
    print("REPORT GENERATION COMPLETE")
    print("=" * 80)

def plot_occupancy_grid(model, events, ax):
    """Plot occupancy grid showing which clients are active at each timestep"""
    # Limit to first 50 timesteps and 30 clients for visibility
    max_t = min(50, model.total_timesteps)
    
    # Get all active clients for each timestep
    grid_data = []
    client_ids = set()
    
    for t in range(max_t):
        active = model.get_active_clients(t)
        for event in active:
            client_ids.add(event.client_id)
            grid_data.append((t, event.client_id))
    
    # Limit number of clients displayed
    client_ids = sorted(list(client_ids))[:30]
    
    # Create grid
    grid = np.zeros((len(client_ids), max_t))
    client_to_idx = {cid: i for i, cid in enumerate(client_ids)}
    
    for t, cid in grid_data:
        if cid in client_to_idx:
            grid[client_to_idx[cid], t] = 1
    
    # Plot
    sns.heatmap(grid, cmap=['white', 'darkblue'], cbar=False, 
                linewidths=0.1, linecolor='gray', ax=ax)
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Client ID', fontsize=10)
    ax.set_title('Client Activity Grid (First 50 timesteps, 30 clients)', fontsize=12, fontweight='bold')
    ax.set_yticks(np.arange(len(client_ids)) + 0.5)
    ax.set_yticklabels([f"{cid.split('_')[1]}" for cid in client_ids], fontsize=6)

def plot_occupancy_timeline(model, ax):
    """Plot number of active clients over time"""
    occupancy = [len(model.get_active_clients(t)) for t in range(model.total_timesteps)]
    
    ax.plot(occupancy, linewidth=2, color='darkblue', alpha=0.7)
    ax.axhline(y=np.mean(occupancy), color='red', linestyle='--', 
               label=f'Mean: {np.mean(occupancy):.1f}', linewidth=2)
    ax.axhline(y=model.arrival_rate * model.mean_duration, color='green', 
               linestyle=':', label=f'Theory: {model.arrival_rate * model.mean_duration:.1f}', linewidth=2)
    ax.fill_between(range(model.total_timesteps), occupancy, alpha=0.3, color='darkblue')
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Active Clients', fontsize=10)
    ax.set_title('Occupancy Over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def plot_arrivals_histogram(model, events, ax):
    """Plot histogram of arrivals per timestep"""
    arrivals_per_t = [len(model.get_arrivals_at_timestep(t)) 
                      for t in range(model.total_timesteps)]
    
    ax.hist(arrivals_per_t, bins=range(0, max(arrivals_per_t) + 2), 
            alpha=0.7, color='darkgreen', edgecolor='black')
    ax.axvline(x=model.arrival_rate, color='red', linestyle='--', 
               label=f'λ = {model.arrival_rate:.1f}', linewidth=2)
    ax.set_xlabel('Arrivals per Timestep', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Arrival Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

def plot_duration_distribution(events, ax):
    """Plot distribution of connection durations"""
    durations = [e.duration for e in events]
    
    ax.hist(durations, bins=30, alpha=0.7, color='darkorange', edgecolor='black')
    ax.axvline(x=np.mean(durations), color='red', linestyle='--', 
               label=f'Mean: {np.mean(durations):.1f}', linewidth=2)
    ax.set_xlabel('Duration (timesteps)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Duration Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

def plot_arrivals_departures(model, ax):
    """Plot arrivals and departures over time"""
    arrivals = [len(model.get_arrivals_at_timestep(t)) for t in range(model.total_timesteps)]
    departures = [len(model.get_departures_at_timestep(t)) for t in range(model.total_timesteps)]
    
    ax.bar(range(model.total_timesteps), arrivals, alpha=0.6, color='green', label='Arrivals')
    ax.bar(range(model.total_timesteps), [-d for d in departures], alpha=0.6, color='red', label='Departures')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Arrivals (+) and Departures (-) per Timestep', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

def plot_client_lifetime_heatmap(model, events, ax):
    """Plot heatmap showing client lifetimes"""
    # Limit to first 100 events for visibility
    limited_events = events[:100]
    
    # Create matrix: rows = clients, columns = timesteps
    max_t = min(100, model.total_timesteps)
    n_clients = len(limited_events)
    
    matrix = np.zeros((n_clients, max_t))
    
    for i, event in enumerate(limited_events):
        start = event.arrival_time
        end = min(event.departure_time, max_t)
        matrix[i, start:end] = 1
    
    # Plot
    sns.heatmap(matrix, cmap=['white', 'navy'], cbar=False, 
                linewidths=0, ax=ax)
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Client Event #', fontsize=10)
    ax.set_title('Client Lifetime Visualization (First 100 clients, 100 timesteps)', 
                 fontsize=12, fontweight='bold')
    
    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='navy', label='Active'),
                      Patch(facecolor='white', edgecolor='black', label='Inactive')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

def generate_markdown_report(model, stats, events, viz_path):
    """Generate markdown report with all information"""
    
    report_path = Path(__file__).parent.parent / "docs" / "arrival_departure_report.md"
    
    report = f"""# Arrival/Departure Model - Functionality Report

## Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Arrival Rate (λ) | {model.arrival_rate:.1f} | Average arrivals per timestep (Poisson) |
| Mean Duration (μ) | {model.mean_duration:.1f} | Average connection duration (Exponential) |
| Total Timesteps | {model.total_timesteps} | Simulation horizon |
| Random Seed | 314 | For reproducibility |

## Simulation Results

### Overall Statistics

| Metric | Value |
|--------|-------|
| **Total Clients Generated** | {stats['total_clients']} |
| **Mean Occupancy** | {stats['mean_occupancy']:.2f} clients/timestep |
| **Max Occupancy** | {stats['max_occupancy']:.0f} clients |
| **Theoretical Occupancy (λ×μ)** | {model.arrival_rate * model.mean_duration:.2f} clients |

### Duration Statistics

| Metric | Value (timesteps) |
|--------|-------------------|
| Mean | {stats['mean_duration']:.2f} |
| Std Dev | {stats['std_duration']:.2f} |
| Min | {stats['min_duration']:.0f} |
| Max | {stats['max_duration']:.0f} |

## Model Behavior

### How It Works

The arrival/departure model simulates a **queueing system** where:

1. **Arrivals**: At each timestep `t`, the number of new clients is drawn from a Poisson distribution with rate λ = {model.arrival_rate}
2. **Duration**: Each arriving client's connection duration is drawn from an Exponential distribution with mean μ = {model.mean_duration}
3. **Departures**: Client `i` departs at timestep `t_arrival + duration`

### Mathematical Foundation

**Poisson Process (Arrivals)**:
```
P(N(t) = k) = (λ^k × e^(-λ)) / k!
```

**Exponential Distribution (Duration)**:
```
f(d) = (1/μ) × e^(-d/μ)
```

**Little's Law (Expected Occupancy)**:
```
L = λ × W = {model.arrival_rate} × {model.mean_duration} = {model.arrival_rate * model.mean_duration:.2f} clients
```

## Visualizations

![Arrival/Departure Model Visualizations]({viz_path.name})

### Visualization Descriptions

1. **Client Activity Grid**: Shows which clients are active (blue) vs inactive (white) over the first 50 timesteps
   - Each row is a client
   - Each column is a timestep
   - Blue cells indicate the client is connected

2. **Occupancy Over Time**: Line plot showing total number of active clients at each timestep
   - **Blue line**: Actual occupancy
   - **Red dashed**: Observed mean occupancy
   - **Green dotted**: Theoretical expected occupancy (λ×μ)

3. **Arrival Distribution**: Histogram showing how many clients arrive at each timestep
   - **Red line**: Expected arrival rate (λ)
   - Demonstrates Poisson distribution

4. **Duration Distribution**: Histogram of connection durations
   - **Red line**: Mean duration
   - Shows exponential distribution shape

5. **Arrivals and Departures**: Bar chart showing:
   - **Green bars (positive)**: New arrivals
   - **Red bars (negative)**: Departures

6. **Client Lifetime Heatmap**: Gantt-style visualization showing when each client is active
   - Each row is one client instance
   - Shows the duration and timing of each connection

## Key Insights

### Observed vs Theoretical

- **Theoretical Mean Occupancy**: {model.arrival_rate * model.mean_duration:.2f} clients
- **Observed Mean Occupancy**:{stats['mean_occupancy']:.2f} clients
- **Difference**: {abs(stats['mean_occupancy'] - model.arrival_rate * model.mean_duration):.2f} clients ({abs(stats['mean_occupancy'] - model.arrival_rate * model.mean_duration) / (model.arrival_rate * model.mean_duration) * 100:.1f}%)

The difference is due to:
1. Finite simulation horizon (transient effects)
2. Random variation
3. Edge effects at start and end of simulation

### Distribution Characteristics

- **Arrival Variability**: Poisson distribution creates natural burstiness in arrivals
- **Duration Spread**: Exponential distribution produces high variability (std = {stats['std_duration']:.2f}, mean = {stats['mean_duration']:.2f})
- **Dynamic Occupancy**: Number of active clients varies significantly over time (max: {stats['max_occupancy']:.0f}, mean: {stats['mean_occupancy']:.2f})

## Integration with Network Environment

This model is used in the WiFi network simulation to:

1. **Dynamically generate active clients** at each timestep
2. **Control client lifecycle** (when they connect/disconnect)
3. **Create realistic load patterns** instead of static client sets
4. **Enable temporal dynamics** in network behavior

Each simulated client is mapped to a real MAC address from the dataset, allowing us to use historical RSSI data while maintaining dynamic arrival/departure patterns.

---
Generated: {pd.Timestamp.now()}
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"[5] Saved markdown report to: {report_path}")

if __name__ == "__main__":
    import pandas as pd
    generate_report()
