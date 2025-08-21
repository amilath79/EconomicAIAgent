import json
import matplotlib.pyplot as plt
from collections import defaultdict

def load_stats(filename="game_stats.json"):
    """Load game statistics from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found!")
        return []

def analyze_survival(stats):
    """Analyze agent survival over time"""
    steps = []
    alive_counts = []
    
    for snapshot in stats:
        steps.append(snapshot['step'])
        alive_counts.append(snapshot['alive_count'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, alive_counts, marker='o')
    plt.xlabel('Step')
    plt.ylabel('Alive Agents')
    plt.title('Agent Survival Over Time')
    plt.grid(True)
    plt.show()

def analyze_energy_by_agent(stats):
    """Analyze energy levels for each agent over time"""
    agent_energy = defaultdict(lambda: {'steps': [], 'energy': []})
    
    for snapshot in stats:
        step = snapshot['step']
        for agent in snapshot['agents']:
            if agent['alive']:
                agent_energy[agent['name']]['steps'].append(step)
                agent_energy[agent['name']]['energy'].append(agent['energy'])
    
    plt.figure(figsize=(12, 8))
    for agent_name, data in agent_energy.items():
        plt.plot(data['steps'], data['energy'], marker='o', label=agent_name)
    
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.title('Agent Energy Levels Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_inventory(stats):
    """Analyze total inventory over time"""
    steps = []
    total_red = []
    total_green = []
    
    for snapshot in stats:
        step = snapshot['step']
        red_count = sum(agent['inventory']['red'] for agent in snapshot['agents'] if agent['alive'])
        green_count = sum(agent['inventory']['green'] for agent in snapshot['agents'] if agent['alive'])
        
        steps.append(step)
        total_red.append(red_count)
        total_green.append(green_count)
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, total_red, 'r-', marker='o', label='Red Food')
    plt.plot(steps, total_green, 'g-', marker='o', label='Green Food')
    plt.xlabel('Step')
    plt.ylabel('Total Food in Inventories')
    plt.title('Total Food Inventory Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_agent_types(stats):
    """Analyze performance by agent type"""
    if not stats:
        return
    
    # Get final snapshot
    final_snapshot = stats[-1]
    
    # Count survivors by type
    type_survival = defaultdict(lambda: {'total': 0, 'survived': 0})
    
    # Process all agents from first snapshot to get totals
    first_snapshot = stats[0]
    for agent in first_snapshot['agents']:
        agent_type = agent['type']
        type_survival[agent_type]['total'] += 1
    
    # Check who survived
    for agent in final_snapshot['agents']:
        if agent['alive']:
            agent_type = agent['type']
            type_survival[agent_type]['survived'] += 1
    
    # Plot survival rates
    types = list(type_survival.keys())
    survival_rates = [type_survival[t]['survived'] / type_survival[t]['total'] * 100 
                     for t in types]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(types, survival_rates, color=['red', 'green', 'blue'])
    plt.xlabel('Agent Type')
    plt.ylabel('Survival Rate (%)')
    plt.title('Survival Rate by Agent Type')
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for bar, rate in zip(bars, survival_rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.show()

def print_summary(stats):
    """Print summary statistics"""
    if not stats:
        print("No statistics available!")
        return
    
    print("\n=== GAME SUMMARY ===")
    print(f"Total steps analyzed: {len(stats)}")
    
    # Initial vs final alive count
    initial_alive = stats[0]['alive_count']
    final_alive = stats[-1]['alive_count']
    print(f"Initial agents: {initial_alive}")
    print(f"Final survivors: {final_alive}")
    print(f"Survival rate: {final_alive/initial_alive*100:.1f}%")
    
    # Average energy at end
    final_agents = [a for a in stats[-1]['agents'] if a['alive']]
    if final_agents:
        avg_energy = sum(a['energy'] for a in final_agents) / len(final_agents)
        print(f"Average final energy: {avg_energy:.1f}")
    
    # Agent details
    print("\n=== FINAL AGENT STATUS ===")
    for agent in stats[-1]['agents']:
        status = "ALIVE" if agent['alive'] else "DEAD"
        print(f"{agent['name']} ({agent['type']}): {status}")
        if agent['alive']:
            print(f"  Energy: {agent['energy']}")
            print(f"  Inventory: Red={agent['inventory']['red']}, Green={agent['inventory']['green']}")
            print(f"  Position: {agent['position']}")
            if agent.get('last_actions'):
                print(f"  Last actions: {', '.join(agent['last_actions'][-3:])}")

def main():
    """Run all analyses"""
    print("Loading game statistics...")
    stats = load_stats()
    
    if not stats:
        print("No statistics to analyze!")
        return
    
    print_summary(stats)
    
    print("\nGenerating visualizations...")
    analyze_survival(stats)
    analyze_energy_by_agent(stats)
    analyze_inventory(stats)
    analyze_agent_types(stats)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # Check if matplotlib is install
    try:
        import matplotlib
        main()
    except ImportError:
        print("Please install matplotlib to use the analysis tool:")
        print("pip install matplotlib")
        print("\nYou can still view the raw statistics in game_stats.json")