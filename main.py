import os
import sys
import csv
import random
import argparse
import json
from datetime import datetime
from config import (
    ENERGY_LOSS_PER_TURN,
    AGENT_MEMORY_SIZE,
    USE_MULTIMODAL
)
from config import REPLENISH_INTERVAL

from environment import Environment
import config


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run agent simulation')
parser.add_argument('--red_rep', type=int, default=config.REPLENISH_RED_COUNT, help='Red replenishment rate')
parser.add_argument('--gre_rep', type=int, default=config.REPLENISH_GREEN_COUNT, help='Green replenishment rate')
parser.add_argument('--con_rate', type=int, default=config.CONSUMPTION_RATE, help='Consumption rate multiplier')
parser.add_argument('--agent_count', type=int, default=config.NUM_AGENTS, help='Number of agents')
parser.add_argument('--grid_size', type=int, default=config.GRID_SIZE, help='Grid size (NxN)')
parser.add_argument('--total_steps', type=int, default=config.TOTAL_STEPS, help='Total simulation steps')
parser.add_argument('--job_id', type=int, default=config.JOB_ID, help='Manual job ID number')
parser.add_argument('--trade_range', type=int, default=0, help='Trading range for agents (0=disabled, >=1=enabled with range)')
parser.add_argument('--rand', type=int, choices=[0, 1], default=0, help='Random mode (0=LLM decisions, 1=random decisions)')



args = parser.parse_args()

# Update config with parsed arguments
config.REPLENISH_RED_COUNT = args.red_rep
config.REPLENISH_GREEN_COUNT = args.gre_rep
config.CONSUMPTION_RATE = args.con_rate
config.NUM_AGENTS = args.agent_count
config.GRID_SIZE = args.grid_size
config.TOTAL_STEPS = args.total_steps
config.TRADE_RANGE = args.trade_range
config.RAND_MODE = args.rand

# Update AGENT_CONFIGS with new consumption rate

config.AGENT_CONFIGS = {}

base_configs = [
    {'red': 4, 'green': 0},  # Red specialist
    {'red': 3, 'green': 1},  # Red preference  
    {'red': 2, 'green': 2},  # Balanced
    {'red': 1, 'green': 3},  # Green preference
    {'red': 0, 'green': 4},  # Green specialist
]

for i in range(config.NUM_AGENTS):
    agent_name = f"Agent{i+1}"
    base_config = base_configs[i % len(base_configs)]  # Cycle through patterns
    config.AGENT_CONFIGS[agent_name] = {
        'red': base_config['red'] * config.CONSUMPTION_RATE,
        'green': base_config['green'] * config.CONSUMPTION_RATE
    }

# NOW import Agent after AGENT_CONFIGS is created
from agent import Agent  # ✅ Move to here

print(f"Using: RED_REP={config.REPLENISH_RED_COUNT}, GRE_REP={config.REPLENISH_GREEN_COUNT}, CON_RATE={config.CONSUMPTION_RATE}, AGENT_COUNT={config.NUM_AGENTS}, GRID_SIZE={config.GRID_SIZE}, TOTAL_STEPS={config.TOTAL_STEPS}, TRADE_RANGE={config.TRADE_RANGE}, RAND_MODE={config.RAND_MODE}")


def generate_unique_positions(num_agents: int, grid_size: int):
    positions = set()
    while len(positions) < num_agents:
        positions.add((
            random.randint(0, grid_size - 1),
            random.randint(0, grid_size - 1)
        ))
    return list(positions)

def main():
    # Generate JOB ID with all parameters - run number at the end
    job_id = f"JOB_R{args.red_rep}_G{args.gre_rep}_C{args.con_rate}_A{args.agent_count}_S{args.grid_size}_T{args.total_steps}_TR{args.trade_range}_RD{args.rand}_{args.job_id}"
    job_dir = f"logs/{job_id}"
    os.makedirs(job_dir, exist_ok=True)

    # Create metadata
    metadata = {
        "job_id": job_id,
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "red_rep": config.REPLENISH_RED_COUNT,
            "gre_rep": config.REPLENISH_GREEN_COUNT,
            "con_rate": config.CONSUMPTION_RATE,
            "agent_count": config.NUM_AGENTS,
            "grid_size": config.GRID_SIZE,
            "total_steps": config.TOTAL_STEPS,
            "trade_range": config.TRADE_RANGE,
            "rand_mode": config.RAND_MODE
        }
    }
    with open(f"{job_dir}/run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Set LLM log path
    from llm import set_log_path
    set_log_path(job_dir)

    # Initialize trading system if enabled
    market = None
    if config.TRADE_RANGE > 0:
        from trading import Market
        market = Market()
        print(f"🔄 Trading system enabled with range {config.TRADE_RANGE}")

    # Prepare environment and agents
    env = Environment()
    positions = generate_unique_positions(config.NUM_AGENTS, config.GRID_SIZE)
    agents = [
        Agent(f"Agent{i+1}", start_pos=positions[i])
        for i in range(config.NUM_AGENTS)
    ]

    # Open trace file
    trace_path = f"{job_dir}/simulation_trace.csv"
    
    # Create trading log file if trading is enabled
    trading_log_path = None
    trading_log_file = None
    if config.TRADE_RANGE > 0:
        trading_log_path = f"{job_dir}/trading_log.txt"
        trading_log_file = open(trading_log_path, "w")
    
    with open(trace_path, "w", newline="") as trace_file:
        trace_writer = csv.writer(trace_file)
        trace_writer.writerow(["step", "agent", "position", "energy", "inventory_red", "inventory_green", "action", "outcome", "cell_content"])
        
        # Main simulation loop
        for step in range(1, config.TOTAL_STEPS + 1):
            print(f"\n--- Step {step} ---")
            alive_count = sum(1 for a in agents if a.alive)
            print(f"Alive: {alive_count}/{config.NUM_AGENTS}")

            for agent in agents:
                if agent.alive:
                    # Get cell content before action
                    x, y = agent.position
                    cell_content = env.get_cell_content(x, y) or "empty"
                    
                    action_result = agent.decide_and_act(env, all_agents=agents)
                    
                    # Log to trace file
                    trace_writer.writerow([
                        step, agent.name, f"({x},{y})", agent.energy,
                        agent.inventory['red'], agent.inventory['green'],
                        action_result, action_result, cell_content
                    ])
                else:
                    action_result = "inactive"

                # Console log
                print(f"{agent.name} @ {agent.position} | E={agent.energy}: {action_result}")

            # Trading phase (if enabled)
            if config.TRADE_RANGE > 0 and market:
                from trading import process_trading_phase
                trading_events = process_trading_phase(agents, market, step)
                
                # Log trading events
                if trading_events and trading_log_file:
                    for event in trading_events:
                        trading_log_file.write(f"{event}\n")
                        print(f"🔄 {event}")
                    trading_log_file.flush()

            # Replenish food periodically
            if step % REPLENISH_INTERVAL == 0:
                print(f"🔄 Replenishing {config.REPLENISH_RED_COUNT} red & "
                      f"{config.REPLENISH_GREEN_COUNT} green")
                env.fixed_replenish(
                    red_count=config.REPLENISH_RED_COUNT,
                    green_count=config.REPLENISH_GREEN_COUNT
                )

        final_survivors = sum(1 for a in agents if a.alive)
        metadata["final_results"] = {
            "total_steps_completed": step,
            "initial_agents": config.NUM_AGENTS,
            "final_survivors": final_survivors,
            "survival_rate": final_survivors / config.NUM_AGENTS * 100,
            "agent_final_status": {
                agent.name: {
                    "survived": agent.alive,
                    "final_energy": agent.energy,
                    "type": agent.type
                } for agent in agents
            }
        }
        
        # Add trading statistics if trading was enabled
        if config.TRADE_RANGE > 0 and market:
            metadata["trading_results"] = market.get_trade_summary()

        # Save updated metadata
        with open(f"{job_dir}/run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    # Close trading log file if it was opened
    if trading_log_file:
        trading_log_file.close()

    print("\nSimulation complete.")
    print(f"Results saved to {job_dir}")
    print(f"Trace file: {trace_path}")
    if trading_log_path:
        print(f"Trading log: {trading_log_path}")
    print(f"Metadata: {job_dir}/run_metadata.json")

if __name__ == "__main__":
    main()