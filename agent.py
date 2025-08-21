import json
import random
from llm import get_agent_action
from config import (
    AGENT_CONFIGS,
    ENERGY_LOSS_PER_TURN,
    AGENT_MEMORY_SIZE
)


class Agent:
    def __init__(self, name, start_pos=(4, 4)):
        self.name = name
        self.position = start_pos
        # Initialize inventory & energy from config
        from config import INITIAL_INVENTORY, INITIAL_ENERGY
        self.inventory = INITIAL_INVENTORY.copy()
        self.energy = INITIAL_ENERGY
        self.alive = True

        # Load this agent's consumption rates from AGENT_CONFIGS
        self.consumption_rates = AGENT_CONFIGS.get(self.name, {'red': 0, 'green': 0})

        # Histories & counters
        self.actions_taken = []
        self.memory = []
        self.movement_history = []
        self.step_count = 0

    @property
    def type(self):
        """Determine agent 'type' from its highest consumption rate."""
        r, g = self.consumption_rates['red'], self.consumption_rates['green']
        if r > g:
            return 'red'
        elif g > r:
            return 'green'
        else:
            return 'balanced'

    def add_memory(self, observation, action, outcome):
        entry = (
            f"Step {self.step_count}: "
            f"Action: {action} | "
            f"Observation: {observation} | "
            f"Outcome: {outcome} | "
            f"Energy: {self.energy} | "
            f"Inventory: {self.inventory}"
        )
        self.memory.append(entry)
        # keep only last N
        if len(self.memory) > AGENT_MEMORY_SIZE:
            self.memory = self.memory[-AGENT_MEMORY_SIZE:]

    # def update_movement_history(self, cell_content, action_taken):
    #     entry = {
    #         "step": self.step_count,
    #         "position": self.position,
    #         "cell_content": cell_content or "empty",
    #         "action_taken": action_taken
    #     }
    #     self.movement_history.append(entry)
    #     if len(self.movement_history) > 3:
    #         self.movement_history.pop(0)

    #     # save to file
    #     with open(f"logs/movement_history_{self.name}.txt", "w", encoding="utf-8") as f:
    #         json.dump(self.movement_history, f, indent=2)

    def get_current_observation(self, environment, all_agents):
        x, y = self.position
        cell = environment.get_cell_content(x, y) or "empty"

        nearby = {'red': 0, 'green': 0, 'agents': 0}
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < environment.size and 0 <= ny < environment.size:
                    c = environment.get_cell_content(nx, ny)
                    if c in nearby:
                        nearby[c] += 1
        for other in all_agents:
            if other.alive and other is not self:
                ox, oy = other.position
                if abs(ox - x) <= 1 and abs(oy - y) <= 1:
                    nearby['agents'] += 1

        return (f"at {self.position}, cell has {cell}, "
                f"nearby {nearby['red']}R {nearby['green']}G {nearby['agents']}A, "
                f"energy {self.energy}")

    def decide_and_act(self, environment, trade_manager=None, all_agents=[]):
        if not self.alive:
            return "inactive"

        self.step_count += 1

        # Lose energy each turn
        self.energy -= ENERGY_LOSS_PER_TURN
        if self.energy <= 0:
            self.alive = False
            return "ran out of energy"

        obs = self.get_current_observation(environment, all_agents)
        x, y = self.position
        cell = environment.get_cell_content(x, y)
        occupied = {a.position for a in all_agents if a.alive and a is not self}

        # Check if we're in random mode
        import config
        if config.RAND_MODE == 1:
            return self._random_action(environment, occupied)

        # Prepare visual if multimodal
        grid_b64 = None

        retry = None
        for _ in range(2):
            action = get_agent_action(
                agent_name=self.name,
                position=self.position,
                inventory=self.inventory,
                cell_content=cell,
                energy=self.energy,
                consumption_rate=self.consumption_rates,
                memory=self.memory,
                grid_image_base64=grid_b64,
                retry_message=retry
            ) or "do nothing"

            self.actions_taken.append(action)
            result = None

            if action.startswith("move"):
                direction = action.split()[1]
                new_pos = {
                    'up':    (max(0, x-1), y),
                    'down':  (min(environment.size-1, x+1), y),
                    'left':  (x, max(0, y-1)),
                    'right': (x, min(environment.size-1, y+1))
                }[direction]
                if new_pos not in occupied and new_pos != self.position:
                    self.position = new_pos
                    result = f"moved {direction} (energy: {self.energy})"
                else:
                    retry = f"move {direction} blocked"
                    result = "move blocked: other agent or wall"

            elif action == "collect":
                item = environment.get_cell_content(x, y)
                if item and item in self.inventory:
                    self.inventory[item] += 1
                    environment.clear_cell(x, y)
                    result = f"collected {item}"
                else:
                    result = "nothing to collect"

            elif action == "eat red" and self.inventory['red'] > 0:
                self.inventory['red'] -= 1
                gain = self.consumption_rates['red']
                self.energy += gain
                result = f"ate red (+{gain})"

            elif action == "eat green" and self.inventory['green'] > 0:
                self.inventory['green'] -= 1
                gain = self.consumption_rates['green']
                self.energy += gain
                result = f"ate green (+{gain})"

            elif action == "do nothing":
                result = "did nothing"

            else:
                retry = f"action '{action}' invalid"
                result = "failed to act"

            # record memory & movement
            self.add_memory(obs, action, result)
            # self.update_movement_history(cell, result)

            return result

        # if both attempts failed
        self.add_memory(obs, "no valid action", "failed both retries")
        self.update_movement_history(cell, "failed to act")
        return "failed to act"

    def _random_action(self, environment, occupied):
        """Execute random action when in random mode"""
        x, y = self.position
        cell = environment.get_cell_content(x, y)
        
        # Available actions
        actions = []
        
        # Movement options
        directions = ['up', 'down', 'left', 'right']
        for direction in directions:
            new_pos = {
                'up':    (max(0, x-1), y),
                'down':  (min(environment.size-1, x+1), y),
                'left':  (x, max(0, y-1)),
                'right': (x, min(environment.size-1, y+1))
            }[direction]
            if new_pos not in occupied and new_pos != self.position:
                actions.append(f"move {direction}")
        
        # Collection option
        if cell and cell in self.inventory:
            actions.append("collect")
        
        # Eating options
        if self.inventory['red'] > 0:
            actions.append("eat red")
        if self.inventory['green'] > 0:
            actions.append("eat green")
        
        # Do nothing option
        actions.append("do nothing")
        
        # Randomly choose action
        action = random.choice(actions)
        self.actions_taken.append(action)
        
        # Execute the action
        if action.startswith("move"):
            direction = action.split()[1]
            new_pos = {
                'up':    (max(0, x-1), y),
                'down':  (min(environment.size-1, x+1), y),
                'left':  (x, max(0, y-1)),
                'right': (x, min(environment.size-1, y+1))
            }[direction]
            self.position = new_pos
            result = f"moved {direction} (energy: {self.energy})"
            
        elif action == "collect":
            item = environment.get_cell_content(x, y)
            self.inventory[item] += 1
            environment.clear_cell(x, y)
            result = f"collected {item}"
            
        elif action == "eat red":
            self.inventory['red'] -= 1
            gain = self.consumption_rates['red']
            self.energy += gain
            result = f"ate red (+{gain})"
            
        elif action == "eat green":
            self.inventory['green'] -= 1
            gain = self.consumption_rates['green']
            self.energy += gain
            result = f"ate green (+{gain})"
            
        else:  # "do nothing"
            result = "did nothing"
        
        # Record memory
        obs = self.get_current_observation(environment, [])
        self.add_memory(obs, action, result)
        
        return result

    def consider_trade_offer(self, nearby_agents, step):
        """
        Decide whether to make a trade offer to nearby agents
        Returns TradeOffer if agent wants to trade, None otherwise
        """
        import config
        from trading import TradeOffer
        
        if config.TRADE_RANGE <= 0 or not nearby_agents:
            return None
        
        # Random mode: make random 2-1 or 1-2 offers
        if config.RAND_MODE == 1:
            return self._random_trade_offer(nearby_agents, step)
        
        # Normal LLM-based logic
        red_inventory = self.inventory['red']
        green_inventory = self.inventory['green']
        red_rate = self.consumption_rates['red']
        green_rate = self.consumption_rates['green']
        
        # Determine what we want based on our consumption preferences
        if red_rate > green_rate and green_inventory > 0:
            # We prefer red, offer some green for red
            target_agent = random.choice(nearby_agents)
            return TradeOffer(
                offerer=self.name,
                target=target_agent.name,
                offer_red=0,
                offer_green=min(1, green_inventory),
                request_red=1,
                request_green=0,
                step=step
            )
        elif green_rate > red_rate and red_inventory > 0:
            # We prefer green, offer some red for green
            target_agent = random.choice(nearby_agents)
            return TradeOffer(
                offerer=self.name,
                target=target_agent.name,
                offer_red=min(1, red_inventory),
                offer_green=0,
                request_red=0,
                request_green=1,
                step=step
            )
        
        return None

    def _random_trade_offer(self, nearby_agents, step):
        """Generate random 2-1 or 1-2 trade offers"""
        from trading import TradeOffer
        
        # Only make offer if we have at least 2 of some resource
        total_red = self.inventory['red']
        total_green = self.inventory['green']
        
        if total_red < 1 and total_green < 1:
            return None
        
        target_agent = random.choice(nearby_agents)
        
        # Randomly choose offer type: 2-1 or 1-2
        offer_types = []
        
        # 2 red for 1 green
        if total_red >= 2:
            offer_types.append(('2R_for_1G', 2, 0, 0, 1))
        
        # 2 green for 1 red  
        if total_green >= 2:
            offer_types.append(('2G_for_1R', 0, 2, 1, 0))
        
        # 1 red for 2 green
        if total_red >= 1:
            offer_types.append(('1R_for_2G', 1, 0, 0, 2))
        
        # 1 green for 2 red
        if total_green >= 1:
            offer_types.append(('1G_for_2R', 0, 1, 2, 0))
        
        if not offer_types:
            return None
        
        # Randomly select an offer type
        offer_type, offer_red, offer_green, request_red, request_green = random.choice(offer_types)
        
        return TradeOffer(
            offerer=self.name,
            target=target_agent.name,
            offer_red=offer_red,
            offer_green=offer_green,
            request_red=request_red,
            request_green=request_green,
            step=step
        )

    def consider_trade_acceptance(self, offers, step):
        """
        Decide which trade offer (if any) to accept
        Returns TradeOffer to accept, or None to reject all
        """
        if not offers:
            return None
        
        import config
        
        # Random mode: randomly accept offers (50% chance)
        if config.RAND_MODE == 1:
            for offer in offers:
                # Check if we have the resources they want
                if (self.inventory['red'] >= offer.request_red and 
                    self.inventory['green'] >= offer.request_green):
                    # 50% chance to accept any valid offer
                    if random.random() < 0.5:
                        return offer
            return None
        
        # Normal LLM-based logic
        red_rate = self.consumption_rates['red']
        green_rate = self.consumption_rates['green']
        
        for offer in offers:
            # Check if we have the resources they want
            if (self.inventory['red'] >= offer.request_red and 
                self.inventory['green'] >= offer.request_green):
                
                # Check if we want what they're offering
                if red_rate > green_rate and offer.offer_red > 0:
                    return offer
                elif green_rate > red_rate and offer.offer_green > 0:
                    return offer
                elif red_rate == green_rate:  # Balanced agent, accept any reasonable offer
                    return offer
        
        return None

    def status(self):
        print(f"{self.name} ({self.type}) @ {self.position} | "
              f"E={self.energy} | Inv={self.inventory} | Alive={self.alive}")

    def get_status_dict(self):
        return {
            'name': self.name,
            'type': self.type,
            'position': self.position,
            'inventory': self.inventory.copy(),
            'energy': self.energy,
            'alive': self.alive,
            'recent_actions': self.actions_taken[-5:],
            'recent_memory': self.memory[-AGENT_MEMORY_SIZE:]
        }
