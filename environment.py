import random
import config

FOOD_TYPES = ['red', 'green', None]

class Environment:
    def __init__(self):
        self.size = config.GRID_SIZE
        self.grid = self._generate_grid()

    def _generate_grid(self):
        return [[random.choice(FOOD_TYPES) for _ in range(self.size)] for _ in range(self.size)]

    def get_cell_content(self, x, y):
        return self.grid[x][y]

    def clear_cell(self, x, y):
        self.grid[x][y] = None

    def count_food(self):
        """Count total food in the environment"""
        red_count = 0
        green_count = 0
        for row in self.grid:
            for cell in row:
                if cell == 'red':
                    red_count += 1
                elif cell == 'green':
                    green_count += 1
        return {'red': red_count, 'green': green_count}

    def print_grid(self, agent_positions=[]):
        agent_map = {agent.position: agent.name[-1] for agent in agent_positions}  # A1, A2...

        for i in range(self.size):
            row = ""
            for j in range(self.size):
                cell = self.grid[i][j]
                symbol = '.'
                if cell == 'red':
                    symbol = 'R'
                elif cell == 'green':
                    symbol = 'G'
                if (i, j) in agent_map:
                    symbol = f"A{agent_map[(i, j)]}"
                row += f"{symbol} "
            print(row)
        print()

    def fixed_replenish(self, red_count=5, green_count=5):
        """Replenish exactly red_count red and green_count green foods randomly."""
        empty_cells = [(x, y) for x in range(self.size) for y in range(self.size) if self.grid[x][y] is None]
        random.shuffle(empty_cells)

        for _ in range(red_count):
            if empty_cells:
                x, y = empty_cells.pop()
                self.grid[x][y] = 'red'
        for _ in range(green_count):
            if empty_cells:
                x, y = empty_cells.pop()
                self.grid[x][y] = 'green'