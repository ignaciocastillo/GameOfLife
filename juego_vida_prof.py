import numpy as np
from numba import njit, prange

@njit(parallel=True)
def step_numba(grid):
    rows, cols = grid.shape
    new_grid = np.zeros((rows, cols), dtype=np.int32)
    for i in prange(rows):
        for j in prange(cols):
            total = 0
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    if x == 0 and y == 0:
                        continue
                    ni, nj = i + x, j + y
                    if 0 <= ni < rows and 0 <= nj < cols:
                        total += grid[ni, nj]
            if grid[i, j] == 1 and total in (2, 3):
                new_grid[i, j] = 1
            elif grid[i, j] == 0 and total == 3:
                new_grid[i, j] = 1
    return new_grid

class GameOfLifeFast:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = np.random.choice([0, 1], size=(rows, cols)).astype(np.int32)

    def step(self):
        self.grid = step_numba(self.grid)

    def run(self, steps=100):
        for _ in range(steps):
            self.step()

@profile
def ejecutar_juego():
    game = GameOfLifeFast(512, 512)
    game.run(100)

if __name__ == "__main__":
    ejecutar_juego()
