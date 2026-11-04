"""Q-learning tabular na grade 4x4 do Exercício 6 do ED3 (EEE048)."""

import sys
from dataclasses import dataclass
from enum import Enum

Cell = tuple[int, int]


class Action(Enum):
    """Ações do rover, declaradas na ordem de desempate do enunciado."""

    UP = ("↑", 0, 1)
    RIGHT = ("→", 1, 0)
    DOWN = ("↓", 0, -1)
    LEFT = ("←", -1, 0)

    def __init__(self, symbol: str, d_col: int, d_row: int) -> None:
        self.symbol = symbol
        self.delta = (d_col, d_row)


class GridWorld:
    """Grade quadrada cuja recompensa é definida pela célula de destino.

    As células são pares (coluna, linha), com a linha 1 na base da figura.
    """

    def __init__(
        self,
        size: int,
        rewards: dict[Cell, float],
        default_reward: float,
        terminal: Cell,
    ) -> None:
        self.size = size
        self.rewards = rewards
        self.default_reward = default_reward
        self.terminal = terminal

    def cells(self) -> list[Cell]:
        """Devolve todas as células da grade."""
        return [(col, row) for col in range(1, self.size + 1) for row in range(1, self.size + 1)]

    def is_terminal(self, cell: Cell) -> bool:
        """Diz se a célula é o estado terminal."""
        return cell == self.terminal

    def move(self, cell: Cell, action: Action) -> Cell:
        """Devolve a célula alcançada por uma ação, sem checar os limites."""
        return (cell[0] + action.delta[0], cell[1] + action.delta[1])

    def actions(self, cell: Cell) -> list[Action]:
        """Lista as ações disponíveis, na ordem de desempate.

        Ação que levaria para fora da grade não está disponível, e o estado
        terminal não tem ação nenhuma.
        """
        if self.is_terminal(cell):
            return []
        return [a for a in Action if self._inside(self.move(cell, a))]

    def step(self, cell: Cell, action: Action) -> tuple[Cell, float]:
        """Aplica uma ação e devolve a nova célula e a recompensa recebida."""
        next_cell = self.move(cell, action)
        return next_cell, self.rewards.get(next_cell, self.default_reward)

    def _inside(self, cell: Cell) -> bool:
        return all(1 <= coord <= self.size for coord in cell)


@dataclass(frozen=True)
class Step:
    """Uma transição vivida no episódio e a atualização que ela causou."""

    cell: Cell
    action: Action
    reward: float
    next_cell: Cell
    q_before: float
    q_after: float


@dataclass(frozen=True)
class Episode:
    """Sequência de passos de um episódio, a partir de uma célula inicial."""

    start: Cell
    steps: list[Step]

    @property
    def path(self) -> list[Cell]:
        """Células visitadas, da inicial até a terminal."""
        return [self.start] + [s.next_cell for s in self.steps]

    @property
    def total_reward(self) -> float:
        """Soma das recompensas do episódio, sem desconto."""
        return sum(s.reward for s in self.steps)


class QLearning:
    """Q-learning tabular com política puramente gananciosa nos episódios."""

    def __init__(self, world: GridWorld, alpha: float, gamma: float) -> None:
        self.world = world
        self.alpha = alpha
        self.gamma = gamma
        self.q: dict[tuple[Cell, Action], float] = {
            (cell, action): 0.0
            for cell in world.cells()
            for action in world.actions(cell)
        }

    def best_value(self, cell: Cell) -> float:
        """Devolve o máximo de Q sobre as ações da célula."""
        # O estado terminal não tem ações e seu valor futuro é zero.
        return max((self.q[cell, a] for a in self.world.actions(cell)), default=0.0)

    def greedy_action(self, cell: Cell) -> Action:
        """Devolve a ação de maior Q; o max fica com o primeiro empate."""
        return max(self.world.actions(cell), key=lambda a: self.q[cell, a])

    def action_probability(self, cell: Cell, action: Action, epsilon: float) -> float:
        """Probabilidade de escolher a ação sob ε-greedy.

        Com probabilidade ε sorteia-se uma ação entre todas as disponíveis,
        inclusive a gananciosa.
        """
        probability = epsilon / len(self.world.actions(cell))
        if action == self.greedy_action(cell):
            probability += 1 - epsilon
        return probability

    def run_episode(self, start: Cell, max_steps: int = 10_000) -> Episode:
        """Roda um episódio guloso atualizando Q a cada transição.

        Raises:
            RuntimeError: se o episódio passar de max_steps sem terminar.
        """
        steps: list[Step] = []
        cell = start
        while not self.world.is_terminal(cell):
            if len(steps) >= max_steps:
                raise RuntimeError(f"Episódio passou de {max_steps} passos sem terminar.")
            action = self.greedy_action(cell)
            next_cell, reward = self.world.step(cell, action)
            q_before = self.q[cell, action]
            target = reward + self.gamma * self.best_value(next_cell)
            self.q[cell, action] = (1 - self.alpha) * q_before + self.alpha * target
            steps.append(Step(cell, action, reward, next_cell, q_before, self.q[cell, action]))
            cell = next_cell
        return Episode(start, steps)


def build_world() -> GridWorld:
    """Monta a grade da figura do enunciado."""
    radiation, mud, exit_cell = -20.0, -5.0, 20.0
    rewards = {
        (2, 4): radiation,
        (3, 1): radiation,
        (2, 3): mud,
        (4, 2): mud,
        (4, 4): exit_cell,
    }
    return GridWorld(size=4, rewards=rewards, default_reward=-1.0, terminal=(4, 4))


def format_cell(cell: Cell) -> str:
    """Escreve a célula como (coluna,linha)."""
    return f"({cell[0]},{cell[1]})"


def format_q_table(learner: QLearning, q: dict[tuple[Cell, Action], float]) -> str:
    """Escreve a tabela Q com uma linha por célula e uma coluna por ação."""
    header = "célula  " + "  ".join(f"{a.symbol:>8}" for a in Action)
    lines = [header]
    for cell in sorted(learner.world.cells()):
        available = learner.world.actions(cell)
        values = [f"{q[cell, a]:8.4f}" if a in available else f"{'-':>8}" for a in Action]
        lines.append(f"{format_cell(cell):<7} " + "  ".join(values))
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    learner = QLearning(build_world(), alpha=0.2, gamma=0.95)
    starts = [(2, 2), (3, 1), (1, 1), (1, 2), (1, 4), (1, 4), (2, 2), (3, 1), (1, 1)]
    q_before_ninth: dict[tuple[Cell, Action], float] = {}

    for number, start in enumerate(starts, start=1):
        if number == 9:
            q_before_ninth = dict(learner.q)
        episode = learner.run_episode(start)
        print(f"Episódio {number}: {len(episode.steps)} passos, "
              f"recompensa total {episode.total_reward:g}")
        print("  trajeto: " + " ".join(format_cell(c) for c in episode.path))
        if number == 1:
            for s in episode.steps:
                print(f"  {format_cell(s.cell)} {s.action.symbol} r={s.reward:g} "
                      f"-> {format_cell(s.next_cell)}  Q: {s.q_before:.4f} -> {s.q_after:.4f}")

    print("\nTabela Q no início do 9º episódio:")
    print(format_q_table(learner, q_before_ninth))

    cell, action = (4, 3), Action.DOWN
    probability = learner.action_probability(cell, action, epsilon=0.1)
    greedy = learner.greedy_action(cell)
    print(f"\nAções em {format_cell(cell)}: {' '.join(a.symbol for a in learner.world.actions(cell))}; "
          f"gananciosa: {greedy.symbol}")
    print(f"P({action.symbol} em {format_cell(cell)}) com ε = 0,1: {probability:.4f}")


if __name__ == "__main__":
    main()
