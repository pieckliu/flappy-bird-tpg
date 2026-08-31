from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
from pathlib import Path

import numpy as np


@dataclass
class Program:
    weights: np.ndarray
    action: int | None = None
    target: int | None = None

    def bid(self, state: np.ndarray) -> float:
        features = np.concatenate([state, state * state, np.ones(1)])
        return float(np.tanh(np.dot(self.weights, features)))


@dataclass
class Team:
    programs: list[Program] = field(default_factory=list)


class TPGAgent:
    def __init__(self, teams: list[Team], root: int = 0):
        self.teams = teams
        self.root = root

    @staticmethod
    def random_agent(rng: np.random.Generator, state_size: int = 5, team_count: int = 4) -> "TPGAgent":
        size = state_size * 2 + 1
        teams = []
        for index in range(team_count):
            programs = [Program(rng.normal(0, 1, size), action=a) for a in (0, 1)]
            if index > 0:
                programs.append(Program(rng.normal(0, 1, size), target=int(rng.integers(0, index))))
            teams.append(Team(programs))
        return TPGAgent(teams)

    def act(self, state: np.ndarray, team_id: int | None = None, visited=None, depth: int = 0) -> int:
        team_id = self.root if team_id is None else team_id
        visited = set() if visited is None else visited
        if depth >= 12 or team_id in visited:
            return 0
        visited.add(team_id)
        winner = max(self.teams[team_id].programs, key=lambda p: p.bid(state))
        if winner.action is not None:
            return winner.action
        return self.act(state, winner.target, visited, depth + 1)

    def mutated(self, rng: np.random.Generator, rate: float = 0.18) -> "TPGAgent":
        child = copy.deepcopy(self)
        for team_index, team in enumerate(child.teams):
            for program in team.programs:
                mask = rng.random(program.weights.size) < rate
                program.weights[mask] += rng.normal(0, 0.45, int(mask.sum()))
                if rng.random() < rate * 0.25:
                    if rng.random() < 0.65:
                        program.action, program.target = int(rng.integers(0, 2)), None
                    else:
                        program.action, program.target = None, int(rng.integers(0, len(child.teams)))
            if rng.random() < rate * 0.2 and len(team.programs) < 7:
                size = team.programs[0].weights.size
                team.programs.append(Program(rng.normal(0, 1, size), action=int(rng.integers(0, 2))))
            if rng.random() < rate * 0.1 and len(team.programs) > 2:
                team.programs.pop(int(rng.integers(0, len(team.programs))))
        child._ensure_actions()
        return child

    def _ensure_actions(self):
        size = self.teams[0].programs[0].weights.size
        for team in self.teams:
            if not any(p.action is not None for p in team.programs):
                team.programs.append(Program(np.zeros(size), action=0))

    def save(self, path: str | Path):
        data = {"root": self.root, "teams": [[{"weights": p.weights.tolist(), "action": p.action, "target": p.target} for p in t.programs] for t in self.teams]}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "TPGAgent":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        teams = [Team([Program(np.array(p["weights"]), p["action"], p["target"]) for p in programs]) for programs in data["teams"]]
        return TPGAgent(teams, data["root"])


def evolve(
    evaluate,
    generations=100,
    population_size=80,
    seed=42,
    on_generation=None,
    on_scored=None,
):
    rng = np.random.default_rng(seed)
    population = [TPGAgent.random_agent(rng) for _ in range(population_size)]
    best, best_fitness = None, float("-inf")
    elite_count = max(2, population_size // 5)
    for generation in range(generations):
        scored = sorted(((evaluate(agent), agent) for agent in population), key=lambda x: x[0], reverse=True)
        if scored[0][0] > best_fitness:
            best_fitness, best = scored[0][0], copy.deepcopy(scored[0][1])
        if on_generation:
            on_generation(generation, scored[0][0], best_fitness, best)
        if on_scored:
            on_scored(generation, scored, best_fitness, best)
        elites = [agent for _, agent in scored[:elite_count]]
        population = [copy.deepcopy(agent) for agent in elites]
        while len(population) < population_size:
            population.append(elites[int(rng.integers(0, len(elites)))].mutated(rng))
    return best, best_fitness
