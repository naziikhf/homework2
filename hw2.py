
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from collections import deque

class State(ABC):
    @abstractmethod
    def __hash__(self) -> int: ...
    @abstractmethod
    def __eq__(self, other: object) -> bool: ...
    @abstractmethod
    def __str__(self) -> str: ...

class Problem(ABC):
    @abstractmethod
    def get_initial_state(self) -> State: ...
    @abstractmethod
    def get_goal_state(self) -> State: ...
    @abstractmethod
    def is_goal_state(self, state: State) -> bool: ...
    @abstractmethod
    def get_successors(self, state: State) -> List[Tuple[Any, State]]: ...

    def get_cost(self, state1: State, action: Any, state2: State) -> float:
        return 1.0

@dataclass(frozen=True, slots=True)
class TowerOfHanoiState(State):
    pegs: Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]

    def __str__(self) -> str:
        return f"TowerOfHanoiState(pegs={self.pegs})"

    def can_move(self, from_peg: int, to_peg: int) -> bool:
        if from_peg == to_peg:
            return False
        if not (0 <= from_peg < 3 and 0 <= to_peg < 3):
            return False
        src = self.pegs[from_peg]
        dst = self.pegs[to_peg]
        if len(src) == 0:
            return False
        if len(dst) == 0:
            return True
        return src[-1] < dst[-1]

    def move_disk(self, from_peg: int, to_peg: int) -> "TowerOfHanoiState":
        if not self.can_move(from_peg, to_peg):
            raise ValueError("Invalid move")

        pegs_list = [list(p) for p in self.pegs]
        disk = pegs_list[from_peg].pop()
        pegs_list[to_peg].append(disk)
        new_pegs = (tuple(pegs_list[0]), tuple(pegs_list[1]), tuple(pegs_list[2]))
        return TowerOfHanoiState(new_pegs)

class TowerOfHanoiProblem(Problem):
    def __init__(self, num_disks: int, start_peg: int = 0, goal_peg: int = 2):
        if num_disks <= 0:
            raise ValueError("num_disks must be positive")
        if not (0 <= start_peg < 3 and 0 <= goal_peg < 3) or start_peg == goal_peg:
            raise ValueError("Invalid start/goal peg")

        self.num_disks = num_disks
        self.start_peg = start_peg
        self.goal_peg = goal_peg

        pegs = [[], [], []]
        pegs[start_peg] = list(range(num_disks, 0, -1))
        self._initial_state = TowerOfHanoiState((tuple(pegs[0]), tuple(pegs[1]), tuple(pegs[2])))

        gpegs = [[], [], []]
        gpegs[goal_peg] = list(range(num_disks, 0, -1))
        self._goal_state = TowerOfHanoiState((tuple(gpegs[0]), tuple(gpegs[1]), tuple(gpegs[2])))

    def get_initial_state(self) -> TowerOfHanoiState:
        return self._initial_state

    def get_goal_state(self) -> TowerOfHanoiState:
        return self._goal_state

    def is_goal_state(self, state: State) -> bool:
        return isinstance(state, TowerOfHanoiState) and state == self._goal_state

    def get_successors(self, state: State) -> List[Tuple[Tuple[int, int], TowerOfHanoiState]]:
        if not isinstance(state, TowerOfHanoiState):
            raise TypeError("state must be TowerOfHanoiState")

        out: List[Tuple[Tuple[int, int], TowerOfHanoiState]] = []
        for i in range(3):
            for j in range(3):
                if state.can_move(i, j):
                    out.append(((i, j), state.move_disk(i, j)))
        return out

    def generate_state_space(
        self,
        max_states: Optional[int] = None
    ) -> Tuple[Set[TowerOfHanoiState], Dict[TowerOfHanoiState, List[Tuple[Tuple[int, int], TowerOfHanoiState]]]]:
        start = self.get_initial_state()
        visited: Set[TowerOfHanoiState] = set([start])
        edges: Dict[TowerOfHanoiState, List[Tuple[Tuple[int, int], TowerOfHanoiState]]] = {}
        q = deque([start])

        while q:
            s = q.popleft()
            succ = self.get_successors(s)
            edges[s] = succ

            for action, ns in succ:
                if ns not in visited:
                    visited.add(ns)
                    if max_states is not None and len(visited) >= max_states:
                        return visited, edges
                    q.append(ns)

        return visited, edges