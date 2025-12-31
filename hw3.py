import heapq
import itertools
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from hw2 import Problem, State, TowerOfHanoiProblem


class Node:
    __slots__ = ("state", "parent", "action", "g", "depth")

    def __init__(self, state: State, parent: Optional["Node"], action: Any, g: float, depth: int):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g
        self.depth = depth


def _reconstruct(goal: Node) -> Tuple[List[Any], List[State], float]:
    actions: List[Any] = []
    states: List[State] = []
    n: Optional[Node] = goal
    while n is not None:
        states.append(n.state)
        actions.append(n.action)
        n = n.parent
    states.reverse()
    actions.reverse()
    if actions and actions[0] is None:
        actions = actions[1:]
    return actions, states, goal.g


def _get_h(problem: Problem, s: State, h: Optional[Callable[[State], float]]) -> float:
    if h is not None:
        return float(h(s))
    if hasattr(problem, "heuristic"):
        return float(getattr(problem, "heuristic")(s))
    return 0.0


def dfs(problem: Problem) -> Tuple[List[Any], List[State], float]:
    start = problem.get_initial_state()
    root = Node(start, None, None, 0.0, 0)
    stack: List[Node] = [root]
    visited: Set[State] = set()

    while stack:
        node = stack.pop()
        if node.state in visited:
            continue
        visited.add(node.state)

        if problem.is_goal_state(node.state):
            return _reconstruct(node)

        succ = problem.get_successors(node.state)
        for action, ns in reversed(succ):
            if ns not in visited:
                child = Node(ns, node, action, node.g + problem.get_cost(node.state, action, ns), node.depth + 1)
                stack.append(child)

    raise ValueError("No solution found")


def bfs(problem: Problem) -> Tuple[List[Any], List[State], float]:
    start = problem.get_initial_state()
    root = Node(start, None, None, 0.0, 0)
    q = deque([root])
    visited: Set[State] = {start}

    while q:
        node = q.popleft()
        if problem.is_goal_state(node.state):
            return _reconstruct(node)

        for action, ns in problem.get_successors(node.state):
            if ns in visited:
                continue
            visited.add(ns)
            child = Node(ns, node, action, node.g + problem.get_cost(node.state, action, ns), node.depth + 1)
            q.append(child)

    raise ValueError("No solution found")


def _dls(problem: Problem, limit: int) -> Optional[Node]:
    start = problem.get_initial_state()
    root = Node(start, None, None, 0.0, 0)
    stack: List[Node] = [root]
    visited_depth: Dict[State, int] = {start: 0}

    while stack:
        node = stack.pop()
        if problem.is_goal_state(node.state):
            return node
        if node.depth == limit:
            continue

        for action, ns in problem.get_successors(node.state):
            nd = node.depth + 1
            prev = visited_depth.get(ns)
            if prev is not None and prev <= nd:
                continue
            visited_depth[ns] = nd
            child = Node(ns, node, action, node.g + problem.get_cost(node.state, action, ns), nd)
            stack.append(child)

    return None


def ids(problem: Problem, max_depth: int = 60) -> Tuple[List[Any], List[State], float]:
    for d in range(max_depth + 1):
        res = _dls(problem, d)
        if res is not None:
            return _reconstruct(res)
    raise ValueError("No solution found")


def ucs(problem: Problem) -> Tuple[List[Any], List[State], float]:
    start = problem.get_initial_state()
    root = Node(start, None, None, 0.0, 0)

    pq: List[Tuple[float, int, Node]] = []
    counter = itertools.count()
    heapq.heappush(pq, (0.0, next(counter), root))

    best_g: Dict[State, float] = {start: 0.0}
    closed: Set[State] = set()

    while pq:
        g, _, node = heapq.heappop(pq)
        if node.state in closed:
            continue
        closed.add(node.state)

        if problem.is_goal_state(node.state):
            return _reconstruct(node)

        for action, ns in problem.get_successors(node.state):
            ng = g + problem.get_cost(node.state, action, ns)
            if ns in closed:
                continue
            old = best_g.get(ns)
            if old is None or ng < old:
                best_g[ns] = ng
                child = Node(ns, node, action, ng, node.depth + 1)
                heapq.heappush(pq, (ng, next(counter), child))

    raise ValueError("No solution found")


def greedy_best_first(problem: Problem, h: Optional[Callable[[State], float]] = None) -> Tuple[List[Any], List[State], float]:
    start = problem.get_initial_state()
    root = Node(start, None, None, 0.0, 0)

    pq: List[Tuple[float, int, Node]] = []
    counter = itertools.count()
    heapq.heappush(pq, (_get_h(problem, start, h), next(counter), root))

    visited: Set[State] = set()

    while pq:
        _, _, node = heapq.heappop(pq)
        if node.state in visited:
            continue
        visited.add(node.state)

        if problem.is_goal_state(node.state):
            return _reconstruct(node)

        for action, ns in problem.get_successors(node.state):
            if ns in visited:
                continue
            child = Node(ns, node, action, node.g + problem.get_cost(node.state, action, ns), node.depth + 1)
            heapq.heappush(pq, (_get_h(problem, ns, h), next(counter), child))

    raise ValueError("No solution found")


def a_star(problem: Problem, h: Optional[Callable[[State], float]] = None) -> Tuple[List[Any], List[State], float]:
    start = problem.get_initial_state()
    root = Node(start, None, None, 0.0, 0)

    pq: List[Tuple[float, int, Node]] = []
    counter = itertools.count()
    heapq.heappush(pq, (_get_h(problem, start, h), next(counter), root))

    best_g: Dict[State, float] = {start: 0.0}
    closed: Set[State] = set()

    while pq:
        f, _, node = heapq.heappop(pq)
        if node.state in closed:
            continue
        closed.add(node.state)

        if problem.is_goal_state(node.state):
            return _reconstruct(node)

        for action, ns in problem.get_successors(node.state):
            ng = node.g + problem.get_cost(node.state, action, ns)
            if ns in closed:
                continue
            old = best_g.get(ns)
            if old is None or ng < old:
                best_g[ns] = ng
                child = Node(ns, node, action, ng, node.depth + 1)
                nf = ng + _get_h(problem, ns, h)
                heapq.heappush(pq, (nf, next(counter), child))

    raise ValueError("No solution found")


def _run_one(name: str, fn: Callable[[], Tuple[List[Any], List[State], float]]) -> None:
    actions, states, cost = fn()
    print(f"{name}: moves={len(actions)}  cost={cost}")


if __name__ == "__main__":
    problem = TowerOfHanoiProblem(num_disks=3)

    _run_one("DFS", lambda: dfs(problem))
    _run_one("BFS", lambda: bfs(problem))
    _run_one("IDS", lambda: ids(problem, max_depth=40))
    _run_one("UCS", lambda: ucs(problem))
    _run_one("GREEDY", lambda: greedy_best_first(problem))
    _run_one("A*", lambda: a_star(problem))
