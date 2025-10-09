from __future__ import annotations
import contextlib
from typing import Any, Callable, Dict, Set


class ReactiveGraph:
    """A minimal Pixie-style reactive graph with automatic dependency tracking and lazy recomputation."""

    _active_node: str | None = None       # Which node is currently being evaluated
    _edges: Dict[str, Set[str]] = {}      # dep -> set of dependents (the graph structure)

    def __init__(self):
        self._cache: Dict[str, Any] = {}       # node -> cached value
        self._dirty: Set[str] = set()          # nodes marked dirty
        self._overrides: Dict[str, Any] = {}   # manually set values

    # ──────────────────────────────
    #  Registration & decorator
    # ──────────────────────────────
    @staticmethod
    def reactive(fn: Callable) -> Callable:
        """Decorator for reactive methods."""
        name = fn.__name__

        def wrapper(self: ReactiveGraph, *args, **kwargs):
            self._register_dependency(name)
            # Check manual override
            if name in self._overrides:
                return self._overrides[name]
            # Return cached if valid
            if name in self._cache and name not in self._dirty:
                return self._cache[name]
            # Compute under recording context
            with self._recording(name):
                value = fn(self, *args, **kwargs)
            self._cache[name] = value
            self._dirty.discard(name)
            return value

        return wrapper

    # ──────────────────────────────
    #  Dependency tracking
    # ──────────────────────────────
    @contextlib.contextmanager
    def _recording(self, name: str):
        prev = ReactiveGraph._active_node
        ReactiveGraph._active_node = name
        try:
            yield
        finally:
            ReactiveGraph._active_node = prev

    def _register_dependency(self, dep: str):
        cur = ReactiveGraph._active_node
        if cur and cur != dep:
            ReactiveGraph._edges.setdefault(dep, set()).add(cur)

    # ──────────────────────────────
    #  Invalidation / mutation
    # ──────────────────────────────
    def set_value(self, name: str, value: Any):
        """Manually tweak a node's value."""
        self._overrides[name] = value
        self.invalidate(name)

    def invalidate(self, name: str):
        """Mark a node and its dependents dirty."""
        to_invalidate = {name}
        visited = set()
        while to_invalidate:
            n = to_invalidate.pop()
            if n in visited:
                continue
            visited.add(n)
            self._dirty.add(n)
            self._cache.pop(n, None)
            for dep in ReactiveGraph._edges.get(n, set()):
                to_invalidate.add(dep)

    # ──────────────────────────────
    #  Debugging / inspection
    # ──────────────────────────────
    def show_graph(self):
        """Print dependencies in dep -> dependents form."""
        for dep, deps in ReactiveGraph._edges.items():
            print(f"{dep} → {', '.join(deps)}")


# class Portfolio(ReactiveGraph):
#     @ReactiveGraph.reactive
#     def price_A(self):
#         return 100

#     @ReactiveGraph.reactive
#     def price_B(self):
#         return 200

#     @ReactiveGraph.reactive
#     def portfolio_value(self):
#         return self.price_A() + self.price_B()

#     @ReactiveGraph.reactive
#     def pnl(self):
#         return self.portfolio_value() - 250
# p = Portfolio()

# print("Initial PnL:", p.pnl())  # 50

# p.set_value("price_A", 120)     # tweak node
# print("Graph dependencies:")
# p.show_graph()

# print("New PnL:", p.pnl())      # 70


# A multithreaded version

# import threading
# import contextlib
# from typing import Any, Callable, Dict, Set


# class ReactiveGraph:
#     """Thread-safe reactive graph with automatic dependency tracking and lazy recomputation."""

#     _thread_ctx = threading.local()  # holds per-thread state (active node name)

#     def __init__(self):
#         # Instance-level state
#         self._edges: Dict[str, Set[str]] = {}  # dep -> dependents
#         self._cache: Dict[str, Any] = {}       # node -> cached value
#         self._dirty: Set[str] = set()          # dirty nodes
#         self._overrides: Dict[str, Any] = {}   # manually overridden values

#     # ──────────────────────────────
#     #  Decorator for reactive nodes
#     # ──────────────────────────────
#     @staticmethod
#     def reactive(fn: Callable) -> Callable:
#         name = fn.__name__

#         def wrapper(self: "ReactiveGraph", *args, **kwargs):
#             self._register_dependency(name)

#             # Manual override takes priority
#             if name in self._overrides:
#                 return self._overrides[name]

#             # Return cached if valid
#             if name in self._cache and name not in self._dirty:
#                 return self._cache[name]

#             # Compute under active context
#             with self._recording(name):
#                 value = fn(self, *args, **kwargs)

#             # Store and mark clean
#             self._cache[name] = value
#             self._dirty.discard(name)
#             return value

#         return wrapper

#     # ──────────────────────────────
#     #  Dependency tracking
#     # ──────────────────────────────
#     @contextlib.contextmanager
#     def _recording(self, name: str):
#         """Temporarily mark which node is being computed in this thread."""
#         prev = getattr(self._thread_ctx, "active_node", None)
#         self._thread_ctx.active_node = name
#         try:
#             yield
#         finally:
#             self._thread_ctx.active_node = prev

#     def _register_dependency(self, dep: str):
#         """Record an edge dep → active_node if inside another computation."""
#         cur = getattr(self._thread_ctx, "active_node", None)
#         if cur and cur != dep:
#             self._edges.setdefault(dep, set()).add(cur)

#     # ──────────────────────────────
#     #  Invalidation / mutation
#     # ──────────────────────────────
#     def set_value(self, name: str, value: Any):
#         """Manually tweak a node's value and invalidate downstream."""
#         self._overrides[name] = value
#         self.invalidate(name)

#     def invalidate(self, name: str):
#         """Mark node and dependents dirty."""
#         to_invalidate = {name}
#         visited = set()
#         while to_invalidate:
#             n = to_invalidate.pop()
#             if n in visited:
#                 continue
#             visited.add(n)
#             self._dirty.add(n)
#             self._cache.pop(n, None)
#             for dep in self._edges.get(n, set()):
#                 to_invalidate.add(dep)

#     # ──────────────────────────────
#     #  Debug / inspection
#     # ──────────────────────────────
#     def show_graph(self):
#         for dep, deps in self._edges.items():
#             print(f"{dep} → {', '.join(deps)}")

