import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class DependencyResolver:
    def __init__(self):
        self.cache = {}

    def resolve(self, methods: dict) -> list:
        """
        Resolve the order of timeseries generation.

        Parameters:
        - methods (dict): A dictionary mapping timeseries types to their methods.

        Returns:
        - list: A list of timeseries types in the order they should be generated.
        """
        if not methods:
            logger.debug("No methods provided for dependency resolution.")
            return []

        methods_key = frozenset(methods.keys())
        if methods_key in self.cache:
            logger.debug(f"Using cached dependency order for methods: {methods_key}")
            return self.cache[methods_key]

        logger.debug(f"Resolving dependencies for methods: {methods_key}")
        resolved_order = self._resolve_dependencies(methods)
        self.cache[methods_key] = resolved_order
        return resolved_order

    @staticmethod
    def _resolve_dependencies(methods: dict) -> list:
        """
        Resolve the order of timeseries generation, supporting multilevel dependencies.

        Parameters:
        - methods (dict): A dictionary mapping timeseries types to their methods.

        Returns:
        - list: A list of timeseries types in the order they should be generated.

        Raises:
        - ValueError: If circular dependencies are detected.
        """
        # Build the dependency graph
        graph = {ts_type: set(method.dependencies) for ts_type, method in methods.items()}

        resolved = []  # Resolved nodes in topological order
        visiting = set()  # Nodes in the current recursive path
        visited = set()  # Nodes already resolved

        def visit(node):
            logger.debug(f"Visiting node: {node}")
            if node in visiting:
                raise ValueError(f"Circular dependency detected: {node}")
            if node not in visited:
                visiting.add(node)
                for dep in graph[node]:
                    if dep not in methods:
                        raise ValueError(f"Unknown dependency '{dep}' for timeseries '{node}'. "
                                         f"Ensure all dependencies are declared.")
                    visit(dep)
                visiting.remove(node)
                visited.add(node)
                resolved.append(node)
                logger.debug(f"Resolved node: {node}")

        for ts_type in methods:
            if ts_type not in visited:
                visit(ts_type)

        return resolved

    @staticmethod
    def visualize_dependencies(methods: dict):
        """
        Visualize the dependency graph.

        Parameters:
        - methods (dict): A dictionary mapping timeseries types to their methods.
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("Visualization requires 'networkx' and 'matplotlib' to be installed.") from e

        graph = defaultdict(list)
        for ts_type, method in methods.items():
            for dep in method.dependencies:
                graph[dep].append(ts_type)

        G = nx.DiGraph(graph)
        nx.draw(G, with_labels=True, node_color="lightblue", font_size=10, node_size=3000, edge_color="gray")
        plt.title("Timeseries Dependency Graph")
        plt.show()
