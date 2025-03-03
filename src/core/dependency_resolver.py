import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class DependencyResolver:
    """
    A class to resolve the order of timeseries generation based on method dependencies.

    This class supports dependency resolution, circular dependency detection,
    caching for performance optimization, and optional visualization of the
    dependency graph.
    """

    def __init__(self):
        """
        Initializes the DependencyResolver.

        Attributes:
            cache (dict): Cache to store previously resolved dependency orders.
        """
        self.cache = {}

    def resolve(self, methods: dict) -> list:
        """
        Resolve the order of timeseries generation.

        Args:
            methods (dict): A dictionary mapping timeseries types to their methods.
                Each method is expected to have a `dependencies` attribute.

        Returns:
            list: A list of timeseries types in the order they should be generated.

        Raises:
            ValueError: If circular dependencies or undefined dependencies are encountered.

        Example:
            >>> methods = {
            ...     "type_a": MethodA(dependencies=["type_b"]),
            ...     "type_b": MethodB(dependencies=[]),
            ... }
            >>> resolver = DependencyResolver()
            >>> resolver.resolve(methods)
            ['type_b', 'type_a']
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

        Args:
            methods (dict): A dictionary mapping timeseries types to their methods.
                Each method must have a `dependencies` attribute.

        Returns:
            list: A list of timeseries types in the topologically sorted order.

        Raises:
            ValueError: If circular dependencies are detected or if any dependency is undefined.

        Example:
            >>> methods = {
            ...     "type_a": MethodA(dependencies=["type_b"]),
            ...     "type_b": MethodB(dependencies=[]),
            ... }
            >>> DependencyResolver._resolve_dependencies(methods)
            ['type_b', 'type_a']
        """
        graph = {ts_type: set(method.dependencies) for ts_type, method in methods.items()}
        resolved = []  # Topologically sorted nodes
        visiting = set()  # Nodes in the current recursive path
        visited = set()  # Nodes already resolved

        def visit(node):
            """
            Recursively visit nodes to perform a depth-first traversal of the graph.

            Args:
                node (str): The current node being visited.

            Raises:
                ValueError: If circular dependencies or undefined dependencies are found.
            """
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
        Visualize the dependency graph for the provided methods.

        Args:
            methods (dict): A dictionary mapping timeseries types to their methods.
                Each method must have a `dependencies` attribute.

        Raises:
            ImportError: If `networkx` or `matplotlib` are not installed.

        Example:
            >>> methods = {
            ...     "type_a": MethodA(dependencies=["type_b"]),
            ...     "type_b": MethodB(dependencies=[]),
            ... }
            >>> DependencyResolver.visualize_dependencies(methods)

        Note:
            This method generates a directed graph showing the dependencies using
            `networkx` and `matplotlib`. Nodes represent timeseries types, and edges
            indicate dependencies.
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError("Visualization requires 'networkx' and 'matplotlib' to be installed.") from e

        # Build the graph representation
        graph = defaultdict(list)
        for ts_type, method in methods.items():
            for dep in method.dependencies:
                graph[dep].append(ts_type)

        G = nx.DiGraph(graph)
        nx.draw(G, with_labels=True, node_color="lightblue", font_size=10, node_size=3000, edge_color="gray")
        plt.title("Timeseries Dependency Graph")
        plt.show()
