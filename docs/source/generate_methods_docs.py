import os
import inspect
from jinja2 import Template
import pkgutil
import importlib
import inspect
from entise.core.base import Method

TEMPLATE_PATH = "./_templates/method.rst"
BASE_OUTPUT_DIR = "./methods"  # Base folder for output


def discover_method_classes(package_name):
    """
    Discover and return a list of classes within the given package that are
    subclasses of Method.

    Parameters:
      package_name (str): The package to search (e.g., "entise.methods").

    Returns:
      list: A list of discovered method classes.
    """
    discovered_classes = []
    package = importlib.import_module(package_name)
    package_path = package.__path__

    # Walk through all modules in the package
    for _, module_name, is_pkg in pkgutil.walk_packages(package_path, prefix=package.__name__ + '.'):
        if not is_pkg:
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                print(f"Could not import module {module_name}: {e}")
                continue
            # Iterate over classes defined in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Ensure the class is defined in this package and is a subclass of Method.
                if issubclass(obj, Method) and obj.__module__.startswith(package_name):
                    discovered_classes.append(obj)
    return discovered_classes


def extract_method_metadata(cls):
    """
    Extract metadata from a timeseries method class.

    Parameters:
      - cls: The method class to analyze.

    Returns:
      dict: Metadata including description, required keys, and required timeseries.
    """
    required_keys = getattr(cls, "required_keys", {})
    required_timeseries = getattr(cls, "required_timeseries", {})
    dependencies = getattr(cls, "dependencies", [])
    docstring = inspect.getdoc(cls)
    # Extract method docstrings for each function in the class
    methods = {name: inspect.getdoc(meth) for name, meth in inspect.getmembers(cls, predicate=inspect.isfunction)}
    return {
        "description": docstring,
        "required_keys": required_keys,
        "required_timeseries": required_timeseries,
        "dependencies": dependencies,
        "methods": methods,
    }


def generate_docs_for_method(cls, timeseries_type, template_path, output_dir):
    """
    Generate documentation for a timeseries method using a Jinja2 template.

    Parameters:
      - cls: The method class to document.
      - timeseries_type: The timeseries type (e.g., "HVAC", "Electricity").
      - template_path: Path to the Jinja2 template.
      - output_dir: Directory to save the generated documentation.
    """
    metadata = extract_method_metadata(cls)

    # Load the Jinja2 template
    with open(template_path, "r") as f:
        template = Template(f.read())

    # Add 'hasattr' to the template's globals so it's available during rendering.
    template.globals['hasattr'] = hasattr

    # Render the template with the metadata and provided timeseries type.
    rendered = template.render(
        method_name=cls.__name__,
        description=metadata["description"],
        required_keys=metadata["required_keys"],
        required_timeseries=metadata["required_timeseries"],
        dependencies=metadata["dependencies"],
        timeseries_type=timeseries_type,
        methods=metadata["methods"],
    )

    # Save the rendered content to a file in the output directory.
    output_file = os.path.join(output_dir, f"{cls.__name__.lower()}.rst")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(rendered)
    # print(f"Documentation for {cls.__name__} written to {output_file}")


def generate_docs_for_all_methods(method_classes, template_path, base_output_dir):
    """
    Loop over all provided method classes and generate documentation pages in the correct folder(s)
    based on the timeseries types they support.

    If a method supports multiple types, it is generated in each corresponding folder.
    """
    for cls in method_classes:
        # Assume each method class has a 'types' attribute listing its valid types.
        supported_types = getattr(cls, "types", None)
        if supported_types is None:
            supported_types = ["HVAC"]  # Fallback if not specified

        for timeseries_type in supported_types:
            # Determine the output folder based on the timeseries type.
            out_dir = os.path.join(base_output_dir, timeseries_type.lower())
            os.makedirs(out_dir, exist_ok=True)
            generate_docs_for_method(cls, timeseries_type, template_path, out_dir)


def generate_index_for_folder(folder_path):
    """
    Generate an index.rst file in the given folder_path that lists all .rst files (except index.rst).
    """
    # List all .rst files in the folder, excluding index.rst
    rst_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".rst") and f.lower() != "index.rst"
    ])

    if not rst_files:
        return

    # Create a title from the folder name (e.g., "hvac" -> "HVAC Methods")
    folder_name = os.path.basename(folder_path)
    title = f"{folder_name.upper()} Methods"
    underline = "=" * len(title)
    # Build the content with a toctree directive
    content_lines = [
        title,
        underline,
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "",
    ]
    for rst_file in rst_files:
        content_lines.append(f"   {rst_file}")

    content = "\n".join(content_lines)

    # Write the content to index.rst in the folder
    output_file = os.path.join(folder_path, "index.rst")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)
    # print(f"Created index file: {output_file}")


def generate_indexes(base_dir):
    """
    Walk through each subdirectory in base_dir and generate an index.rst file.
    """
    for entry in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, entry)
        if os.path.isdir(subdir_path):
            generate_index_for_folder(subdir_path)


if __name__ == "__main__":
    # Automatically discover all method classes in the package.
    method_classes = discover_method_classes("entise.methods")
    # Generate documentation for each method in the appropriate folder(s).
    generate_docs_for_all_methods(method_classes, TEMPLATE_PATH, BASE_OUTPUT_DIR)
    # Generate an index.rst in each subfolder under BASE_OUTPUT_DIR.
    generate_indexes(BASE_OUTPUT_DIR)
