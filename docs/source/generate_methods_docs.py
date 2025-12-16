import importlib
import inspect
import os
import pkgutil

from jinja2 import Template

from entise.constants import VALID_TYPES
from entise.core.base import Method
from entise.core.base_auxiliary import AuxiliaryMethod

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
    for _, module_name, is_pkg in pkgutil.walk_packages(package_path, prefix=package.__name__ + "."):
        try:
            # Import the module or package
            module = importlib.import_module(module_name)

            # If it's a package, recursively discover classes in its submodules
            if is_pkg:
                # Recursively discover classes in the subpackage
                subpackage_classes = discover_method_classes(module_name)
                discovered_classes.extend(subpackage_classes)

            # Iterate over classes defined in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Ensure the class is defined in this package and is a subclass of Method or AuxiliaryMethod.
                if (
                    (issubclass(obj, Method) or issubclass(obj, AuxiliaryMethod))
                    and obj.__module__.startswith(package_name)
                    and obj is not AuxiliaryMethod  # Exclude the base AuxiliaryMethod class
                    and obj is not Method
                ):  # Exclude the base Method class
                    discovered_classes.append(obj)
        except Exception as e:
            print(f"Could not import module/package {module_name}: {e}")
            continue
    return discovered_classes


def extract_method_metadata(cls):
    """
    Extract metadata from a timeseries method class.

    Parameters:
      - cls: The method class to analyze.

    Returns:
      dict: Metadata including description, required keys, and required timeseries.
    """
    # Get required keys as a list and convert to a dictionary with str type as default
    required_keys_list = getattr(cls, "required_keys", [])
    required_keys = {key: str for key in required_keys_list}

    # Get required timeseries as a list and convert to a dictionary with empty dict as default
    required_timeseries_list = getattr(cls, "required_timeseries", [])
    required_timeseries = {ts: {} for ts in required_timeseries_list}

    dependencies = getattr(cls, "dependencies", [])
    docstring = inspect.getdoc(cls) or ""

    # Extract method docstrings and source code for each function in the class, excluding dunder, private, and inherited methods
    methods = {}
    for name, meth in inspect.getmembers(cls, predicate=inspect.isfunction):
        # Skip dunder methods (methods starting with __) and private methods (methods starting with _)
        if not name.startswith("__") and not name.startswith("_"):
            # Check if the method is defined in this class (not inherited)
            if meth.__qualname__.split(".")[0] == cls.__name__:
                docstring = inspect.getdoc(meth) or ""
                try:
                    source_code = inspect.getsource(meth)
                except (IOError, TypeError):
                    source_code = "# Source code not available"
                methods[name] = {"docstring": docstring, "source_code": source_code}
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
    template.globals["hasattr"] = hasattr

    # Render the template with the metadata and provided timeseries type.
    file_name = getattr(cls, "name", cls.__name__.lower())
    rendered = template.render(
        method_name=file_name,
        cls=cls,  # Pass the class object to access its attributes
        description=metadata["description"],
        required_keys=metadata["required_keys"],
        required_timeseries=metadata["required_timeseries"],
        dependencies=metadata["dependencies"],
        timeseries_type=timeseries_type,
        methods=metadata["methods"],
    )

    # Save the rendered content to a file in the output directory.
    output_file = os.path.join(output_dir, f"{file_name}.rst")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(rendered)
    # print(f"Documentation for {cls.__name__} written to {output_file}")


def generate_docs_for_all_methods(method_classes, template_path, base_output_dir):
    """
    Loop over all provided method classes and generate documentation pages in the correct folder(s)
    based on the timeseries types they support.

    If a method supports multiple types, it is generated in each corresponding folder.
    For auxiliary methods, they are organized by their subtype (e.g., internal, solar).
    """
    # First, clean up any existing auxiliary method files in the main auxiliary directory
    # to avoid duplicate documentation
    auxiliary_dir = os.path.join(base_output_dir, "auxiliary")
    if os.path.exists(auxiliary_dir):
        for file in os.listdir(auxiliary_dir):
            file_path = os.path.join(auxiliary_dir, file)
            if os.path.isfile(file_path) and file.endswith(".rst") and file != "index.rst":
                os.remove(file_path)

    for cls in method_classes:
        # Get the module path to check if it's in the auxiliary directory
        module_path = cls.__module__.split(".")

        # Check if this is an auxiliary method either by inheritance or by module path
        is_auxiliary = issubclass(cls, AuxiliaryMethod) or "auxiliary" in module_path
        if is_auxiliary:
            # For auxiliary methods, determine the subtype from the module path
            if "auxiliary" in module_path and len(module_path) >= 4:
                # Extract the subtype (e.g., 'internal', 'solar')
                subtype = module_path[-2]
                # Place in auxiliary/subtype directory
                out_dir = os.path.join(base_output_dir, "auxiliary", subtype)
                os.makedirs(out_dir, exist_ok=True)
                generate_docs_for_method(cls, "auxiliary", template_path, out_dir)
            else:
                # Fallback for auxiliary methods without a clear subtype
                out_dir = os.path.join(base_output_dir, "auxiliary")
                os.makedirs(out_dir, exist_ok=True)
                generate_docs_for_method(cls, "auxiliary", template_path, out_dir)
        else:
            # Check if the class has a types attribute
            has_types = hasattr(cls, "types")

            # Get the types attribute if it exists
            if has_types:
                supported_types = cls.types
            else:
                supported_types = getattr(cls, "types", None)

            if supported_types is None:
                # Determine type from module path instead of defaulting to HVAC
                module_path = cls.__module__.split(".")
                if len(module_path) >= 3 and module_path[2] in VALID_TYPES:
                    # Use the module directory name as the type
                    supported_types = [module_path[2].upper()]
                else:
                    # If we can't determine a specific type, use an empty list to avoid categorization
                    supported_types = []

            for timeseries_type in supported_types:
                # Determine the output folder based on the timeseries type.
                out_dir = os.path.join(base_output_dir, timeseries_type.lower())
                os.makedirs(out_dir, exist_ok=True)
                generate_docs_for_method(cls, timeseries_type, template_path, out_dir)


def generate_index_for_folder(folder_path):
    """
    Generate an index.rst file in the given folder_path that lists all .rst files (except index.rst).
    For the auxiliary directory, only include links to subdirectories.
    """
    folder_name = os.path.basename(folder_path)

    # Special handling for the auxiliary directory
    if folder_name.lower() == "auxiliary":
        # Create a custom index file that only includes links to subdirectories
        title = "Auxiliary methods"
        underline = "=" * len(title)
        content_lines = [
            title,
            underline,
            "",
            ".. toctree::",
            "   :maxdepth: 1",
            "",
        ]

        # Add links to subdirectories
        for entry in os.listdir(folder_path):
            subdir_path = os.path.join(folder_path, entry)
            if os.path.isdir(subdir_path):
                content_lines.append(f"   {entry}/index")

        content = "\n".join(content_lines)

        # Write the content to index.rst in the folder
        output_file = os.path.join(folder_path, "index.rst")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
        return

    # For other directories, use the original logic
    # List all .rst files in the folder, excluding index.rst
    rst_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".rst") and f.lower() != "index.rst"])

    if not rst_files:
        return

    # Create a title from the folder name (e.g., "hvac" -> "HVAC Methods")
    title = f"{folder_name.camel()} methods"
    underline = "=" * len(title)
    # Build the content with a toctree directive
    content_lines = [
        title,
        underline,
        "",
        ".. toctree::",
        "   :maxdepth: 1",
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


def update_parent_index(parent_dir, subdir_name):
    """
    Update the parent directory's index.rst file to include a link to the subdirectory.

    Parameters:
      - parent_dir: Path to the parent directory
      - subdir_name: Name of the subdirectory to include
    """
    index_path = os.path.join(parent_dir, "index.rst")
    if not os.path.exists(index_path):
        return

    # Read the current content
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if the toctree directive exists
    if ".. toctree::" not in content:
        return

    # Check if the subdirectory is already included
    if f"{subdir_name}/index" in content:
        return

    # Find the toctree section and add the subdirectory
    lines = content.split("\n")
    toctree_index = -1
    for i, line in enumerate(lines):
        if ".. toctree::" in line:
            toctree_index = i
            break

    if toctree_index == -1:
        return

    # Find where to insert the new entry (after the last entry in the toctree)
    insert_index = toctree_index + 1
    while insert_index < len(lines) and (not lines[insert_index].strip() or lines[insert_index].startswith("   ")):
        insert_index += 1

    # Insert the new entry
    lines.insert(insert_index, f"   {subdir_name}/index")

    # Write the updated content
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_indexes(base_dir):
    """
    Walk through each subdirectory in base_dir and generate an index.rst file.
    Also handles nested subdirectories (e.g., auxiliary/internal).
    """
    for entry in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, entry)
        if os.path.isdir(subdir_path):
            # Generate index for this directory
            generate_index_for_folder(subdir_path)

            # Check for subdirectories and generate indexes for them too
            for subentry in os.listdir(subdir_path):
                subsubdir_path = os.path.join(subdir_path, subentry)
                if os.path.isdir(subsubdir_path):
                    generate_index_for_folder(subsubdir_path)

                    # Update the parent directory's index to include the subdirectory
                    update_parent_index(subdir_path, subentry)


if __name__ == "__main__":
    try:
        # Automatically discover all method classes in the package.
        print("Discovering method classes...")
        method_classes = discover_method_classes("entise.methods")
        print(f"Discovered {len(method_classes)} method classes")

        # Print discovered classes for debugging
        for cls in method_classes:
            print(f"Found class: {cls.__name__} in module {cls.__module__}")

        # Generate documentation for each method in the appropriate folder(s).
        print("Generating documentation...")
        generate_docs_for_all_methods(method_classes, TEMPLATE_PATH, BASE_OUTPUT_DIR)

        # Generate an index.rst in each subfolder under BASE_OUTPUT_DIR.
        print("Generating indexes...")
        generate_indexes(BASE_OUTPUT_DIR)

        print("Documentation generation completed successfully!")
    except Exception as e:
        import traceback

        print(f"Error during documentation generation: {e}")
        traceback.print_exc()
