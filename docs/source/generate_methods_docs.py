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
            for _, obj in inspect.getmembers(module, inspect.isclass):
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


def get_enriched_description(cls) -> str:
    """Return an in-depth description for known methods if the class docstring is
    missing or too short. The text is crafted to add value and cite the
    underlying standard/package where applicable.

    This keeps runtime code untouched while ensuring method docs are helpful.
    """
    mod = getattr(cls, "__module__", "")
    name = getattr(cls, "__name__", "")
    key = getattr(cls, "name", name).lower()

    # Common references
    refs = {
        "pvlib": "pvlib-python: https://pvlib-python.readthedocs.io/",
        "windpowerlib": "windpowerlib: https://windpowerlib.readthedocs.io/",
        "demandlib": "demandlib (BDEW SLPs): https://demandlib.readthedocs.io/",
        "iso_13790": "ISO 13790: Energy performance of buildings — Calculation of energy use for space heating and cooling",
        "vdi_6007": "VDI 6007: Calculation of transient thermal response of rooms and buildings",
        "dhwcalc": "DHWcalc (Jordan & Vajen): http://www.solar-heating.at/downloads/DHWcalc_english.pdf",
        "ruhnau": "Ruhnau et al. (heat pump COP vs. ambient temperature): https://doi.org/10.1016/j.enpol.2019.111179",
        "geomA": "GeoMA Occupancy: Geiger et al. (geospatial occupancy approximation)",
        "pht": "PHT Occupancy detection via power/heat thresholding",
    }

    # Heuristics by domain/module
    if "entise.methods.pv" in mod or key == "pvlib":
        return (
            "Photovoltaic (PV) generation based on pvlib-python. This method sets up a "
            "pvlib.pvsystem model chain using panel orientation (tilt/azimuth), array and inverter "
            "parameters, and meteorological inputs (GHI/DNI/DHI, temperature, wind) to compute AC power. "
            "It leverages pvlib's well-tested transposition, temperature, and system-performance models.\n\n"
            f"References: {refs['pvlib']}."
        )

    if "entise.methods.wind" in mod or "wplib" in key or "wind" in key:
        return (
            "Wind power generation using windpowerlib. The method constructs a turbine with hub height and "
            "power curve, corrects wind speeds to hub height if needed, and applies windpowerlib's model "
            "chain to obtain electrical output. Results depend on wind class and roughness length inputs.\n\n"
            f"References: {refs['windpowerlib']}."
        )

    if "entise.methods.electricity" in mod or "demandlib" in key:
        return (
            "Electricity demand profiles using demandlib's BDEW standard load profiles (SLPs). Given a time "
            "horizon and an annual demand, the method builds BDEW SLPs (e.g., H0 household) for the covered years, "
            "optionally adjusts for holidays, and scales to the requested annual energy. The 15-minute SLP is then "
            "aligned to the target resolution using energy-conserving resampling.\n\n"
            f"References: {refs['demandlib']}."
        )

    if "entise.methods.hvac" in mod and name.upper().startswith("R1C1"):
        return (
            "1R1C thermal RC model for room/building air node. A single thermal resistance (R) to ambient and a "
            "single thermal capacitance (C) capture conductive/ventilation losses and thermal inertia. Internal and "
            "solar gains as well as heating/cooling power are balanced each time step to integrate indoor temperature "
            "forward. Suitable for quick load estimates and control studies."
        )

    if "entise.methods.hvac" in mod and name.upper().startswith("R5C1"):
        return (
            "5R1C grey-box HVAC model inspired by ISO 13790 simplified dynamic method. It separates heat transfer "
            "paths and uses an aggregated thermal mass to represent building dynamics. Compared to 1R1C it better "
            "captures radiant/convective splits and envelope interactions for more accurate heat load estimation.\n\n"
            f"Reference: {refs['iso_13790']}."
        )

    if "entise.methods.hvac" in mod and "7R2C" in name.upper():
        return (
            "7R2C multi-node thermal model aligned with VDI 6007 concepts. Two capacitances (air/mass) and multiple "
            "resistances (walls, windows, surfaces) allow representing phase shifts and damping of temperature waves, "
            "useful for envelope studies and solar gains interactions.\n\n"
            f"Reference: {refs['vdi_6007']}."
        )

    if "entise.methods.dhw" in mod:
        return (
            "Domestic hot water (DHW) demand based on Jordan & Vajen profiles as popularized by DHWcalc. The method "
            "synthesizes stochastic tapping events (draw volumes and times) consistent with dwelling size and occupants, "
            "and converts them to thermal energy using cold/hot water temperatures and heater assumptions.\n\n"
            f"Reference: {refs['dhwcalc']}."
        )

    if "entise.methods.hp" in mod or key == "ruhnau":
        return (
            "Heat pump performance (COP) timeseries following Ruhnau et al.: COP as a function of temperature lift "
            "between ambient/source and sink (e.g., radiator or floor heating). The method maps source/sink types to "
            "typical supply/return setpoints and computes COP over time; can be combined with HVAC loads to estimate "
            "electricity demand.\n\n"
            f"Reference: {refs['ruhnau']}."
        )

    if "entise.methods.occupancy" in mod:
        return (
            "Occupancy profile generation/detection. Methods in this family either generate plausible presence schedules "
            "from lightweight priors (e.g., GeoMA) or infer occupancy from measured signals via thresholding and schedule "
            "heuristics (e.g., PHT). Outputs are occupancy indicators or internal gains proxies for downstream HVAC models."
        )

    if "entise.methods.auxiliary" in mod:
        return (
            "Auxiliary strategy used by main methods to derive intermediate inputs (e.g., solar or internal gains, ventilation). "
            "Selectors pick the most specific strategy given available object keys and input data, enabling automatic yet "
            "overrideable preprocessing within the workflow."
        )

    # Default fallback: empty string to keep original behavior
    return ""


def extract_method_metadata(cls):
    """
    Extract metadata from a timeseries method class using the current Method API.

    Parameters:
      - cls: The method class to analyze.

    Returns:
      dict: Metadata including description, requirements, supported types, and outputs.
    """
    # Description from class docstring (fallback to enriched description when too short)
    class_doc = inspect.getdoc(cls) or ""
    enriched = get_enriched_description(cls)
    if len(class_doc.strip()) < 80 and enriched:
        class_doc = enriched

    # Sanitize description for RST rendering: dedent, normalize spaces, and
    # ensure blank lines before bullet lists to avoid "Unexpected indentation".
    import re
    from textwrap import dedent

    def sanitize_description(text: str) -> str:
        t = dedent(text).replace("\r\n", "\n").replace("\r", "\n")
        # Collapse 3+ blank lines to max 2
        t = re.sub(r"\n{3,}", "\n\n", t)
        lines = t.split("\n")
        out: list[str] = []
        i = 0
        in_bullets = False
        prev_nonempty = False
        n = len(lines)
        while i < n:
            ln = lines[i]
            stripped = ln.lstrip()
            is_bullet = stripped.startswith("- ")

            # If a bullet starts without a separating blank line, insert one
            if is_bullet and prev_nonempty and (not out or out[-1] != ""):
                out.append("")

            if is_bullet:
                # Merge wrapped bullet lines until blank line or next bullet
                item = stripped[2:].strip()
                j = i + 1
                while j < n:
                    nxt = lines[j]
                    nxt_stripped = nxt.strip()
                    if nxt_stripped == "":
                        break
                    if nxt.lstrip().startswith("- "):
                        break
                    # continuation of the same bullet -> join with a space
                    item += " " + nxt_stripped
                    j += 1
                out.append(f"- {item}")
                in_bullets = True
                prev_nonempty = True
                # If the next line ends the list, ensure a trailing blank line
                # will be added by the general logic below when we encounter
                # the terminating condition
                i = j
                # If we stopped because of a blank line, consume it and emit one
                if i < n and lines[i].strip() == "":
                    if out and out[-1] != "":
                        out.append("")
                    in_bullets = False
                    prev_nonempty = False
                    i += 1
                continue
            else:
                # If previous lines were bullets and now normal text begins, add a blank line
                if in_bullets and stripped != "":
                    if out and out[-1] != "":
                        out.append("")
                    in_bullets = False
                # Avoid accidental block quotes: use lstrip
                out.append(stripped)
                prev_nonempty = stripped != ""
                i += 1
                continue

        # Trim excessive blank lines at the end and ensure exactly one
        while out and out[-1] == "":
            out.pop()
        out.append("")
        return "\n".join(out)

    class_doc = sanitize_description(class_doc)

    # Requirements from Method/AuxiliaryMethod API
    required_keys_list = getattr(cls, "required_keys", []) or []
    optional_keys_list = getattr(cls, "optional_keys", []) or []
    required_data_list = getattr(cls, "required_data", []) or []
    optional_data_list = getattr(cls, "optional_data", []) or []

    # Outputs
    output_summary = getattr(cls, "output_summary", {}) or {}
    output_timeseries = getattr(cls, "output_timeseries", {}) or {}

    # Supported types and method key
    types = getattr(cls, "types", []) or []
    method_key = getattr(cls, "name", cls.__name__)

    # Extract public methods' docstrings and sources (skip dunder/private and inherited)
    public_methods: dict = {}
    for name, meth in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        if getattr(meth, "__qualname__", "").split(".")[0] != cls.__name__:
            continue
        meth_doc = inspect.getdoc(meth) or ""
        try:
            source_code = inspect.getsource(meth)
        except (IOError, TypeError):
            source_code = "# Source code not available"
        public_methods[name] = {"docstring": meth_doc, "source_code": source_code}

    return {
        "description": class_doc,
        "method_key": method_key,
        "types": types,
        "required_keys": required_keys_list,
        "optional_keys": optional_keys_list,
        "required_data": required_data_list,
        "optional_data": optional_data_list,
        "output_summary": output_summary,
        "output_timeseries": output_timeseries,
        "methods": public_methods,
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
    with open(template_path, "r", encoding="utf-8") as f:
        template = Template(f.read())

    # Add 'hasattr' to the template's globals so it's available during rendering.
    template.globals["hasattr"] = hasattr

    # Render the template with the metadata and provided timeseries type.
    file_name = getattr(cls, "name", cls.__name__.lower())
    rendered = template.render(
        method_name=file_name,
        cls=cls,  # Pass the class object to access its attributes
        description=metadata["description"],
        method_key=metadata["method_key"],
        supported_types=metadata["types"],
        required_keys=metadata["required_keys"],
        optional_keys=metadata["optional_keys"],
        required_data=metadata["required_data"],
        optional_data=metadata["optional_data"],
        output_summary=metadata["output_summary"],
        output_timeseries=metadata["output_timeseries"],
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
    Always writes an index.rst (with a placeholder note if folder has no pages yet).
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
        has_subdirs = False
        for entry in os.listdir(folder_path):
            subdir_path = os.path.join(folder_path, entry)
            if os.path.isdir(subdir_path):
                content_lines.append(f"   {entry}/index")
                has_subdirs = True

        if not has_subdirs:
            content_lines.extend(
                [
                    "",
                    ".. note::",
                    "   No auxiliary methods discovered yet.",
                ]
            )

        content = "\n".join(content_lines)

        # Write the content to index.rst in the folder
        output_file = os.path.join(folder_path, "index.rst")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
        return

    # For other directories, list all .rst files in the folder, excluding index.rst
    rst_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".rst") and f.lower() != "index.rst"])

    # Create a title from the folder name (e.g., "hvac" -> "HVAC Methods")
    pretty_name = folder_name.replace("_", " ").title()
    title = f"{pretty_name} methods"
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

    if not rst_files:
        content_lines.extend(
            [
                "",
                ".. note::",
                "   No methods documented yet for this type.",
            ]
        )

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


def ensure_top_level_index(base_dir):
    """Create or update the top-level methods/index.rst with links to all type folders and auxiliary.

    Uses entise.constants.VALID_TYPES to determine main type folders.
    """
    from entise.constants import VALID_TYPES

    os.makedirs(base_dir, exist_ok=True)
    index_path = os.path.join(base_dir, "index.rst")

    # Create content
    title = "Methods"
    underline = "=" * len(title)
    lines = [
        ".. _methods:",
        "",
        title,
        underline,
        "",
        "This section documents all available time series generation methods in EnTiSe. Pages are auto-generated from the code at build time, so they stay consistent with the current API.",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "",
    ]

    # Add main types from constants
    for ts_type in sorted(VALID_TYPES):
        lines.append(f"   {ts_type}/index")

    # Add auxiliary index
    lines.append("   auxiliary/index")

    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def bootstrap_methods_tree(base_dir):
    """Ensure folder structure exists for all main types (from constants) and auxiliary subfolders.

    - Creates base_dir and top-level index if missing.
    - Creates subfolders for each VALID_TYPES item and auxiliary/ plus discovered auxiliary subpackages.
    - Writes placeholder index.rst files to empty folders to make Sphinx toctrees resolvable.
    """
    import importlib

    from entise.constants import VALID_TYPES

    os.makedirs(base_dir, exist_ok=True)

    # Ensure top-level index exists/updated
    ensure_top_level_index(base_dir)

    # Create per-type folders and minimal index
    for ts_type in sorted(VALID_TYPES):
        tdir = os.path.join(base_dir, ts_type)
        os.makedirs(tdir, exist_ok=True)
        generate_index_for_folder(tdir)

    # Auxiliary root
    aux_dir = os.path.join(base_dir, "auxiliary")
    os.makedirs(aux_dir, exist_ok=True)
    generate_index_for_folder(aux_dir)

    # Discover auxiliary subpackages (filesystem) and precreate
    try:
        pkg = importlib.import_module("entise.methods.auxiliary")
        for _, name, ispkg in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
            if ispkg:
                parts = name.split(".")
                # subtype is the last component after 'auxiliary'
                if "auxiliary" in parts:
                    idx = parts.index("auxiliary")
                    if idx + 1 < len(parts):
                        subtype = parts[idx + 1]
                        subdir = os.path.join(aux_dir, subtype)
                        os.makedirs(subdir, exist_ok=True)
                        generate_index_for_folder(subdir)
                        update_parent_index(aux_dir, subtype)
    except Exception as e:
        print(f"[docs] Warning: could not precreate auxiliary subfolders: {e}")


def generate_indexes(base_dir):
    """
    Walk through each subdirectory in base_dir and generate an index.rst file.
    Also handles nested subdirectories (e.g., auxiliary/internal).
    """
    # Always ensure top-level exists
    ensure_top_level_index(base_dir)

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
