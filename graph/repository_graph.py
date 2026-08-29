import json
import networkx as nx

from graph.import_resolver import (
    build_module_map,
    resolve_import,
)


DATASET_PATH = "data/processed/repository_dataset.json"


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# =========================================================
# UNIQUE IDS
# =========================================================

def build_function_id(file_record, function):
    return (
        f"{file_record['relative_path']}::"
        f"{function['qualified_name']}::"
        f"L{function['line_start']}"
    )


def build_class_id(file_record, class_record):
    return (
        f"{file_record['relative_path']}::"
        f"{class_record['name']}::"
        f"L{class_record['line_start']}"
    )


# =========================================================
# NORMALIZE CALLS
# =========================================================

def normalize_call(call_name, parent_class=None):
    """
    Convert:

        self.create_session

    into:

        UserService.create_session

    when the current function belongs to UserService.
    """

    if not isinstance(call_name, str):
        return call_name

    if call_name.startswith("self.") and parent_class:

        method_name = call_name.split(".", 1)[1]

        return f"{parent_class}.{method_name}"

    return call_name


# =========================================================
# IMPORT MAP
# =========================================================

class ImportMap(dict):
    """
    Maps names visible inside one file to import metadata.

    Examples:

    from flask.cli import find_best_app

        find_best_app
            ->
        {
            module: flask.cli,
            name: find_best_app
        }


    from flask.cli import find_best_app as finder

        finder
            ->
        {
            module: flask.cli,
            name: find_best_app
        }


    import flask.cli as cli

        cli
            ->
        {
            module: flask.cli,
            name: None
        }
    """

    def add(self, import_record):

        if not isinstance(import_record, dict):
            return

        import_type = import_record.get("type")
        module = import_record.get("module")
        name = import_record.get("name")
        alias = import_record.get("alias")

        # -------------------------------------
        # from x import y
        # -------------------------------------

        if import_type == "from":

            visible_name = alias or name

            if visible_name:

                self[visible_name] = {
                    "module": module,
                    "name": name,
                    "type": "from",
                }

        # -------------------------------------
        # import x
        # import x as y
        # -------------------------------------

        elif import_type == "import":

            if not module:
                return

            visible_name = (
                alias
                or module.split(".", 1)[0]
            )

            self[visible_name] = {
                "module": module,
                "name": None,
                "type": "import",
            }

    def resolve(self, call_name):
        """
        Resolve visible names.

        Examples:

        finder()
            -> imported symbol

        cli.find_best_app()
            -> module + function

        flask.cli.find_best_app()
            -> module-qualified function
        """

        if not isinstance(call_name, str):
            return None

        first_part, separator, remainder = (
            call_name.partition(".")
        )

        import_data = self.get(first_part)

        if import_data is None:
            return None

        result = dict(import_data)

        if separator and remainder:

            # import flask.cli as cli
            # cli.find_best_app()

            if result.get("name") is None:

                result["name"] = remainder

            else:

                # Rare chained imported symbol case

                result["name"] = (
                    f"{result['name']}."
                    f"{remainder}"
                )

        return result


# =========================================================
# FUNCTION LOOKUPS
# =========================================================

def add_function_lookup(
    function_lookup,
    file_function_lookup,
    file_record,
    function,
    function_id
):
    qualified_name = function["qualified_name"]
    simple_name = function["name"]
    relative_path = file_record["relative_path"]

    # ==========================================
    # GLOBAL LOOKUP
    # ==========================================

    function_lookup.setdefault(
        qualified_name,
        []
    )

    if function_id not in function_lookup[qualified_name]:
        function_lookup[qualified_name].append(
            function_id
        )

    # Only add simple name separately if it is
    # actually different from qualified name.
    #
    # Example:
    #
    # UserService.login
    # vs
    # login
    #
    if simple_name != qualified_name:

        function_lookup.setdefault(
            simple_name,
            []
        )

        if function_id not in function_lookup[simple_name]:
            function_lookup[simple_name].append(
                function_id
            )

    # ==========================================
    # FILE-SPECIFIC LOOKUP
    # ==========================================

    file_lookup = file_function_lookup.setdefault(
        relative_path,
        {}
    )

    file_lookup.setdefault(
        qualified_name,
        []
    )

    if function_id not in file_lookup[qualified_name]:
        file_lookup[qualified_name].append(
            function_id
        )

    if simple_name != qualified_name:

        file_lookup.setdefault(
            simple_name,
            []
        )

        if function_id not in file_lookup[simple_name]:
            file_lookup[simple_name].append(
                function_id
            )


# =========================================================
# LOOKUP FUNCTION IN SPECIFIC FILE
# =========================================================

def find_function_in_file(
    file_function_lookup,
    file_path,
    function_name
):

    if not file_path or not function_name:
        return []

    lookup = file_function_lookup.get(
        file_path,
        {}
    )

    # Exact match
    candidates = lookup.get(
        function_name,
        []
    )

    if candidates:
        return candidates

    # -----------------------------------------------------
    # If something like Class.method was provided,
    # try the final part too.
    # -----------------------------------------------------

    simple_name = function_name.split(".")[-1]

    return lookup.get(
        simple_name,
        []
    )


# =========================================================
# BUILD REPOSITORY GRAPH
# =========================================================

def build_repository_graph(dataset):

    graph = nx.MultiDiGraph()

    repository_id = "repository:test_repo"

    graph.add_node(
        repository_id,
        node_type="repository",
        name="test_repo"
    )

    # -----------------------------------------------------
    # LOOKUPS
    # -----------------------------------------------------

    function_lookup = {}

    file_function_lookup = {}

    class_lookup = {}

    # =====================================================
    # PASS 1
    # FILES + CLASSES + FUNCTIONS
    # =====================================================

    for file_record in dataset:

        relative_path = file_record["relative_path"]

        file_id = f"file::{relative_path}"

        # -------------------------------------------------
        # FILE
        # -------------------------------------------------

        graph.add_node(
            file_id,
            node_type="file",
            path=relative_path,
            language=file_record["language"],
            size_bytes=file_record["size_bytes"]
        )

        graph.add_edge(
            repository_id,
            file_id,
            relation="CONTAINS"
        )

        # -------------------------------------------------
        # CLASSES
        # -------------------------------------------------

        for class_record in file_record["classes"]:

            class_id = build_class_id(
                file_record,
                class_record
            )

            graph.add_node(
                class_id,
                node_type="class",
                name=class_record["name"],
                file=relative_path,
                line_start=class_record["line_start"],
                line_end=class_record["line_end"]
            )

            graph.add_edge(
                file_id,
                class_id,
                relation="CONTAINS"
            )

            class_key = (
                relative_path,
                class_record["name"]
            )

            class_lookup[class_key] = class_id

        # -------------------------------------------------
        # FUNCTIONS
        # -------------------------------------------------

        for function in file_record["functions"]:

            function_id = build_function_id(
                file_record,
                function
            )

            graph.add_node(
                function_id,
                node_type="function",
                name=function["name"],
                qualified_name=function["qualified_name"],
                function_type=function["function_type"],
                parent_class=function["parent_class"],
                file=relative_path,
                line_start=function["line_start"],
                line_end=function["line_end"],
                is_async=function.get(
                    "is_async",
                    False
                )
            )

            # ---------------------------------------------
            # Add function to lookup tables
            # ---------------------------------------------

            add_function_lookup(
                function_lookup,
                file_function_lookup,
                file_record,
                function,
                function_id
            )

            # ---------------------------------------------
            # METHOD
            # ---------------------------------------------

            if function["parent_class"]:

                class_key = (
                    relative_path,
                    function["parent_class"]
                )

                class_id = class_lookup.get(
                    class_key
                )

                if class_id:

                    graph.add_edge(
                        class_id,
                        function_id,
                        relation="CONTAINS"
                    )

                else:

                    graph.add_edge(
                        file_id,
                        function_id,
                        relation="CONTAINS"
                    )

            # ---------------------------------------------
            # NORMAL FUNCTION
            # ---------------------------------------------

            else:

                graph.add_edge(
                    file_id,
                    function_id,
                    relation="CONTAINS"
                )

    # =====================================================
    # PASS 2
    # STRUCTURED IMPORT RESOLUTION
    # =====================================================

    module_map = build_module_map(dataset)

    internal_imports = 0
    external_imports = 0

    # -----------------------------------------------------
    # Create import maps for each file
    # -----------------------------------------------------

    file_import_maps = {}

    for file_record in dataset:

        imports = ImportMap()

        for import_record in file_record.get(
            "imports",
            []
        ):

            imports.add(
                import_record
            )

        file_import_maps[
            file_record["relative_path"]
        ] = imports

    # -----------------------------------------------------
    # Create graph IMPORT edges
    # -----------------------------------------------------

    for file_record in dataset:

        source_file_id = (
            f"file::{file_record['relative_path']}"
        )

        for import_record in file_record.get(
            "imports",
            []
        ):

            if not isinstance(
                import_record,
                dict
            ):
                continue

            target_path = resolve_import(
                import_record,
                module_map
            )

            # ---------------------------------------------
            # INTERNAL IMPORT
            # ---------------------------------------------

            if target_path:

                target_file_id = (
                    f"file::{target_path}"
                )

                graph.add_edge(
                    source_file_id,
                    target_file_id,
                    relation="IMPORTS_INTERNAL",
                    import_type=import_record.get(
                        "type"
                    ),
                    module=import_record.get(
                        "module"
                    ),
                    imported_name=import_record.get(
                        "name"
                    ),
                    alias=import_record.get(
                        "alias"
                    ),
                    level=import_record.get(
                        "level",
                        0
                    ),
                )

                internal_imports += 1

            # ---------------------------------------------
            # EXTERNAL IMPORT
            # ---------------------------------------------

            else:

                module_name = import_record.get(
                    "module"
                )

                imported_name = import_record.get(
                    "name"
                )

                if module_name is None:
                    module_name = imported_name

                if not module_name:
                    continue

                module_id = (
                    f"external_module::{module_name}"
                )

                if not graph.has_node(
                    module_id
                ):

                    graph.add_node(
                        module_id,
                        node_type="external_module",
                        name=module_name
                    )

                graph.add_edge(
                    source_file_id,
                    module_id,
                    relation="IMPORTS_EXTERNAL",
                    import_type=import_record.get(
                        "type"
                    ),
                    imported_name=imported_name,
                    alias=import_record.get(
                        "alias"
                    ),
                    level=import_record.get(
                        "level",
                        0
                    ),
                )

                external_imports += 1

    # =====================================================
    # PASS 3
    # IMPORT-AWARE FUNCTION CALL RESOLUTION
    # =====================================================

    resolved_calls = 0
    ambiguous_calls = 0
    unresolved_calls = 0

    direct_resolved = 0
    same_file_resolved = 0
    import_resolved = 0

    for file_record in dataset:

        relative_path = file_record[
            "relative_path"
        ]

        current_import_map = (
            file_import_maps.get(
                relative_path,
                ImportMap()
            )
        )

        for function in file_record["functions"]:

            caller_id = build_function_id(
                file_record,
                function
            )

            calls = function.get(
                "calls",
                []
            )

            for call in calls:

                candidates = []

                resolution_type = None

                # =========================================
                # STEP 1
                # NORMALIZE self.method()
                # =========================================

                target_name = normalize_call(
                    call,
                    function.get(
                        "parent_class"
                    )
                )

                # =========================================
                # STEP 2
                # SAME FILE LOOKUP
                #
                # Prefer local functions before global
                # search.
                # =========================================

                candidates = find_function_in_file(
                    file_function_lookup,
                    relative_path,
                    target_name
                )

                if len(candidates) == 1:

                    resolution_type = (
                        "same_file"
                    )

                # =========================================
                # STEP 3
                # GLOBAL LOOKUP
                # =========================================

                if not candidates:

                    candidates = (
                        function_lookup.get(
                            target_name,
                            []
                        )
                    )

                    if len(candidates) == 1:

                        resolution_type = (
                            "direct"
                        )

                # =========================================
                # STEP 4
                # IMPORT-AWARE LOOKUP
                # =========================================

                if not candidates:

                    imported = (
                        current_import_map.resolve(
                            call
                        )
                    )

                    if imported:

                        module_name = imported.get(
                            "module"
                        )

                        imported_function = (
                            imported.get(
                                "name"
                            )
                        )

                        # ---------------------------------
                        # Find repository file belonging
                        # to imported module
                        # ---------------------------------

                        target_file = None

                        if module_name:

                            target_file = (
                                module_map.get(
                                    module_name
                                )
                            )

                        # ---------------------------------
                        # If exact module didn't work,
                        # use resolve_import logic
                        # ---------------------------------

                        if (
                            target_file is None
                            and module_name
                        ):

                            fake_import_record = {
                                "type": "from",
                                "module": module_name,
                                "name": imported_function,
                                "alias": None,
                                "level": 0,
                            }

                            target_file = (
                                resolve_import(
                                    fake_import_record,
                                    module_map
                                )
                            )

                        # ---------------------------------
                        # Search only inside imported file
                        # ---------------------------------

                        if (
                            target_file
                            and imported_function
                        ):

                            candidates = (
                                find_function_in_file(
                                    file_function_lookup,
                                    target_file,
                                    imported_function
                                )
                            )

                            if len(candidates) == 1:

                                resolution_type = (
                                    "import"
                                )

                # =========================================
                # RESULT
                # =========================================

                if len(candidates) == 1:

                    target_id = candidates[0]

                    graph.add_edge(
                        caller_id,
                        target_id,
                        relation="CALLS",
                        original_call=call,
                        resolution=resolution_type
                    )

                    resolved_calls += 1

                    if resolution_type == "direct":

                        direct_resolved += 1

                    elif (
                        resolution_type
                        == "same_file"
                    ):

                        same_file_resolved += 1

                    elif (
                        resolution_type
                        == "import"
                    ):

                        import_resolved += 1

                elif len(candidates) > 1:

                    ambiguous_calls += 1

                else:

                    unresolved_calls += 1

    # =====================================================
    # STATS
    # =====================================================

    stats = {

        "resolved_calls":
            resolved_calls,

        "ambiguous_calls":
            ambiguous_calls,

        "unresolved_calls":
            unresolved_calls,

        "internal_imports":
            internal_imports,

        "external_imports":
            external_imports,

        "direct_resolved":
            direct_resolved,

        "same_file_resolved":
            same_file_resolved,

        "import_resolved":
            import_resolved,
    }

    return graph, stats


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    dataset = load_dataset(
        DATASET_PATH
    )

    graph, stats = (
        build_repository_graph(
            dataset
        )
    )

    # =====================================================
    # GENERAL
    # =====================================================

    print(
        "\nRepoMind Repository Graph"
    )

    print(
        "-------------------------"
    )

    print(
        f"Total nodes: "
        f"{graph.number_of_nodes()}"
    )

    print(
        f"Total edges: "
        f"{graph.number_of_edges()}"
    )

    # =====================================================
    # NODE TYPES
    # =====================================================

    node_counts = {}

    for _, data in graph.nodes(
        data=True
    ):

        node_type = data.get(
            "node_type",
            "unknown"
        )

        node_counts[node_type] = (
            node_counts.get(
                node_type,
                0
            )
            + 1
        )

    print(
        "\nNode Types"
    )

    print(
        "-------------------------"
    )

    for (
        node_type,
        count
    ) in node_counts.items():

        print(
            f"{node_type}: {count}"
        )

    # =====================================================
    # RELATIONSHIP TYPES
    # =====================================================

    relation_counts = {}

    for _, _, data in graph.edges(
        data=True
    ):

        relation = data.get(
            "relation",
            "unknown"
        )

        relation_counts[relation] = (
            relation_counts.get(
                relation,
                0
            )
            + 1
        )

    print(
        "\nRelationship Types"
    )

    print(
        "-------------------------"
    )

    for (
        relation,
        count
    ) in relation_counts.items():

        print(
            f"{relation}: {count}"
        )

    # =====================================================
    # IMPORT RESOLUTION
    # =====================================================

    print(
        "\nImport Resolution"
    )

    print(
        "-------------------------"
    )

    print(
        f"Internal imports: "
        f"{stats['internal_imports']}"
    )

    print(
        f"External imports: "
        f"{stats['external_imports']}"
    )

    # =====================================================
    # CALL RESOLUTION
    # =====================================================

    print(
        "\nCall Resolution"
    )

    print(
        "-------------------------"
    )

    print(
        f"Resolved calls: "
        f"{stats['resolved_calls']}"
    )

    print(
        f"Ambiguous calls: "
        f"{stats['ambiguous_calls']}"
    )

    print(
        f"Unresolved calls: "
        f"{stats['unresolved_calls']}"
    )

    print(
        "\nResolution Breakdown"
    )

    print(
        "-------------------------"
    )

    print(
        f"Same-file resolved: "
        f"{stats['same_file_resolved']}"
    )

    print(
        f"Direct/global resolved: "
        f"{stats['direct_resolved']}"
    )

    print(
        f"Import-aware resolved: "
        f"{stats['import_resolved']}"
    )

    # =====================================================
    # EXAMPLE INTERNAL IMPORTS
    # =====================================================

    print(
        "\nExample Internal Imports"
    )

    print(
        "-------------------------"
    )

    shown = 0

    for (
        source,
        target,
        data
    ) in graph.edges(
        data=True
    ):

        if (
            data.get("relation")
            == "IMPORTS_INTERNAL"
        ):

            source_data = (
                graph.nodes[source]
            )

            target_data = (
                graph.nodes[target]
            )

            print(
                f"{source_data.get('path')}"
                " -> "
                f"{target_data.get('path')}"
                " | module="
                f"{data.get('module')}"
                " | name="
                f"{data.get('imported_name')}"
            )

            shown += 1

            if shown >= 10:
                break

    # =====================================================
    # EXAMPLE IMPORT-AWARE CALLS
    # =====================================================

    print(
        "\nExample Import-Aware Calls"
    )

    print(
        "-------------------------"
    )

    shown = 0

    for (
        source,
        target,
        data
    ) in graph.edges(
        data=True
    ):

        if (
            data.get("relation") == "CALLS"
            and
            data.get("resolution") == "import"
        ):

            source_data = graph.nodes[source]
            target_data = graph.nodes[target]

            print(
                f"{source_data.get('qualified_name')}"
                f" [{source_data.get('file')}]"
                " -> "
                f"{target_data.get('qualified_name')}"
                f" [{target_data.get('file')}]"
            )

            shown += 1

            if shown >= 10:
                break