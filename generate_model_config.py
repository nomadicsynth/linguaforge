#!/usr/bin/env python3
"""
Generate default YAML config for transformers model config classes.
Usage: python generate_model_config.py <ConfigClassName> [-o output.yaml]
"""

import argparse
import yaml
import sys


def verify_config(config_class, yaml_content):
    """
    Verify that the generated YAML can be loaded by the config class.
    """
    try:
        config_dict = yaml.safe_load(yaml_content)
        config = config_class.from_dict(config_dict)
        # Verify by comparing to_dict output
        generated = config.to_dict()
        if generated == config_dict:
            print(f"✓ Verification successful: {config_class.__name__} loaded from YAML")
            return True
        else:
            print(f"✗ Verification failed: Generated config differs from loaded config")
            return False
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def parse_params(doc: str) -> dict[str, str]:
    params: dict[str, str] = {}
    current = None
    for line in doc.splitlines():
        if line.startswith('        '):             # 8 spaces → description line
            if current is not None:
                params[current] = (params[current] + ' ' + line.strip()).strip()
        elif line[:4] == '    ' and line[4:5] != ' ':  # exactly 4 spaces → key line
            current = line.strip().split()[0]
            params[current] = " ".join(line.strip().split()[1:])  # type hint
    return params


def wizard(config, keep_defaults: bool, ignore_keys: list, depth: int=0):
    output_indent = "  " * depth
    config_dict = {}
    docs = parse_params(config.__doc__ or "")
    if isinstance(config, dict):
        items = config.items()
    else:
        items = config.to_dict().items()
    for key, value in items:
        if isinstance(key, str) and any(k in key for k in ignore_keys):
            print(f"{output_indent}Skipping {key} (internal parameter)")
            continue
        try:
            key_doc = docs.get(key, None)
            if key_doc is not None:
                print(f"{output_indent}Documentation for {key}: {key_doc}")
        except AttributeError:
            pass

        try:
            value_type_hint = config.__dataclass_fields__.get(key).type
            if not isinstance(value, (dict, list)):
                print(f"{output_indent}Type hint for {key}: {value_type_hint}")
        except AttributeError:
            value_type_hint = ''
            pass

        # if it's nested, recurse
        if isinstance(value, dict):
            print(f"{output_indent}{key} is nested. Recursing into it.")
            ret = wizard(value, keep_defaults, ignore_keys, depth=depth + 1)
            if len(ret) > 0:
                config_dict[key] = ret
            continue

        # handle lists
        if isinstance(value, list) or "list[str]" in str(value_type_hint):
            print(f"{output_indent}{key} is a list. Please enter comma-separated values.")
            user_input = input(f"{output_indent}{key} (default: {value}): ")
            if user_input:
                config_dict[key] = [item.strip() for item in user_input.split(",")]
            else:
                if keep_defaults:
                    config_dict[key] = value
            continue

        # handle booleans
        if isinstance(value, bool):
            user_input = input(f"{output_indent}{key} (default: {value}) [y/n]: ")
            if user_input.lower() in ["y", "yes", "t", "true"]:
                config_dict[key] = True
            elif user_input.lower() in ["n", "no", "f", "false"]:
                config_dict[key] = False
            else:
                if keep_defaults:
                    config_dict[key] = value
            continue

        user_input = input(f"{output_indent}{key} (default: {value}): ")
        if user_input:
            config_dict[key] = user_input
        else:
            if keep_defaults:
                config_dict[key] = value

    return config_dict


def main():
    parser = argparse.ArgumentParser(
        description="Generate default YAML config for transformers model config classes"
    )
    parser.add_argument(
        "config_class",
        help="Model config class name (e.g., BertConfig, RobertaConfig)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file name",
        default=None
    )
    parser.add_argument(
        "-w", "--wizard",
        action="store_true",
        help="Run in wizard mode to interactively set config parameters"
    )
    parser.add_argument(
        "-k", "--keep-defaults",
        action="store_true",
        help="Keep default values for parameters not modified in wizard mode"
    )
    # TODO: change this to be for verifying a supplied config file, not the generated one
    # parser.add_argument(
    #     "-v", "--verify",
    #     action="store_true",
    #     help="Verify generated YAML by loading it back"
    # )

    args = parser.parse_args()

    # Get config class
    from pydoc import locate
    config_class = locate(f"transformers.{args.config_class}")
    if config_class is None:
        print(f"Error: Config class '{args.config_class}' not found in transformers.")
        sys.exit(1)

    # Generate YAML
    config = config_class()

    ignore_keys = ["_name_or_path", "model_type", "transformers_version"]

    # Run wizard mode if specified
    if args.wizard:
        print(f"Running in wizard mode for {args.config_class}. Press Enter to keep default values.")
        config_dict = wizard(config, args.keep_defaults, ignore_keys=ignore_keys)
    else:
        config_dict = config.to_dict()
        # Remove ignored keys
        for key in ignore_keys:
            config_dict.pop(key, None)

    if args.output is None or args.output == "":
        args.output = f"config_{config.model_type}.yaml"

    yaml_output = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

    # Write to file if specified
    if args.output is not None:
        saved = False
        while not saved:
            try:
                with open(args.output, "x") as f:
                    f.write(yaml_output)
                print(f"✓ Written to: {args.output}")
                saved = True
            except FileExistsError:
                overwrite = input(f"File {args.output} already exists. Overwrite? (y/N): ")
                if overwrite.lower() == "y":
                    with open(args.output, "w") as f:
                        f.write(yaml_output)
                    print(f"✓ Written to: {args.output}")
                    saved = True
                elif not overwrite or overwrite.lower() == "n":
                    new_name = input(f"Enter a new filename for the output (or press Enter to cancel): ")
                    if new_name:
                        args.output = new_name
                    else:
                        print("Operation cancelled. Exiting.")
                        sys.exit(0)

    # Verify if requested
    # if args.verify:
    #     verify_config(config_class, yaml_output)


if __name__ == "__main__":
    main()
