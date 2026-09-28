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
    config_dict = config.to_dict()

    # Remove `transformers_version` if present
    if "transformers_version" in config_dict:
        del config_dict["transformers_version"]

    if args.output is None or args.output == "":
        args.output = f"config_{config_dict.get('model_type', 'default')}.yaml"

    def wizard(config_dict, depth=0):
        output_indent = "  " * depth
        new_config_dict = {}
        for key, value in config_dict.items():
            try:
                key_doc = config.__dataclass_fields__.get(key).__doc__
                if key_doc is not None and "The type of the None singleton." not in key_doc:
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
                ret = wizard(value, depth=depth+1)
                if len(ret) > 0:
                    new_config_dict[key] = ret
                continue

            # handle lists
            if isinstance(value, list) or "list[str]" in str(value_type_hint):
                print(f"{output_indent}{key} is a list. Please enter comma-separated values.")
                user_input = input(f"{output_indent}{key} (default: {value}): ")
                if user_input:
                    new_config_dict[key] = [item.strip() for item in user_input.split(",")]
                else:
                    if args.keep_defaults:
                        new_config_dict[key] = value
                continue

            # handle booleans
            if isinstance(value, bool):
                user_input = input(f"{output_indent}{key} (default: {value}) [y/n]: ")
                if user_input.lower() in ["y", "yes", "t", "true"]:
                    new_config_dict[key] = True
                elif user_input.lower() in ["n", "no", "f", "false"]:
                    new_config_dict[key] = False
                else:
                    if args.keep_defaults:
                        new_config_dict[key] = value
                continue

            user_input = input(f"{output_indent}{key} (default: {value}): ")
            if user_input:
                new_config_dict[key] = user_input
            else:
                if args.keep_defaults:
                    new_config_dict[key] = value

        return new_config_dict
    
    # Run wizard mode if specified
    if args.wizard:
        print(f"Running in wizard mode for {args.config_class}. Press Enter to keep default values.")
        config_dict = wizard(config_dict)

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
