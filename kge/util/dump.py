import time
import os
from collections import OrderedDict
import sys
import torch
import csv
import yaml
import re
import socket
import copy

from kge.job import Trace
from kge import Config


## EXPORTED METHODS #####################################################################


def add_dump_parsers(subparsers):
    # 'kge dump' can have associated sub-commands which can have different args
    parser_dump = subparsers.add_parser("dump", help="Dump objects to stdout")
    subparsers_dump = parser_dump.add_subparsers(
        title="dump_command", dest="dump_command"
    )
    subparsers_dump.required = True
    _add_dump_trace_parser(subparsers_dump)
    _add_dump_checkpoint_parser(subparsers_dump)
    _add_dump_config_parser(subparsers_dump)


def dump(args):
    """Execute the 'kge dump' commands. """
    if args.dump_command == "trace":
        _dump_trace(args)
    elif args.dump_command == "checkpoint":
        _dump_checkpoint(args)
    elif args.dump_command == "config":
        _dump_config(args)
    else:
        raise ValueError()


def get_config_for_job_id(job_id, folder_path):
    config = Config(load_default=True)
    if job_id:
        config_path = os.path.join(
            folder_path, "config", job_id.split("-")[0] + ".yaml"
        )
    else:
        config_path = os.path.join(folder_path, "config.yaml")
    if os.path.isfile(config_path):
        config.load(config_path, create=True)
    else:
        raise Exception("Could not find config file for {}".format(job_id))
    return config


### DUMP CHECKPOINT #####################################################################


def _add_dump_checkpoint_parser(subparsers_dump):
    parser_dump_checkpoint = subparsers_dump.add_parser(
        "checkpoint", help=("Dump information stored in a checkpoint")
    )
    parser_dump_checkpoint.add_argument(
        "source",
        help="A path to either a checkpoint or a job folder (then uses best or, "
        "if not present, last checkpoint).",
        nargs="?",
        default=".",
    )
    parser_dump_checkpoint.add_argument(
        "--keys",
        "-k",
        type=str,
        nargs="*",
        help="List of keys to include (separated by space)",
    )


def _dump_checkpoint(args):
    """Execute the 'dump checkpoint' command."""

    # Determine checkpoint to use
    if os.path.isfile(args.source):
        checkpoint_file = args.source
    else:
        checkpoint_file = Config.get_best_or_last_checkpoint(args.source)

    # Load the checkpoint and strip some fieleds
    checkpoint = torch.load(checkpoint_file, map_location="cpu")

    # Dump it
    print(f"# Dump of checkpoint: {checkpoint_file}")
    excluded_keys = {"model", "optimizer_state_dict"}
    if args.keys is not None:
        excluded_keys = {key for key in excluded_keys if key not in args.keys}
        excluded_keys = excluded_keys.union(
            {key for key in checkpoint if key not in args.keys}
        )
    excluded_keys = {key for key in excluded_keys if key in checkpoint}
    for key in excluded_keys:
        del checkpoint[key]
    if excluded_keys:
        print(f"# Excluded keys: {excluded_keys}")
    yaml.dump(checkpoint, sys.stdout)


### DUMP TRACE ##########################################################################


def _add_dump_trace_parser(subparsers_dump):
    parser_dump_trace = subparsers_dump.add_parser(
        "trace",
        help=(
            "Dump trace to csv (default) and/or stdout. The tracefile will be processed"
            " backwards starting from the last entry. Further options allow to start"
            " processing from a particular checkpoint or job_id or epoch_number."
        ),
    )
    parser_dump_trace.add_argument(
        "source",
        help="A path to either a checkpoint or a job folder.",
        nargs="?",
        default=".",
    )
    parser_dump_trace.add_argument(
        "--train",
        action="store_const",
        const=True,
        default=False,
        help=(
            "Include entries from training jobs (enabled when none of --train, --valid,"
            " or --test is specified)."
        ),
    )
    parser_dump_trace.add_argument(
        "--valid",
        action="store_const",
        const=True,
        default=False,
        help=(
            "Include entries from validation during training and evaluation on"
            " the validation data split (enabled when none of --train, --valid, or"
            " --test is specified)."
        ),
    )
    parser_dump_trace.add_argument(
        "--test",
        action="store_const",
        const=True,
        default=False,
        help=(
            "Include entries from evaluation on the test data split (enabled when "
            "  none of --train, --valid, or --test is specified)."
        ),
    )
    parser_dump_trace.add_argument(
        "--search",
        action="store_const",
        const=True,
        default=False,
        help="Dump the tracefile of a search job. The options --train, --valid, and"
             " --test are not applicable.",
    )
    parser_dump_trace.add_argument(
        "--keysfile",
        default=False,
        help=(
            "A path to a file which contains lines in the format"
            " 'new_key_name'='key_name'. For every line in the keys file, the command"
            " will look for the value of 'key_name' in the trace entries and config and"
            " add a respective column in the csv file with name 'new_key_name'."
            " Additionally, for 'key_name' the special keys '$folder', '$machine'"
            " '$checkpoint' and '$base_model' can be used. "
        ),
    )
    parser_dump_trace.add_argument(
        "--keys",
        "-k",
        nargs="*",
        type=str,
        help=(
            "A list of 'key' entries (separated by space). Each 'key' has form"
            " 'new_key_name=key_name' or 'key_name'. This will add columns as in the"
            " --keysfile option. When only 'key_name' is provided, it is also used as"
            " column name."
        ),
    )
    parser_dump_trace.add_argument(
        "--checkpoint",
        default=False,
        action="store_const",
        const=True,
        help=(
            "If source is a path to a job folder and --checkpoint is set, the best"
            " (if present) or last checkpoint will be used to determine the job_id from"
            " where the tracefile will be processed backwards."
        ),
    )
    parser_dump_trace.add_argument(
        "--job_id",
        default=False,
        help=(
            "Specifies the training job id in the trace"
            " from where to start processing the tracefile backward."
        ),
    )
    parser_dump_trace.add_argument(
        "--max_epoch",
        default=False,
        help=(
            "Specifies the max epoch number in the tracefile"
            " from where to start processing backwards. Can not be used with --truncate"
            " when a checkpoint is provided."
        ),
    )
    parser_dump_trace.add_argument(
        "--truncate",
        default=False,
        action="store_const",
        const=True,
        help=(
            "When a checkpoint is used (by providing one explicitly as source or by"
            " using --checkpoint), --truncate will define the max_epoch number to"
            " process as provided by the checkpoint. Can only be used when a checkpoint"
            " is provided and it cannot be used together with --max_epoch."
        ),
    )
    parser_dump_trace.add_argument(
        "--yaml",
        action="store_const",
        const=True,
        default=False,
        help="Instead of using a CSV file with a subset of entries, dump the full"
             " trace file. Additional entries from the config can be added with the"
             " --keysfile and --keys option."
    )
    parser_dump_trace.add_argument(
        "--batch",
        action="store_const",
        const=True,
        default=False,
        help="Include entries on batch level.",
    )
    parser_dump_trace.add_argument(
        "--example",
        action="store_const",
        const=True,
        default=False,
        help="Include entries on example level.",
    )
    parser_dump_trace.add_argument(
        "--no-header",
        action="store_const",
        const=True,
        default=False,
        help="Exclude column names (header) from the csv file.",
    )
    parser_dump_trace.add_argument(
        "--no-default-keys",
        "-K",
        action="store_const",
        const=True,
        default=False,
        help="Exclude default keys from the csv file.",
    )


def _dump_trace(args):
    """Execute the 'dump trace' command."""
    if (args.train or args.valid or args.test) and args.search:
        print(
            "--search and --train, --valid, --test are mutually exclusive",
            file=sys.stderr,
        )
        exit(1)
    if args.max_epoch and args.truncate:
        print(
            "--max epoch and --truncate cannot be used together",
            file=sys.stderr
        )
        exit(1)
    entry_type_specified = True
    if not (args.train or args.valid or args.test or args.search):
        entry_type_specified = False
        args.train = True
        args.valid = True
        args.test = True

    checkpoint_path = None
    if ".pt" in os.path.split(args.source)[-1]:
        checkpoint_path = args.source
        folder_path = os.path.split(args.source)[0]
    else:
        # determine job_id and epoch from last/best checkpoint automatically
        if args.checkpoint:
            checkpoint_path = Config.get_best_or_last_checkpoint(args.source)
        folder_path = args.source
        if not args.checkpoint and args.truncate:
            print(
                "--truncate can only be used when a checkpoint is specified."
                "Consider using --checkpoint or provide a checkpoint file as source",
                file=sys.stderr
            )
            exit(1)
    trace = os.path.join(folder_path, "trace.yaml")
    if not os.path.isfile(trace):
        sys.stderr.write("No trace found at {}\n".format(trace))
        exit(1)
    keymap = OrderedDict()
    additional_keys = []
    if args.keysfile:
        with open(args.keysfile, "r") as keyfile:
            additional_keys = keyfile.readlines()
    if args.keys:
        additional_keys += args.keys
    for line in additional_keys:
        line = line.rstrip("\n").replace(" ", "")
        name_key = line.split("=")
        if len(name_key) == 1:
            name_key += name_key
        keymap[name_key[0]] = name_key[1]

    job_id = None
    epoch = int(args.max_epoch)
    # use job_id and epoch from checkpoint
    if checkpoint_path and args.truncate:
        checkpoint = torch.load(f=checkpoint_path, map_location="cpu")
        job_id = checkpoint["job_id"]
        epoch = checkpoint["epoch"]
    # only use job_id from checkpoint
    elif checkpoint_path:
        checkpoint = torch.load(f=checkpoint_path, map_location="cpu")
        job_id = checkpoint["job_id"]
    # override job_id and epoch with user arguments
    if args.job_id:
        job_id = args.job_id
    if not epoch:
        epoch = float("inf")

    entries, job_epochs = [], {}
    if not args.search:
        entries, job_epochs = Trace.grep_training_trace_entries(
            tracefile=trace,
            train=args.train,
            test=args.test,
            valid=args.valid,
            example=args.example,
            batch=args.batch,
            job_id=job_id,
            epoch_of_last=epoch,
        )
    if not entries and (args.search or not entry_type_specified):
        entries = Trace.grep_entries(tracefile=trace, conjunctions=[f"scope: train"])
        epoch = None
        if entries:
            args.search = True
    if not entries:
        print("No relevant trace entries found.", file=sys.stderr)
        exit(1)

    if not args.yaml:
        csv_writer = csv.writer(sys.stdout)
        # dict[new_name] = (lookup_name, where)
        # if where=="config"/"trace" it will be looked up automatically
        # if where=="sep" it must be added in in the write loop separately
        if args.no_default_keys:
            default_attributes = OrderedDict()
        else:
            default_attributes = OrderedDict(
                [
                    ("job_id", ("job_id", "sep")),
                    ("dataset", ("dataset.name", "config")),
                    ("model", ("model", "sep")),
                    ("reciprocal", ("reciprocal", "sep")),
                    ("job", ("job", "sep")),
                    ("job_type", ("type", "trace")),
                    ("split", ("split", "sep")),
                    ("epoch", ("epoch", "trace")),
                    ("avg_loss", ("avg_loss", "trace")),
                    ("avg_penalty", ("avg_penalty", "trace")),
                    ("avg_cost", ("avg_cost", "trace")),
                    ("metric_name", ("valid.metric", "config")),
                    ("metric", ("metric", "sep")),
                ]
            )
            if args.search:
                default_attributes["child_folder"] = ("folder", "trace")
                default_attributes["child_job_id"] = ("child_job_id", "sep")

        if not args.no_header:
            csv_writer.writerow(
                list(default_attributes.keys()) + [key for key in keymap.keys()]
            )
    # store configs for job_id's s.t. they need to be loaded only once
    configs = {}
    warning_shown = False
    for entry in entries:
        if epoch and not entry.get("epoch") <= float(epoch):
            continue
        # filter out not needed entries from a previous job when
        # a job was resumed from the middle
        if entry.get("job") == "train":
            job_id = entry.get("job_id")
            if entry.get("epoch") > job_epochs[job_id]:
                continue

        # find relevant config file
        child_job_id = entry.get("child_job_id") if "child_job_id" in entry else None
        config_key = (
            entry.get("folder") + "/" + str(child_job_id)
            if args.search
            else entry.get("job_id")
        )
        if config_key in configs.keys():
            config = configs[config_key]
        else:
            if args.search:
                if not child_job_id and not warning_shown:
                    # This warning is from Dec 19, 2019. TODO remove
                    print(
                        "Warning: You are dumping the trace of an older search job. "
                        "This is fine only if "
                        "the config.yaml files in each subfolder have not been modified "
                        "after running the corresponding training job.",
                        file=sys.stderr,
                    )
                    warning_shown = True
                config = get_config_for_job_id(
                    child_job_id, os.path.join(folder_path, entry.get("folder"))
                )
                entry["type"] = config.get("train.type")
            else:
                config = get_config_for_job_id(entry.get("job_id"), folder_path)
            configs[config_key] = config

        new_attributes = OrderedDict()
        if config.get_default("model") == "reciprocal_relations_model":
            model = config.get_default("reciprocal_relations_model.base_model.type")
            # the string that substitutes $base_model in keymap if it exists
            subs_model = "reciprocal_relations_model.base_model"
            reciprocal = 1
        else:
            model = config.get_default("model")
            subs_model = model
            reciprocal = 0
        for new_key in keymap.keys():
            lookup = keymap[new_key]
            if "$base_model" in lookup:
                lookup = lookup.replace("$base_model", subs_model)
            try:
                if lookup == "$folder":
                    val = os.path.abspath(folder_path)
                elif lookup == "$checkpoint":
                    val = os.path.abspath(checkpoint_path)
                elif lookup == "$machine":
                    val = socket.gethostname()
                else:
                    val = config.get_default(lookup)
            except:
                # creates empty field if key is not existing
                val = entry.get(lookup)
            if type(val) == bool and val:
                val = 1
            elif type(val) == bool and not val:
                val = 0
            new_attributes[new_key] = val
        if not args.yaml:
            # find the actual values for the default attributes
            actual_default = default_attributes.copy()
            for new_key in default_attributes.keys():
                lookup, where = default_attributes[new_key]
                if where == "config":
                    actual_default[new_key] = config.get(lookup)
                elif where == "trace":
                    actual_default[new_key] = entry.get(lookup)
            # keys with separate treatment
            # "split" in {train,test,valid} for the datatype
            # "job" in {train,eval,valid,search}
            if entry.get("job") == "train":
                if "split" in entry:
                    actual_default["split"] = entry.get("split")
                else:
                    actual_default["split"] = "train"
                actual_default["job"] = "train"
            elif entry.get("job") == "eval":
                if "split" in entry:
                    actual_default["split"] = entry.get("split")  # test or valid
                else:
                    # deprecated
                    actual_default["split"] = entry.get("data")  # test or valid
                if entry.get("resumed_from_job_id"):
                    actual_default["job"] = "eval"  # from "kge eval"
                else:
                    actual_default["job"] = "valid"  # child of training job
            else:
                actual_default["job"] = entry.get("job")
                if "split" in entry:
                    actual_default["split"] = entry.get("split")
                else:
                    # deprecated
                    actual_default["split"] = entry.get("data")  # test or valid
            actual_default["job_id"] = entry.get("job_id").split("-")[0]
            actual_default["model"] = model
            actual_default["reciprocal"] = reciprocal
            # lookup name is in config value is in trace
            actual_default["metric"] = entry.get(config.get_default("valid.metric"))
            if args.search:
                actual_default["child_job_id"] = entry.get("child_job_id").split("-")[0]
            for key in list(actual_default.keys()):
                if key not in default_attributes:
                    del actual_default[key]
            csv_writer.writerow(
                [actual_default[new_key] for new_key in actual_default.keys()]
                + [new_attributes[new_key] for new_key in new_attributes.keys()]
            )
        else:
            entry.update({"reciprocal": reciprocal, "model": model})
            if keymap:
                entry.update(new_attributes)
            sys.stdout.write(re.sub("[{}']", "", str(entry)))
            sys.stdout.write("\n")


### DUMP CONFIG ########################################################################


def _add_dump_config_parser(subparsers_dump):
    parser_dump_config = subparsers_dump.add_parser(
        "config", help=("Dump a configuration")
    )
    parser_dump_config.add_argument(
        "source",
        help="A path to either a checkpoint, a config file, or a job folder.",
        nargs="?",
        default=".",
    )

    parser_dump_config.add_argument(
        "--minimal",
        "-m",
        default=False,
        action="store_const",
        const=True,
        help="Only dump configuration options different from the default configuration (default)",
    )
    parser_dump_config.add_argument(
        "--raw",
        "-r",
        default=False,
        action="store_const",
        const=True,
        help="Dump the config as is",
    )
    parser_dump_config.add_argument(
        "--full",
        "-f",
        default=False,
        action="store_const",
        const=True,
        help="Add all values from the default configuration before dumping the config",
    )

    parser_dump_config.add_argument(
        "--include",
        "-i",
        type=str,
        nargs="*",
        help="List of keys to include (separated by space). "
        "All subkeys are also included. Cannot be used with --raw.",
    )

    parser_dump_config.add_argument(
        "--exclude",
        "-e",
        type=str,
        nargs="*",
        help="List of keys to exclude (separated by space). "
        "All subkeys are also exluded. Applied after --include. "
        "Cannot be used with --raw.",
    )


def _dump_config(args):
    """Execute the 'dump config' command."""
    if not (args.raw or args.full or args.minimal):
        args.minimal = True

    if args.raw + args.full + args.minimal != 1:
        raise ValueError("Exactly one of --raw, --full, or --minimal must be set")

    if args.raw and (args.include or args.exclude):
        raise ValueError(
            "--include and --exclude cannot be used with --raw "
            "(use --full or --minimal instead)."
        )

    config = Config()
    config_file = None
    if os.path.isdir(args.source):
        config_file = os.path.join(args.source, "config.yaml")
        config.load(config_file)
    elif ".yaml" in os.path.split(args.source)[-1]:
        config_file = args.source
        config.load(config_file)
    else:  # a checkpoint
        checkpoint_file = torch.load(args.source, map_location="cpu")
        if args.raw:
            config = checkpoint_file["config"]
        else:
            config.load_options(checkpoint_file["config"].options)

    def print_options(options):
        # drop all arguments that are not included
        if args.include:
            args.include = set(args.include)
            options_copy = copy.deepcopy(options)
            for key in options_copy.keys():
                prefix = key
                keep = False
                while True:
                    if prefix in args.include:
                        keep = True
                        break
                    else:
                        last_dot_index = prefix.rfind(".")
                        if last_dot_index < 0:
                            break
                        else:
                            prefix = prefix[:last_dot_index]
                if not keep:
                    del options[key]

        # remove all arguments that are excluded
        if args.exclude:
            args.exclude = set(args.exclude)
            options_copy = copy.deepcopy(options)
            for key in options_copy.keys():
                prefix = key
                while True:
                    if prefix in args.exclude:
                        del options[key]
                        break
                    else:
                        last_dot_index = prefix.rfind(".")
                        if last_dot_index < 0:
                            break
                        else:
                            prefix = prefix[:last_dot_index]

        # convert the remaining options to a Config and print it
        config = Config(load_default=False)
        config.set_all(options, create=True)
        print(yaml.dump(config.options))

    if args.raw:
        if config_file:
            with open(config_file, "r") as f:
                print(f.read())
        else:
            print_options(config.options)
    elif args.full:
        print_options(config.options)
    else:  # minimal
        default_config = Config()
        imports = config.get("import")
        if imports is not None:
            if not isinstance(imports, list):
                imports = [imports]
            for module_name in imports:
                default_config._import(module_name)
        default_options = Config.flatten(default_config.options)
        new_options = Config.flatten(config.options)
        minimal_options = {}

        for option, value in new_options.items():
            if option not in default_options or default_options[option] != value:
                minimal_options[option] = value

        print_options(minimal_options)
