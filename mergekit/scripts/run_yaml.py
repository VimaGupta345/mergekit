# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

import logging

import click
import yaml
import time

from mergekit.config import MergeConfiguration
from mergekit.merge import run_merge
from mergekit.options import MergeOptions, add_merge_options

import torch
import torch.cuda

from mergekit.metric_logging import TaskLoggingContextManagerGPU, MemoryLoggingContextManager

@click.command("mergekit-yaml")
@click.argument("config_file")
@click.argument("out_path")
@click.option(
    "--verbose", "-v", type=bool, default=False, is_flag=True, help="Verbose logging"
)
@add_merge_options
def main(
    merge_options: MergeOptions,
    config_file: str,
    out_path: str,
    verbose: bool,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    with open(config_file, "r", encoding="utf-8") as file:
        config_source = file.read()

    merge_config: MergeConfiguration = MergeConfiguration.model_validate(
        yaml.safe_load(config_source)
    )

    torch.cuda.reset_peak_memory_stats()

    # compute time taken in this step
    # Start timing
    #start_time = time.time()
    with TaskLoggingContextManagerGPU(task_type="MERGE_TOTAL"):
        with MemoryLoggingContextManager():
            run_merge(
                merge_config,
                out_path,
                options=merge_options,
                config_source=config_source,
            )

if __name__ == "__main__":
    main()
