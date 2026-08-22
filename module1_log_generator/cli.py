
from __future__ import annotations

import logging
import sys

import click

from module1_log_generator.config.parser import load_and_validate
from module1_log_generator.core.pipeline import GeneratorPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)


@click.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to service graph YAML configuration file.",
)
@click.option(
    "--scale",
    "-s",
    default="medium",
    type=click.Choice(["small", "medium", "large"]),
    show_default=True,
    help="Generation scale preset.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    default="parquet",
    type=click.Choice(["parquet", "csv", "ndjson"]),
    show_default=True,
    help="Output format.",
)
@click.option(
    "--output-dir",
    "-o",
    default="data/generated",
    show_default=True,
    help="Directory for generated output files.",
)
@click.option(
    "--hours",
    "-t",
    default=None,
    type=float,
    help="Override duration in hours (ignores scale preset). Not restricted "
    "to whole days — e.g. --hours 2 for a 2-hour run.",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed for reproducible generation.",
)
@click.option(
    "--visualize-only",
    is_flag=True,
    default=False,
    help="Render the service dependency graph preview HTML and exit "
    "without generating any log data.",
)
def main(
    config: str,
    scale: str,
    output_format: str,
    output_dir: str,
    hours: float | None,
    seed: int | None,
    visualize_only: bool,
) -> None:
    from module1_log_generator.config.defaults import SCALE_PRESETS

    svc_config = load_and_validate(config)

    if visualize_only:
        from pathlib import Path

        from module1_log_generator.core.service_graph import ServiceGraph
        from module1_log_generator.viz.graph_visualizer import render_service_graph_html

        graph = ServiceGraph(svc_config)
        out_path = Path(output_dir) / "service_graph.html"
        render_service_graph_html(svc_config, graph, out_path)
        logging.getLogger(__name__).info("Service graph preview written to %s", out_path)
        return

    if hours is None:
        preset = SCALE_PRESETS.get(scale, SCALE_PRESETS["medium"])
        hours = preset["duration_hours"]

    pipeline = GeneratorPipeline(
        config=svc_config,
        scale=scale,
        output_format=output_format,
        output_dir=output_dir,
        seed=seed,
    )
    pipeline.run(duration_hours=hours)


if __name__ == "__main__":
    main()
