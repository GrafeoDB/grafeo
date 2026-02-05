"""
Grafeo CLI - Command-line interface for Grafeo graph databases.

This module provides a Python CLI wrapper for the Grafeo admin functionality.
"""

import json
import sys
from pathlib import Path
from typing import Optional

try:
    import click
except ImportError:
    print("Error: click is required for the CLI. Install with: uv add grafeo[cli]")
    sys.exit(1)

from grafeo import GrafeoDB


def format_bytes(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    if bytes_val >= 1024**3:
        return f"{bytes_val / (1024**3):.2f} GB"
    elif bytes_val >= 1024**2:
        return f"{bytes_val / (1024**2):.2f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.2f} KB"
    else:
        return f"{bytes_val} bytes"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a simple table."""
    if not rows:
        return

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))

    # Print header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row in rows:
        row_line = " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))
        print(row_line)


def print_key_value(items: list[tuple[str, str]], as_json: bool = False) -> None:
    """Print key-value pairs."""
    if as_json:
        print(json.dumps(dict(items), indent=2))
    else:
        max_key = max(len(k) for k, _ in items)
        for key, value in items:
            print(f"{key.ljust(max_key)} : {value}")


@click.group()
@click.option("--format", "-f", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress messages")
@click.version_option()
@click.pass_context
def cli(ctx: click.Context, format: str, quiet: bool) -> None:
    """Grafeo database administration tool."""
    ctx.ensure_object(dict)
    ctx.obj["format"] = format
    ctx.obj["quiet"] = quiet


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def info(ctx: click.Context, path: str) -> None:
    """Display database information."""
    db = GrafeoDB.open(path)
    info_dict = db.info()

    as_json = ctx.obj["format"] == "json"
    if as_json:
        print(json.dumps(info_dict, indent=2, default=str))
    else:
        items = [
            ("Mode", str(info_dict.get("mode", "unknown"))),
            ("Nodes", str(info_dict.get("node_count", 0))),
            ("Edges", str(info_dict.get("edge_count", 0))),
            ("Persistent", str(info_dict.get("is_persistent", False))),
            ("Path", str(info_dict.get("path") or "(in-memory)")),
            ("WAL Enabled", str(info_dict.get("wal_enabled", False))),
            ("Version", str(info_dict.get("version", "unknown"))),
        ]
        print_key_value(items)


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def stats(ctx: click.Context, path: str) -> None:
    """Show detailed statistics."""
    db = GrafeoDB.open(path)
    stats_dict = db.detailed_stats()

    as_json = ctx.obj["format"] == "json"
    if as_json:
        print(json.dumps(stats_dict, indent=2, default=str))
    else:
        memory = stats_dict.get("memory_bytes", 0)
        disk = stats_dict.get("disk_bytes")
        items = [
            ("Nodes", str(stats_dict.get("node_count", 0))),
            ("Edges", str(stats_dict.get("edge_count", 0))),
            ("Labels", str(stats_dict.get("label_count", 0))),
            ("Edge Types", str(stats_dict.get("edge_type_count", 0))),
            ("Property Keys", str(stats_dict.get("property_key_count", 0))),
            ("Indexes", str(stats_dict.get("index_count", 0))),
            ("Memory Usage", format_bytes(memory)),
            ("Disk Usage", format_bytes(disk) if disk else "N/A"),
        ]
        print_key_value(items)


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def schema(ctx: click.Context, path: str) -> None:
    """Display schema information."""
    db = GrafeoDB.open(path)
    schema_dict = db.schema()

    as_json = ctx.obj["format"] == "json"
    if as_json:
        print(json.dumps(schema_dict, indent=2, default=str))
    else:
        mode = schema_dict.get("mode", "lpg")
        print(f"Mode: {mode.upper()}\n")

        if mode == "lpg":
            labels = schema_dict.get("labels", [])
            if labels:
                print("Labels:")
                print_table(["Name", "Count"], [[l["name"], str(l["count"])] for l in labels])
                print()

            edge_types = schema_dict.get("edge_types", [])
            if edge_types:
                print("Edge Types:")
                print_table(["Name", "Count"], [[e["name"], str(e["count"])] for e in edge_types])
                print()

            prop_keys = schema_dict.get("property_keys", [])
            if prop_keys:
                print("Property Keys:")
                for key in prop_keys:
                    print(f"  - {key}")
        else:
            predicates = schema_dict.get("predicates", [])
            if predicates:
                print("Predicates:")
                print_table(["IRI", "Count"], [[p["iri"], str(p["count"])] for p in predicates])
                print()

            graphs = schema_dict.get("named_graphs", [])
            if graphs:
                print("Named Graphs:")
                for g in graphs:
                    print(f"  - {g}")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def validate(ctx: click.Context, path: str) -> None:
    """Validate database integrity."""
    db = GrafeoDB.open(path)
    result = db.validate()

    as_json = ctx.obj["format"] == "json"
    if as_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        errors = result.get("errors", [])
        warnings = result.get("warnings", [])

        if not errors:
            click.secho("Database is valid", fg="green")
        else:
            click.secho("Database has errors", fg="red")

        print(f"\nErrors: {len(errors)}, Warnings: {len(warnings)}")

        if errors:
            print("\nErrors:")
            print_table(
                ["Code", "Message", "Context"],
                [[e["code"], e["message"], e.get("context", "-")] for e in errors],
            )

        if warnings:
            print("\nWarnings:")
            print_table(
                ["Code", "Message", "Context"],
                [[w["code"], w["message"], w.get("context", "-")] for w in warnings],
            )

        if errors:
            sys.exit(1)


@cli.group()
def wal() -> None:
    """Manage Write-Ahead Log."""
    pass


@wal.command("status")
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def wal_status(ctx: click.Context, path: str) -> None:
    """Show WAL status."""
    db = GrafeoDB.open(path)
    status = db.wal_status()

    as_json = ctx.obj["format"] == "json"
    if as_json:
        print(json.dumps(status, indent=2, default=str))
    else:
        items = [
            ("Enabled", str(status.get("enabled", False))),
            ("Path", str(status.get("path") or "N/A")),
            ("Size", format_bytes(status.get("size_bytes", 0))),
            ("Records", str(status.get("record_count", 0))),
            ("Last Checkpoint", str(status.get("last_checkpoint") or "Never")),
            ("Current Epoch", str(status.get("current_epoch", 0))),
        ]
        print_key_value(items)


@wal.command("checkpoint")
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def wal_checkpoint(ctx: click.Context, path: str) -> None:
    """Force a WAL checkpoint."""
    quiet = ctx.obj["quiet"]
    if not quiet:
        click.echo("Forcing WAL checkpoint...")

    db = GrafeoDB.open(path)
    db.wal_checkpoint()

    if not quiet:
        click.secho("WAL checkpoint completed", fg="green")


@cli.group()
def backup() -> None:
    """Manage backups."""
    pass


@backup.command("create")
@click.argument("path", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, type=click.Path(), help="Output backup path")
@click.pass_context
def backup_create(ctx: click.Context, path: str, output: str) -> None:
    """Create a database backup."""
    quiet = ctx.obj["quiet"]
    if not quiet:
        click.echo(f"Creating backup of {path}...")

    db = GrafeoDB.open(path)
    db.save(output)

    if not quiet:
        click.secho(f"Backup created at {output}", fg="green")


@backup.command("restore")
@click.argument("backup_path", type=click.Path(exists=True))
@click.argument("target_path", type=click.Path())
@click.option("--force", is_flag=True, help="Overwrite existing target")
@click.pass_context
def backup_restore(ctx: click.Context, backup_path: str, target_path: str, force: bool) -> None:
    """Restore from a backup."""
    quiet = ctx.obj["quiet"]
    target = Path(target_path)

    if target.exists() and not force:
        click.echo(f"Error: Target {target_path} already exists. Use --force to overwrite.", err=True)
        sys.exit(1)

    if target.exists() and force:
        if not quiet:
            click.echo(f"Removing existing database at {target_path}...")
        import shutil
        shutil.rmtree(target_path)

    if not quiet:
        click.echo(f"Restoring from {backup_path}...")

    db = GrafeoDB.open(backup_path)
    db.save(target_path)

    if not quiet:
        click.secho(f"Database restored to {target_path}", fg="green")


def main() -> None:
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
