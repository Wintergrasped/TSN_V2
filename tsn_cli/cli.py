"""
Command-line interface for TSN management.
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from tsn_common.config import get_settings
from tsn_common.db import get_session
from tsn_common.logging import get_logger, setup_logging
from tsn_common.models import AudioFile, AudioFileState, Callsign, NetSession

app = typer.Typer(help="The Spoken Network CLI")
console = Console()
logger = get_logger(__name__)


@app.command()
def init_db():
    """Initialize database tables and seed data."""
    from tsn_common.db_init import create_all_tables, seed_phonetic_corrections
    
    console.print("[bold green]Initializing database...[/bold green]")
    
    asyncio.run(create_all_tables())
    asyncio.run(seed_phonetic_corrections())
    
    console.print("[bold green]✓[/bold green] Database initialized successfully")


@app.command()
def status():
    """Show system status and processing queue."""
    
    async def get_status():
        async with get_session() as session:
            from sqlalchemy import func, select
            
            # Count files by state
            result = await session.execute(
                select(
                    AudioFile.state,
                    func.count(AudioFile.id).label("count"),
                )
                .group_by(AudioFile.state)
            )
            
            state_counts = {row.state: row.count for row in result.all()}
            
            # Count callsigns
            result = await session.execute(select(func.count(Callsign.id)))
            callsign_count = result.scalar()
            
            # Count nets
            result = await session.execute(select(func.count(NetSession.id)))
            net_count = result.scalar()
            
            return state_counts, callsign_count, net_count
    
    state_counts, callsign_count, net_count = asyncio.run(get_status())
    
    # Display results
    console.print("\n[bold]TSN System Status[/bold]\n")
    
    # Files by state
    table = Table(title="Audio Files by State")
    table.add_column("State", style="cyan")
    table.add_column("Count", style="magenta", justify="right")
    
    for state in AudioFileState:
        count = state_counts.get(state, 0)
        if count > 0:
            table.add_row(state.value, str(count))
    
    console.print(table)
    
    # Summary
    console.print(f"\n[bold]Total Callsigns:[/bold] {callsign_count}")
    console.print(f"[bold]Total Net Sessions:[/bold] {net_count}")


@app.command()
def list_callsigns(
    limit: int = typer.Option(20, help="Number of callsigns to show"),
    validated_only: bool = typer.Option(False, help="Show only validated callsigns"),
):
    """List callsigns in database."""
    
    async def get_callsigns():
        async with get_session() as session:
            from sqlalchemy import select
            
            query = select(Callsign).order_by(Callsign.seen_count.desc()).limit(limit)
            
            if validated_only:
                query = query.where(Callsign.validated == True)
            
            result = await session.execute(query)
            return list(result.scalars().all())
    
    callsigns = asyncio.run(get_callsigns())
    
    table = Table(title="Callsigns")
    table.add_column("Callsign", style="cyan")
    table.add_column("Validated", style="green")
    table.add_column("Seen Count", style="magenta", justify="right")
    table.add_column("Last Seen", style="yellow")
    
    for cs in callsigns:
        validated = "✓" if cs.validated else "✗"
        table.add_row(
            cs.callsign,
            validated,
            str(cs.seen_count),
            cs.last_seen.strftime("%Y-%m-%d %H:%M"),
        )
    
    console.print(table)


@app.command()
def list_nets(
    limit: int = typer.Option(10, help="Number of nets to show"),
):
    """List recent net sessions."""
    
    async def get_nets():
        async with get_session() as session:
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload
            
            result = await session.execute(
                select(NetSession)
                .options(selectinload(NetSession.ncs_callsign))
                .order_by(NetSession.start_time.desc())
                .limit(limit)
            )
            return list(result.scalars().all())
    
    nets = asyncio.run(get_nets())
    
    table = Table(title="Recent Net Sessions")
    table.add_column("Net Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("NCS", style="magenta")
    table.add_column("Start Time", style="yellow")
    
    for net in nets:
        ncs = net.ncs_callsign.callsign if net.ncs_callsign else "Unknown"
        table.add_row(
            net.net_name,
            net.net_type,
            ncs,
            net.start_time.strftime("%Y-%m-%d %H:%M"),
        )
    
    console.print(table)


@app.command()
def reprocess(
    audio_file_id: str = typer.Argument(..., help="Audio file UUID to reprocess"),
    from_state: str = typer.Option(
        "transcribed",
        help="Reset to this state (transcribed, callsigns_extracted)",
    ),
):
    """Reprocess a specific audio file."""
    
    async def reset_file():
        async with get_session() as session:
            from uuid import UUID
            
            file_id = UUID(audio_file_id)
            audio_file = await session.get(AudioFile, file_id)
            
            if not audio_file:
                return False
            
            # Reset state
            if from_state == "transcribed":
                audio_file.state = AudioFileState.QUEUED_EXTRACTION
            elif from_state == "callsigns_extracted":
                audio_file.state = AudioFileState.QUEUED_ANALYSIS
            else:
                return False
            
            audio_file.retry_count = 0
            
            return True
    
    success = asyncio.run(reset_file())
    
    if success:
        console.print(f"[bold green]✓[/bold green] File {audio_file_id} reset to {from_state}")
    else:
        console.print(f"[bold red]✗[/bold red] File not found or invalid state")


@app.command()
def clean_failed(
    dry_run: bool = typer.Option(True, help="Show what would be deleted without deleting"),
):
    """Clean up failed processing entries."""
    
    async def get_failed():
        async with get_session() as session:
            from sqlalchemy import select
            
            result = await session.execute(
                select(AudioFile).where(
                    AudioFile.state.in_([
                        AudioFileState.FAILED_TRANSCRIPTION,
                        AudioFileState.FAILED_EXTRACTION,
                        AudioFileState.FAILED_ANALYSIS,
                    ])
                )
            )
            
            failed_files = list(result.scalars().all())
            
            if not dry_run:
                for file in failed_files:
                    await session.delete(file)
            
            return len(failed_files)
    
    count = asyncio.run(get_failed())
    
    if dry_run:
        console.print(f"[yellow]Would delete {count} failed files (use --no-dry-run to confirm)[/yellow]")
    else:
        console.print(f"[bold green]✓[/bold green] Deleted {count} failed files")


@app.command()
def profile(
    callsign: str = typer.Argument(..., help="Callsign to view profile"),
):
    """View callsign profile."""
    
    async def get_profile():
        async with get_session() as session:
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload
            
            result = await session.execute(
                select(Callsign)
                .options(selectinload(Callsign.profile))
                .where(Callsign.callsign == callsign.upper())
            )
            
            cs = result.scalar_one_or_none()
            return cs
    
    cs = asyncio.run(get_profile())
    
    if not cs:
        console.print(f"[bold red]✗[/bold red] Callsign {callsign} not found")
        return
    
    console.print(f"\n[bold cyan]{cs.callsign}[/bold cyan]")
    console.print(f"Validated: {'✓' if cs.validated else '✗'}")
    console.print(f"Seen: {cs.seen_count} times")
    console.print(f"First seen: {cs.first_seen.strftime('%Y-%m-%d %H:%M')}")
    console.print(f"Last seen: {cs.last_seen.strftime('%Y-%m-%d %H:%M')}")
    
    if cs.profile and cs.profile.ai_summary:
        console.print(f"\n[bold]Profile:[/bold]")
        console.print(cs.profile.ai_summary)


def main():
    """Main entry point."""
    settings = get_settings()
    setup_logging(settings.logging)
    app()


if __name__ == "__main__":
    main()
