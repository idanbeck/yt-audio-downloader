#!/usr/bin/env python3
"""YouTube Audio Downloader - Downloads audio from YouTube URLs in a YAML file."""

import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

console = Console()


def check_dependency(name: str) -> bool:
    """Check if a command-line tool is installed."""
    return shutil.which(name) is not None


def get_ytdlp_version() -> str | None:
    """Get installed yt-dlp version."""
    try:
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    except:
        return None


def sanitize_filename(name: str) -> str:
    """Remove characters that are invalid in filenames."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "")
    return name.strip()


def diagnose_error(stderr: str, stdout: str) -> str:
    """Parse yt-dlp error and return a human-readable diagnosis."""
    combined = f"{stderr} {stdout}".lower()
    
    error_patterns = {
        "video unavailable": "Video deleted, private, or doesn't exist",
        "private video": "Video is private",
        "sign in to confirm your age": "Age-restricted (requires cookies)",
        "sign in to confirm you": "Bot detection (requires cookies)",
        "this video is not available": "Geo-blocked or region-restricted",
        "copyright claim": "Removed due to copyright",
        "terminated": "Channel/account terminated",
        "this live event will begin": "Scheduled livestream",
        "premieres in": "Premiere hasn't started",
        "members-only": "Members-only content",
        "429": "Rate limited (wait and retry)",
        "403": "Forbidden (try with cookies)",
        "no js runtime": "Deno/JS runtime not found",
        "javascript runtime": "Deno/JS runtime not found",
        "deno": "Deno runtime issue",
    }
    for pattern, diagnosis in error_patterns.items():
        if pattern in combined:
            return diagnosis
    
    # Return actual error lines if unknown
    for line in stderr.split("\n"):
        if "error" in line.lower():
            return line.strip()[:100]
    
    # Check stdout for errors too
    for line in stdout.split("\n"):
        if "error" in line.lower():
            return line.strip()[:100]
    
    return f"Unknown: {stderr[:200]}" if stderr else "Unknown error (check -v for details)"


def download_audio(
    url: str, 
    output_name: str, 
    output_dir: Path, 
    cookies_from: str | None = None,
    verbose: bool = False
) -> tuple[bool, str]:
    """Download audio from YouTube URL using yt-dlp."""
    safe_name = sanitize_filename(output_name)
    # Ensure output_dir is absolute to avoid any path issues
    output_dir = output_dir.resolve()
    output_template = str(output_dir / f"{safe_name}.%(ext)s")
    final_path = output_dir / f"{safe_name}.mp3"
    
    if verbose:
        print(f"    [DEBUG] Output template: {output_template}")

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "192K",
        "--postprocessor-args", "ffmpeg:-ab 192k",
        "-o", output_template,
        "--no-playlist",
        "--embed-thumbnail",
        "--add-metadata",
        "--progress",
        "--newline",
        "--no-check-certificates",
    ]
    
    # Add cookie extraction from browser
    if cookies_from:
        cmd.extend(["--cookies-from-browser", cookies_from])
    
    if verbose:
        cmd.append("--verbose")
    
    cmd.append(url)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            if final_path.exists():
                return True, ""
            for ext in [".m4a", ".webm", ".opus", ".wav"]:
                alt_path = output_dir / f"{safe_name}{ext}"
                if alt_path.exists():
                    return False, f"Not converted to MP3 (got {ext}). Check ffmpeg."
            return True, ""

        diagnosis = diagnose_error(result.stderr, result.stdout)
        error_msg = diagnosis
        if verbose:
            error_msg += f"\n\nSTDERR:\n{result.stderr}\n\nSTDOUT:\n{result.stdout}"
        return False, error_msg

    except subprocess.TimeoutExpired:
        return False, "Timeout (>5min)"
    except FileNotFoundError:
        return False, "yt-dlp not found"
    except Exception as e:
        return False, f"Exception: {e}"


def find_yaml_file(directory: Path) -> Path | None:
    """Find and select a YAML file in the given directory."""
    yaml_files = list(directory.glob("*.yaml")) + list(directory.glob("*.yml"))
    if not yaml_files:
        return None
    if len(yaml_files) == 1:
        return yaml_files[0]

    console.print("\n[yellow]Multiple YAML files found:[/yellow]")
    for i, f in enumerate(yaml_files):
        console.print(f"  {i + 1}. {f.name}")
    choice = console.input("Select number (Enter for first): ").strip()
    if choice:
        return yaml_files[int(choice) - 1]
    return yaml_files[0]


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download audio from YouTube URLs in YAML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  yt-download                        # Basic usage
  yt-download -c chrome              # Use Chrome cookies (defeats bot detection)
  yt-download -c firefox             # Use Firefox cookies
  yt-download -v                     # Verbose error output
        """
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show full error output")
    parser.add_argument("-f", "--file", type=str, help="Specify YAML file path")
    parser.add_argument(
        "-c", "--cookies", type=str, metavar="BROWSER",
        help="Extract cookies from browser: chrome, firefox, brave, edge, safari, opera"
    )
    args = parser.parse_args()

    console.print(Panel.fit("[bold blue]YouTube Audio Downloader[/bold blue]", border_style="blue"))

    # Check dependencies
    errors = []
    
    # Check ffmpeg
    if not check_dependency("ffmpeg"):
        errors.append(("ffmpeg", "Required for MP3 conversion", [
            "brew install ffmpeg",
            "sudo apt install ffmpeg", 
            "choco install ffmpeg"
        ]))
    else:
        console.print("[green]✓[/green] ffmpeg found")
    
    # Check Deno (required for YouTube as of Nov 2025)
    if not check_dependency("deno"):
        errors.append(("deno", "Required for YouTube downloads (since Nov 2025)", [
            "curl -fsSL https://deno.land/install.sh | sh",
            "brew install deno",
            "choco install deno"
        ]))
    else:
        console.print("[green]✓[/green] deno found")
    
    # Check yt-dlp version
    ytdlp_version = get_ytdlp_version()
    if ytdlp_version:
        console.print(f"[green]✓[/green] yt-dlp version: [cyan]{ytdlp_version}[/cyan]")
        # Warn if old version
        if "2024" in ytdlp_version or "2025.01" in ytdlp_version or "2025.02" in ytdlp_version:
            console.print("[yellow]  ⚠ Your yt-dlp may be outdated. Run: [cyan]pip install -U yt-dlp[default][/cyan][/yellow]")
    else:
        errors.append(("yt-dlp", "YouTube downloader", ["pip install -U 'yt-dlp[default]'"]))
    
    if errors:
        console.print()
        console.print(Panel(
            "\n".join([
                f"[red]✗ {name}[/red] - {desc}\n  Install:\n" + 
                "\n".join([f"    [cyan]{cmd}[/cyan]" for cmd in cmds])
                for name, desc, cmds in errors
            ]),
            title="Missing Dependencies",
            border_style="red"
        ))
        sys.exit(1)

    if args.cookies:
        console.print(f"[green]✓[/green] Using cookies from: [cyan]{args.cookies}[/cyan]")
        console.print("  [dim](Make sure the browser is closed)[/dim]")

    current_dir = Path.cwd()

    if args.file:
        yaml_file = Path(args.file)
        if not yaml_file.exists():
            console.print(f"[red]Error: File not found: {args.file}[/red]")
            sys.exit(1)
    else:
        yaml_file = find_yaml_file(current_dir)
        if not yaml_file:
            console.print("[red]Error: No YAML file found in current directory.[/red]")
            sys.exit(1)

    console.print(f"[green]✓[/green] Using: [cyan]{yaml_file.name}[/cyan]")

    # Create output folder with same name as YAML file (without extension)
    output_dir = current_dir / yaml_file.stem
    output_dir.mkdir(exist_ok=True)
    console.print(f"[green]✓[/green] Output folder: [cyan]{output_dir.name}/[/cyan]")

    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    sources = data.get("sources", [])
    if not sources:
        console.print("[red]No sources found in YAML file.[/red]")
        sys.exit(1)

    console.print(f"[green]✓[/green] Found [bold]{len(sources)}[/bold] sources\n")

    successful = 0
    failed_items = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        overall = progress.add_task("[cyan]Overall progress", total=len(sources))

        for i, source in enumerate(sources, 1):
            name = source.get("name_en") or source.get("name", "Unknown")
            artist = source.get("user_en") or source.get("user", "Unknown")
            url = source.get("url")
            output_name = f"{artist} - {name}"

            display_name = output_name[:50] + "..." if len(output_name) > 50 else output_name
            progress.update(overall, description=f"[cyan]{display_name}")

            if not url:
                progress.console.print(f"  [yellow]⊘ Skipped (no URL):[/yellow] {display_name}")
                failed_items.append({"name": name, "artist": artist, "url": "N/A", "reason": "No URL"})
                progress.advance(overall)
                continue

            # Pass output_dir (the subfolder) not current_dir
            success, error_msg = download_audio(
                url=url, 
                output_name=output_name, 
                output_dir=output_dir,  # This is the subfolder!
                cookies_from=args.cookies,
                verbose=args.verbose
            )

            if success:
                progress.console.print(f"  [green]✓[/green] {display_name}")
                successful += 1
            else:
                progress.console.print(f"  [red]✗[/red] {display_name}")
                error_first_line = error_msg.split('\n')[0][:80]
                progress.console.print(f"    [red]{error_first_line}[/red]")
                if args.verbose and "\n" in error_msg:
                    for line in error_msg.split("\n")[1:15]:
                        progress.console.print(f"    [dim]{line}[/dim]")
                failed_items.append({
                    "name": name, 
                    "artist": artist, 
                    "url": url, 
                    "reason": error_first_line
                })

            progress.advance(overall)

    # Summary
    console.print()
    summary = Table(title="Download Summary", show_header=False, border_style="blue")
    summary.add_column("Label", style="bold")
    summary.add_column("Value")
    summary.add_row("✓ Successful", f"[green]{successful}[/green]")
    summary.add_row("✗ Failed", f"[red]{len(failed_items)}[/red]")
    summary.add_row("Total", str(len(sources)))
    summary.add_row("Output folder", f"[cyan]{output_dir.name}/[/cyan]")
    console.print(summary)

    if failed_items:
        console.print()
        fail_table = Table(title="Failed Downloads", border_style="red")
        fail_table.add_column("Song", style="cyan", no_wrap=True, max_width=35)
        fail_table.add_column("Reason", style="yellow", max_width=40)
        fail_table.add_column("URL", style="dim", max_width=30)

        for item in failed_items:
            song = f"{item['artist']} - {item['name']}"
            if len(song) > 35:
                song = song[:32] + "..."
            fail_table.add_row(song, item["reason"], item["url"])

        console.print(fail_table)


if __name__ == "__main__":
    main()