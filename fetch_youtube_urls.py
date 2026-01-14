#!/usr/bin/env python3
"""
Fetch YouTube URLs for wedding mix songs.
Uses yt-dlp to search YouTube and get actual video URLs.

Usage:
    pip install yt-dlp pyyaml
    python fetch_youtube_urls.py
"""

import subprocess
import json
import yaml
import re
from pathlib import Path
from typing import Optional

def search_youtube(query: str, max_results: int = 1) -> Optional[str]:
    """Search YouTube and return the first video URL."""
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--flat-playlist",
                "--no-warnings",
                "-j",
                f"ytsearch{max_results}:{query}"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip())
            if data.get("id"):
                return f"https://www.youtube.com/watch?v={data['id']}"
    except Exception as e:
        print(f"  Error searching for '{query}': {e}")
    return None

def main():
    yaml_path = Path(__file__).parent / "wedding-mix-songs.yaml"

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    sources = data.get("sources", [])
    updated = 0

    for i, song in enumerate(sources):
        name = song.get("name", "")
        user = song.get("user", "")
        name_en = song.get("name_en", "")

        # Build search query - Hebrew name + artist is usually most accurate
        search_query = f"{name} {user}"

        print(f"[{i+1}/{len(sources)}] Searching: {name} - {user}")

        url = search_youtube(search_query)

        if url:
            song["url"] = url
            updated += 1
            print(f"  Found: {url}")
        else:
            # Try English name as fallback
            if name_en:
                fallback_query = f"{name_en} {song.get('user_en', user)}"
                print(f"  Trying fallback: {fallback_query}")
                url = search_youtube(fallback_query)
                if url:
                    song["url"] = url
                    updated += 1
                    print(f"  Found: {url}")
                else:
                    print(f"  NOT FOUND")
            else:
                print(f"  NOT FOUND")

    # Write updated YAML
    output_path = yaml_path.parent / "wedding-mix-songs-updated.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\nDone! Updated {updated}/{len(sources)} songs.")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()
