# YouTube Audio Downloader

Download audio from YouTube URLs defined in a YAML file.

## Installation

Requires Python 3.10+ and [Poetry](https://python-poetry.org/docs/#installation).

### 1. Install Deno (required for YouTube as of Nov 2025)

YouTube now requires a JavaScript runtime to bypass their anti-bot challenges.

```bash
# macOS/Linux
curl -fsSL https://deno.land/install.sh | sh

# macOS (Homebrew)
brew install deno

# Windows
choco install deno
# or
winget install --id=DenoLand.Deno
```

### 2. Install ffmpeg (required for MP3 conversion)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
choco install ffmpeg
```

### 3. Install the project

```bash
poetry install
```

### 4. Update yt-dlp (if you have an old version)

```bash
pip install -U "yt-dlp[default]"
```

## Usage

1. Place your YAML file in the current directory
2. Run the downloader:

```bash
poetry run yt-download
```

### Options

```bash
# Use browser cookies (helps with bot detection)
poetry run yt-download -c chrome
poetry run yt-download -c firefox
poetry run yt-download -c brave

# Verbose output for debugging
poetry run yt-download -v

# Specify YAML file
poetry run yt-download -f /path/to/songs.yaml

# Force re-download everything (ignore archive)
poetry run yt-download --force
```

## Smart Skipping

The downloader automatically tracks which videos have been downloaded in a `.downloaded_archive.txt` file. When you run it again, it will skip any songs that have already been downloaded.

To force re-downloading everything:
```bash
poetry run yt-download --force
```

## Audio Analysis & Sorting

Analyze downloaded songs and rename them with sortable prefixes based on BPM and intensity.

### Sort by BPM
```bash
poetry run yt-download --sort_by bpm -c chrome
```
Files renamed to: `120-Artist - Song.mp3`

### Sort by Intensity
```bash
poetry run yt-download --sort_by intensity -c chrome
```
Files renamed to: `3high-Artist - Song.mp3`

Intensity levels:
- `1low` - Calm, quiet songs
- `2med` - Moderate energy
- `3high` - Energetic, loud
- `4vhigh` - Very high energy (bangers)

### Sort by Multiple Keys
```bash
poetry run yt-download --sort_by bpm intensity -c chrome
```
Files renamed to: `128-3high-Artist - Song.mp3`

### Analyze Existing Files Only
```bash
poetry run yt-download --analyze_only --sort_by bpm intensity
```
Skip downloading, just analyze and rename existing MP3s.

## Audio Mastering

Apply professional mastering to your downloaded tracks, matching loudness and punch of commercial releases.

### Master with Commercial Profile (default)
```bash
poetry run yt-download --master -c chrome
```
Loud, punchy, wide stereo - perfect for DJ sets or radio. Target: **-9 LUFS**

### Master for Streaming
```bash
poetry run yt-download --master streaming -c chrome
```
Optimized for Spotify, Apple Music, YouTube. Target: **-14 LUFS**

### Master with Vinyl Profile
```bash
poetry run yt-download --master vinyl -c chrome
```
Warm, dynamic, vintage feel with subtle saturation. Target: **-12 LUFS**

### Master Existing Files Only
```bash
poetry run yt-download --master_only
poetry run yt-download --master_only --master vinyl
```

### Re-master with Different Profile
```bash
poetry run yt-download --master vinyl --remaster
```
The `--remaster` flag deletes existing `_mastered.mp3` files before re-mastering.

### How Files Are Handled

- **Original files are NEVER modified** - they stay pristine
- Mastered files are saved separately as `songname_mastered.mp3`
- You can always re-master from the untouched originals

### Mastering Profiles

| Profile | Target LUFS | Character |
|---------|-------------|-----------|
| `commercial` | -11 | Polished, punchy, radio-ready. Preserves dynamics. |
| `streaming` | -14 | Spotify/Apple Music optimized. Natural dynamics. |
| `vinyl` | -12 | Warm, analog feel with subtle saturation. |
| `dynamic` | -16 | Maximum dynamic range, minimal processing. |

### Smart Mastering

The mastering engine automatically:

1. **Detects already-mastered tracks** - Modern, well-produced songs are skipped or given only light processing
2. **Identifies recording era** - 70s analog, 80s digital, vintage, or modern
3. **Applies era-appropriate processing**:
   - **Vintage/70s**: Noise reduction, gentle air boost
   - **80s digital**: De-harsh treatment, reduced high-frequency harshness
   - **Modern**: Light touch or skipped entirely

### Handling Older Recordings

For songs from the 70s-90s, the mastering automatically:
- Removes tape hiss and background noise
- Reduces harsh "digital edge" from early digital recordings  
- De-esses sibilant vocals
- Gates noise during quiet sections
- Avoids over-widening narrow stereo mixes

### Mastering Output

```
Mastering Summary:
  ✓ Full mastering: 15        # Older tracks that needed work
  ~ Light touch: 8            # Good tracks with minor adjustments
  ⊘ Already good (skipped): 12 # Modern tracks left alone
```

### What the Mastering Does

1. **High-pass filter** - Removes sub-bass rumble (<30Hz)
2. **Bass boost** - Adds weight in the low end (100Hz)
3. **Presence boost** - Clarity and punch (3kHz)
4. **Air/sparkle** - High-end shimmer (10kHz+)
5. **Glue compression** - Cohesive, punchy sound
6. **Stereo widening** - M/S processing for width
7. **Loudness normalization** - Hits target LUFS
8. **Limiting** - Brick-wall limiter, no clipping

Mastered files are saved as `songname_mastered.mp3` at 320kbps.

## DJ Mix Creation (Experimental)

Automatically create a DJ mix from your downloaded songs with beat matching, smart transitions, and intelligent section selection.

### Basic Mix
```bash
poetry run yt-download --create_mix -c chrome
```
Creates a 60-minute mix with 1.5 minutes per song, DJ-style transitions.

### Custom Mix Length
```bash
poetry run yt-download --create_mix --mix_length 30 -c chrome    # 30 min mix
poetry run yt-download --create_mix --mix_length 90 -c chrome    # 90 min mix
```

### Custom Song Length
```bash
poetry run yt-download --create_mix --avg_song_length 2.0    # 2 min per song
poetry run yt-download --create_mix --avg_song_length 1.0    # 1 min per song (more songs)
```

### Transition Styles
```bash
poetry run yt-download --create_mix --transition_style dj      # Quick beat-matched cuts (default)
poetry run yt-download --create_mix --transition_style smooth  # Long 6-second crossfades
poetry run yt-download --create_mix --transition_style radio   # Medium 3-second fades
poetry run yt-download --create_mix --transition_style drop    # Hard cuts on drops
```

### Ordering by BPM/Intensity
```bash
poetry run yt-download --create_mix --sort_by bpm              # Order songs by BPM
poetry run yt-download --create_mix --sort_by intensity        # Order by energy level
poetry run yt-download --create_mix --sort_by bpm intensity    # BPM first, then intensity
```

### Disable Beat Matching
```bash
poetry run yt-download --create_mix --no_beat_match
```

### Mix Existing Files Only
```bash
poetry run yt-download --mix_only --sort_by bpm --transition_style smooth
```

### How Mix Creation Works

1. **Audio Analysis** - Each song is analyzed for:
   - Actual BPM (detected, not from metadata)
   - Beat grid and downbeats
   - Segment boundaries (verse, chorus, etc.)
   - Energy curve over time
   - Good transition points (low energy moments, phrase boundaries)

2. **Smart Section Selection** - Picks the best parts of each song:
   - Uses `best_sections` from YAML if available
   - Otherwise finds highest-energy segments
   - Snaps to beat boundaries for clean cuts

3. **Beat Matching** - Time-stretches songs to a common BPM:
   - Uses median BPM of all songs as target
   - Preserves pitch while changing tempo

4. **Transitions** - Applies crossfades between songs:
   - DJ style: Quick cuts on beat boundaries
   - Smooth: Long gradual fades
   - Drop: Hard cuts into choruses

## YAML Format

```yaml
sources:
  - name: שני משוגעים
    name_en: Shnei Meshugaim (Two Crazy Ones)
    user: עומר אדם
    user_en: Omer Adam
    url: https://www.youtube.com/watch?v=E0k1Ej0WRaM
    # Additional fields (tags, timestamps, etc.) are ignored

  - name: Another Song
    name_en: Another Song English
    user_en: Artist Name
    url: https://www.youtube.com/watch?v=XXXXXXXXXXX
```

### Required fields
- `url` - YouTube video URL

### Optional fields (used for filename)
- `name_en` or `name` - Song title (English preferred)
- `user_en` or `user` - Artist name (English preferred)

Output filename format: `Artist - Song Title.mp3`

Output folder: Named after the YAML file (e.g., `wedding_songs.yaml` → `wedding_songs/`)

## Troubleshooting

### "Unknown error" / Downloads fail silently

1. **Update yt-dlp**: `pip install -U "yt-dlp[default]"`
2. **Install Deno**: Required since November 2025
3. **Use cookies**: `yt-download -c chrome`

### Bot detection / 403 errors

Use browser cookies to authenticate:
```bash
poetry run yt-download -c chrome
```
Make sure Chrome is closed when running this.

### "Video unavailable"

The video was deleted, made private, or the URL is incorrect.

## Configuration

Edit `yt_audio_downloader/main.py` to change:

| Setting | Default | Options |
|---------|---------|---------|
| Audio format | `mp3` | `m4a`, `opus`, `flac`, `wav` |
| Quality | `192K` | `128K`, `192K`, `256K`, `320K` |

## Why Deno?

As of November 2025, YouTube requires solving JavaScript challenges to download videos. yt-dlp now needs an external JS runtime to handle this. Deno is recommended because it's sandboxed (no filesystem/network access by default) and comes as a single portable executable.

## Dependencies

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube downloader
- [Deno](https://deno.com/) - JavaScript runtime
- [PyYAML](https://pyyaml.org/) - YAML parser
- [ffmpeg](https://ffmpeg.org/) - Audio conversion
- [Rich](https://rich.readthedocs.io/) - Terminal UI
- [librosa](https://librosa.org/) - Audio analysis (for --sort_by and --create_mix)
- [NumPy](https://numpy.org/) - Numerical computing
- [pedalboard](https://github.com/spotify/pedalboard) - Audio processing (for --master)
- [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) - LUFS measurement
- [soundfile](https://github.com/bastibe/python-soundfile) - Audio file I/O
- [pydub](https://github.com/jiaaro/pydub) - Audio manipulation (for --create_mix)
- [pyrubberband](https://github.com/bmcfee/pyrubberband) - Time-stretching (for beat matching)

### System Dependencies

For `pyrubberband` (beat matching), you also need the `rubberband` library:

```bash
# macOS
brew install rubberband

# Ubuntu/Debian
sudo apt install rubberband-cli

# Windows
# Download from https://breakfastquay.com/rubberband/
```

## License

MIT