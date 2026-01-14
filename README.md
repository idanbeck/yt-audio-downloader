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
```

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

## License

MIT