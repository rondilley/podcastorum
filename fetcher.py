"""Podcast Fetcher — discover RSS feeds, download new episodes to spool directory."""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import feedparser
import requests

import config

FEEDS_FILE = config.BASE_DIR / "feeds.json"
STATE_FILE = config.BASE_DIR / ".fetch_state.json"
USER_AGENT = "Podcastorum/1.0 (podcast-fetcher)"
REQUEST_TIMEOUT = 30


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_feeds() -> dict:
    """Load the feeds configuration file."""
    if not FEEDS_FILE.exists():
        return {"podcasts": []}
    return json.loads(FEEDS_FILE.read_text(encoding="utf-8"))


def save_feeds(data: dict) -> None:
    """Save the feeds configuration file."""
    FEEDS_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_state() -> dict:
    """Load the fetch state tracker."""
    if not STATE_FILE.exists():
        return {}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def save_state(data: dict) -> None:
    """Save the fetch state tracker."""
    STATE_FILE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def mark_downloaded(state: dict, podcast_name: str, guid: str, filepath: str) -> None:
    """Record a downloaded episode in the state tracker."""
    if podcast_name not in state:
        state[podcast_name] = {"downloaded_guids": [], "downloaded_files": {}}
    entry = state[podcast_name]
    if guid not in entry["downloaded_guids"]:
        entry["downloaded_guids"].append(guid)
    entry["downloaded_files"][guid] = filepath
    entry["last_checked"] = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# RSS feed discovery
# ---------------------------------------------------------------------------

class _LinkTagParser(HTMLParser):
    """Extract RSS/Atom feed URLs from HTML <link> tags."""

    def __init__(self):
        super().__init__()
        self.feed_urls = []

    def handle_starttag(self, tag, attrs):
        if tag != "link":
            return
        attr_dict = dict(attrs)
        rel = attr_dict.get("rel", "").lower()
        link_type = attr_dict.get("type", "").lower()
        href = attr_dict.get("href", "")
        if rel == "alternate" and link_type in (
            "application/rss+xml",
            "application/atom+xml",
            "application/xml",
            "text/xml",
        ) and href:
            self.feed_urls.append(href)


def _is_valid_feed(url: str) -> bool:
    """Check if a URL returns a parseable RSS/Atom feed."""
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT,
                            headers={"User-Agent": USER_AGENT}, stream=True)
        if resp.status_code != 200:
            return False
        # Read just enough to check if it's XML-ish
        chunk = resp.raw.read(2048).decode("utf-8", errors="replace")
        resp.close()
        return "<rss" in chunk.lower() or "<feed" in chunk.lower() or "<channel" in chunk.lower()
    except Exception:
        return False


def discover_feed_url(website_url: str) -> str | None:
    """Try to find an RSS feed URL from a website URL."""
    website_url = normalize_url(website_url)

    # Step 1: Parse HTML for <link> tags
    try:
        resp = requests.get(website_url, timeout=REQUEST_TIMEOUT,
                            headers={"User-Agent": USER_AGENT})
        if resp.status_code == 200:
            parser = _LinkTagParser()
            parser.feed(resp.text)
            for feed_url in parser.feed_urls:
                absolute_url = urljoin(website_url, feed_url)
                if _is_valid_feed(absolute_url):
                    return absolute_url
    except requests.RequestException as e:
        print(f"  Warning: Could not fetch {website_url}: {e}", file=sys.stderr)

    # Step 2: Probe common paths
    common_paths = ["/feed", "/rss", "/feed.xml", "/rss.xml", "/atom.xml",
                    "/podcast.xml", "/feed/podcast", "/index.xml"]
    parsed = urlparse(website_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    for path in common_paths:
        probe_url = base + path
        if _is_valid_feed(probe_url):
            return probe_url

    return None


# ---------------------------------------------------------------------------
# Feed parsing
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """Ensure a URL has a scheme."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def parse_feed(feed_url: str) -> dict:
    """Parse an RSS feed and return podcast info with episodes."""
    feed_url = normalize_url(feed_url)

    # Fetch the content first so we can give a clear error if it's not a feed
    try:
        resp = requests.get(feed_url, timeout=REQUEST_TIMEOUT,
                            headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Could not fetch feed URL: {e}")

    content = resp.text
    # Quick sanity check — if it looks like HTML, it's probably not a feed
    if "<html" in content[:500].lower() and "<rss" not in content[:500].lower():
        raise ValueError(
            f"URL returned HTML, not an RSS feed. "
            f"Try using 'add <website-url>' instead of '--rss' to auto-discover the feed."
        )

    feed = feedparser.parse(content)

    if feed.bozo and not feed.entries:
        raise ValueError(f"Failed to parse feed: {feed.bozo_exception}")

    podcast_title = feed.feed.get("title", "Unknown Podcast")
    episodes = []

    for entry in feed.entries:
        # Find the audio enclosure
        audio_url = None
        audio_type = None
        audio_size = 0

        for enc in entry.get("enclosures", []):
            enc_type = enc.get("type", "")
            if enc_type.startswith("audio/") or enc.get("url", "").endswith(
                (".mp3", ".m4a", ".ogg", ".wav", ".aac")
            ):
                audio_url = enc.get("url")
                audio_type = enc_type
                audio_size = int(enc.get("length", 0) or 0)
                break

        # Some feeds use media:content instead of enclosures
        if not audio_url:
            for media in entry.get("media_content", []):
                if media.get("medium") == "audio" or media.get("type", "").startswith("audio/"):
                    audio_url = media.get("url")
                    audio_type = media.get("type", "")
                    break

        if not audio_url:
            continue  # Skip entries without audio

        # Parse publication date
        published = None
        if entry.get("published_parsed"):
            published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
        elif entry.get("updated_parsed"):
            published = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc).isoformat()

        guid = entry.get("id", entry.get("link", audio_url))

        episodes.append({
            "guid": guid,
            "title": entry.get("title", "Untitled"),
            "url": audio_url,
            "published": published,
            "size": audio_size,
            "type": audio_type,
        })

    return {"title": podcast_title, "episodes": episodes}


def filter_episodes(episodes: list, latest_n: int = None, since: str = None,
                    downloaded_guids: set = None) -> list:
    """Filter episodes by recency, date, and already-downloaded status."""
    filtered = episodes

    # Remove already-downloaded
    if downloaded_guids:
        filtered = [e for e in filtered if e["guid"] not in downloaded_guids]

    # Filter by date
    if since:
        since_dt = datetime.fromisoformat(since)
        if since_dt.tzinfo is None:
            since_dt = since_dt.replace(tzinfo=timezone.utc)
        filtered = [
            e for e in filtered
            if e.get("published") and datetime.fromisoformat(e["published"]) >= since_dt
        ]

    # Limit to latest N
    if latest_n is not None:
        filtered = filtered[:latest_n]

    return filtered


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    """Remove filesystem-unsafe characters from a filename."""
    # Replace problematic characters with safe alternatives
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    # Truncate to reasonable length
    if len(name) > 200:
        name = name[:200].strip()
    return name


def _guess_extension(url: str, content_type: str = "") -> str:
    """Guess file extension from URL or content type."""
    # Try URL path first
    path = urlparse(url).path
    for ext in (".mp3", ".m4a", ".ogg", ".wav", ".aac", ".opus"):
        if ext in path.lower():
            return ext
    # Fall back to content type
    type_map = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/aac": ".aac",
    }
    return type_map.get(content_type, ".mp3")


def download_episode(episode: dict, dest_dir: Path, podcast_name: str) -> Path:
    """Download a podcast episode to the destination directory.

    Uses a .part temp file for atomic writes — incomplete downloads
    won't appear as valid files.
    """
    ext = _guess_extension(episode["url"], episode.get("type", ""))
    filename = sanitize_filename(f"{podcast_name} - {episode['title']}") + ext
    dest_path = dest_dir / filename
    part_path = dest_dir / (filename + ".part")

    if dest_path.exists():
        print(f"  Already exists: {filename}")
        return dest_path

    print(f"  Downloading: {episode['title']}")
    resp = requests.get(episode["url"], stream=True, timeout=REQUEST_TIMEOUT,
                        headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(part_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r    {downloaded / 1_048_576:.1f} / {total / 1_048_576:.1f} MB ({pct:.0f}%)",
                      end="", flush=True)
            else:
                print(f"\r    {downloaded / 1_048_576:.1f} MB", end="", flush=True)

    print()  # newline after progress

    # Atomic rename
    part_path.rename(dest_path)
    print(f"    Saved: {filename}")
    return dest_path


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------

def cmd_add(args) -> None:
    """Add a new podcast source."""
    feeds = load_feeds()

    if args.rss:
        feed_url = args.rss
        print(f"Using provided RSS URL: {feed_url}")
    else:
        url = args.url
        print(f"Discovering RSS feed for {url}...")
        feed_url = discover_feed_url(url)
        if not feed_url:
            print(f"Error: Could not find RSS feed for {url}", file=sys.stderr)
            print("Try providing the feed URL directly with --rss", file=sys.stderr)
            sys.exit(1)
        print(f"  Found feed: {feed_url}")

    # Check for duplicate feed URL
    for p in feeds["podcasts"]:
        if p["feed_url"] == feed_url:
            print(f"Already tracking: {p['name']} ({feed_url})")
            return

    # Parse feed to get the title
    print("Fetching feed metadata...")
    info = parse_feed(feed_url)
    name = args.name or info["title"]

    feeds["podcasts"].append({
        "name": name,
        "feed_url": feed_url,
        "website": args.url if not args.rss else "",
        "added": datetime.now(timezone.utc).isoformat(),
    })
    save_feeds(feeds)
    print(f"Added: {name} ({len(info['episodes'])} episodes available)")


def cmd_list(args) -> None:
    """List all tracked podcasts."""
    feeds = load_feeds()
    state = load_state()

    if not feeds["podcasts"]:
        print("No podcasts tracked. Add one with: python fetcher.py add <url>")
        return

    for p in feeds["podcasts"]:
        name = p["name"]
        downloaded = len(state.get(name, {}).get("downloaded_guids", []))
        last_checked = state.get(name, {}).get("last_checked", "never")
        if last_checked != "never":
            last_checked = last_checked[:19].replace("T", " ")
        print(f"  {name}")
        print(f"    Feed: {p['feed_url']}")
        print(f"    Downloaded: {downloaded} episodes")
        print(f"    Last checked: {last_checked}")
        print()


def cmd_fetch(args) -> None:
    """Fetch new episodes from tracked podcasts."""
    feeds = load_feeds()
    state = load_state()

    if not feeds["podcasts"]:
        print("No podcasts tracked. Add one with: python fetcher.py add <url>")
        return

    config.PODCASTS_DIR.mkdir(exist_ok=True)
    new_files = []
    errors = 0

    for p in feeds["podcasts"]:
        name = p["name"]

        # Filter to specific podcast if requested
        if args.podcast and args.podcast.lower() != name.lower():
            continue

        print(f"\n{'='*60}")
        print(f"Checking: {name}")
        print(f"{'='*60}")

        try:
            info = parse_feed(p["feed_url"])
        except Exception as e:
            print(f"  Error parsing feed: {e}", file=sys.stderr)
            errors += 1
            continue

        downloaded_guids = set(state.get(name, {}).get("downloaded_guids", []))
        episodes = filter_episodes(
            info["episodes"],
            latest_n=args.latest,
            since=args.since,
            downloaded_guids=downloaded_guids,
        )

        if not episodes:
            print("  No new episodes")
            # Still update last_checked
            if name not in state:
                state[name] = {"downloaded_guids": [], "downloaded_files": {}}
            state[name]["last_checked"] = datetime.now(timezone.utc).isoformat()
            save_state(state)
            continue

        print(f"  {len(episodes)} new episode(s) to download")

        for episode in episodes:
            try:
                filepath = download_episode(episode, config.PODCASTS_DIR, name)
                mark_downloaded(state, name, episode["guid"], str(filepath))
                save_state(state)  # Save after each download in case of interruption
                new_files.append(filepath)
            except Exception as e:
                print(f"  Error downloading '{episode['title']}': {e}", file=sys.stderr)
                errors += 1

    print(f"\nDone: {len(new_files)} episode(s) downloaded", end="")
    if errors:
        print(f", {errors} error(s)")
    else:
        print()

    # Optionally run the analysis pipeline
    if args.analyze and new_files:
        import summarizer
        for audio_path in new_files:
            print(f"\n{'='*60}")
            print(f"Analyzing: {audio_path.name}")
            print(f"{'='*60}")
            try:
                summarizer.process_podcast(audio_path)
            except SystemExit:
                pass  # process_podcast calls sys.exit on error
            except Exception as e:
                print(f"Error analyzing {audio_path.name}: {e}", file=sys.stderr)

    sys.exit(1 if errors else 0)


def cmd_remove(args) -> None:
    """Remove a tracked podcast."""
    feeds = load_feeds()
    name_lower = args.name.lower()

    original_count = len(feeds["podcasts"])
    feeds["podcasts"] = [p for p in feeds["podcasts"] if p["name"].lower() != name_lower]

    if len(feeds["podcasts"]) == original_count:
        print(f"Not found: {args.name}")
        print("Use 'python fetcher.py list' to see tracked podcasts")
        sys.exit(1)

    save_feeds(feeds)
    print(f"Removed: {args.name}")

    # Clean up state
    state = load_state()
    removed = [k for k in state if k.lower() == name_lower]
    for k in removed:
        del state[k]
    if removed:
        save_state(state)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Podcast Fetcher — discover and download podcast episodes"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- add ---
    add_parser = subparsers.add_parser("add", help="Add a podcast source")
    add_parser.add_argument("url", nargs="?", default="",
                            help="Website URL (e.g., darknetdiaries.com)")
    add_parser.add_argument("--rss", help="RSS feed URL (skip auto-discovery)")
    add_parser.add_argument("--name", help="Override podcast name")

    # --- list ---
    subparsers.add_parser("list", help="List tracked podcasts")

    # --- fetch ---
    fetch_parser = subparsers.add_parser("fetch", help="Download new episodes")
    fetch_parser.add_argument("--podcast", help="Fetch from a specific podcast only")
    fetch_parser.add_argument("--latest", type=int, help="Only download the N most recent episodes")
    fetch_parser.add_argument("--since", help="Only download episodes after this date (YYYY-MM-DD)")
    fetch_parser.add_argument("--analyze", action="store_true",
                              help="Run the podcastorum analysis pipeline on downloaded episodes")

    # --- remove ---
    remove_parser = subparsers.add_parser("remove", help="Remove a tracked podcast")
    remove_parser.add_argument("name", help="Podcast name to remove")

    args = parser.parse_args()

    commands = {
        "add": cmd_add,
        "list": cmd_list,
        "fetch": cmd_fetch,
        "remove": cmd_remove,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
