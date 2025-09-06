#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
import praw
import prawcore   # <-- add

from analysis import *


TIME_FILTERS = {"all", "day", "hour", "month", "week", "year"}
LISTING_CHOICES = {"hot", "new", "top", "rising"}


def verify_subreddit_exists(reddit, name: str):
    """
    Return a valid subreddit display name if found; otherwise (None, suggestion)
    """
    # try exact match first
    try:
        hits = reddit.subreddits.search_by_name(name, exact=True)
        if hits:
            # use the canonical casing
            return hits[0].display_name, None
    except Exception:
        pass

    # try fuzzy search for a suggestion
    try:
        for sr in reddit.subreddits.search(name, limit=1):
            return None, sr.display_name
    except Exception:
        pass
    return None, None


def safe_iter_submissions(subr, listing, limit, time_filter):
    """Wrap iterator to surface redirect/forbidden nicely."""
    try:
        return iter_submissions(subr, listing, limit, time_filter)
    except prawcore.Forbidden as e:
        raise RuntimeError("forbidden/private") from e
    except prawcore.NotFound as e:
        raise RuntimeError("not_found") from e
    except prawcore.Redirect as e:
        raise RuntimeError("redirect_to_search") from e



def read_subreddits_from_file(path: str):
    """
    Accepts whitespace- or comma-separated names, supports comments (#),
    strips optional 'r/' prefixes, removes duplicates while preserving order.
    """
    from pathlib import Path
    text = Path(path).read_text(encoding="utf-8")

    # allow commas or whitespace
    raw = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()   # drop inline comments
        if not line:
            continue
        raw.extend([p.strip() for p in line.replace(",", " ").split() if p.strip()])

    cleaned = []
    seen = set()
    for name in raw:
        name = name.lstrip("r/").strip()       # allow "r/xyz"
        key = name.lower()                      # case-insensitive de-dupe
        if key not in seen:
            seen.add(key)
            cleaned.append(name)
    if not cleaned:
        raise ValueError(f"No subreddit names found in {path}")
    return cleaned


def fetch_top_comments(submission, top_n: int):
    if top_n <= 0:
        return []
    submission.comments.replace_more(limit=0)
    return [c.body for c in submission.comments[:top_n] if getattr(c, "body", "").strip()]


def parse_args():
    p = argparse.ArgumentParser(
        description="Scrape Reddit posts from a list of subreddits via Reddit API (PRAW)."
    )

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--subreddits",
        "-s",
        nargs="+",
        help="List of subreddit names (without r/), e.g. askreddit python dataengineering",
    )
    group.add_argument(
        "--subreddits-file",
        "-sf",
        help="Path to a file containing subreddit names (comma/space/newline separated).",
    )

    p.add_argument(
        "--listing", "-l",
        choices=sorted(LISTING_CHOICES),
        default="hot",
        help="Which feed to scrape (hot/new/top/rising). Default: hot",
    )
    p.add_argument(
        "--time-filter",
        choices=sorted(TIME_FILTERS),
        default="day",
        help="Time filter for 'top' listing (ignored for others). Default: day",
    )
    p.add_argument("--limit", "-n", type=int, default=100, help="Max posts per subreddit. Default: 100")
    p.add_argument("--outdir", "-o", default="out", help="Output folder. Default: ./out")
    p.add_argument("--prefix", default=None, help="Optional filename prefix, e.g. 'run1'. If omitted, auto-generated.")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between subreddits. Default: 0")
    p.add_argument("--analyze", action="store_true",
               help="Run OpenAI analysis on scraped posts and produce structured outputs.")
    p.add_argument("--include-comments", type=int, default=0,
               help="Top N comments per post to include in the analysis prompt (0 = none).")
    p.add_argument("--batch-size", type=int, default=20,
               help="How many posts to analyze per OpenAI call.")
    p.add_argument("--model", default=None,
               help="Override model (else uses OPENAI_MODEL from .env).")
    p.add_argument("--table", choices=["csv", "md", "both"], default="csv",
               help="Tabular export for OpenAI analysis: csv, md, or both. Default: csv")
    p.add_argument("--md-top", type=int, default=50,
               help="Max rows for the Markdown preview table. Default: 50")


    return p.parse_args()


def load_reddit_client():
    load_dotenv()
    required = [
        "REDDIT_CLIENT_ID",
        "REDDIT_CLIENT_SECRET",
        "REDDIT_USERNAME",
        "REDDIT_PASSWORD",
        "REDDIT_USER_AGENT",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"Missing env vars in .env: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
        ratelimit_seconds=5,
    )

def iter_submissions(subreddit, listing, limit, time_filter):
    if listing == "hot":
        return subreddit.hot(limit=limit)
    elif listing == "new":
        return subreddit.new(limit=limit)
    elif listing == "rising":
        return subreddit.rising(limit=limit)
    elif listing == "top":
        # time_filter: 'all', 'day', 'hour', 'month', 'week', 'year'
        return subreddit.top(limit=limit, time_filter=time_filter)
    else:
        raise ValueError("Unknown listing type")

def normalize(sub):
    """
    Turn a PRAW Submission into a flat dict safe for CSV/JSONL.
    """
    # Convert epoch -> ISO 8601
    created_iso = datetime.utcfromtimestamp(sub.created_utc).isoformat() + "Z"

    return {
        "id": sub.id,
        "subreddit": str(sub.subreddit),
        "title": (sub.title or "").replace("\n", " ").strip(),
        "author": getattr(sub.author, "name", None),
        "url": sub.url,
        "permalink": f"https://reddit.com{sub.permalink}",
        "is_self": bool(sub.is_self),
        "selftext": (sub.selftext or "").strip(),
        "score": int(sub.score or 0),
        "upvote_ratio": float(sub.upvote_ratio) if sub.upvote_ratio is not None else None,
        "num_comments": int(sub.num_comments or 0),
        "over_18": bool(getattr(sub, "over_18", False)),
        "spoiler": bool(getattr(sub, "spoiler", False)),
        "stickied": bool(getattr(sub, "stickied", False)),
        "locked": bool(getattr(sub, "locked", False)),
        "domain": getattr(sub, "domain", None),
        "link_flair_text": getattr(sub, "link_flair_text", None),
        "author_flair_text": getattr(sub, "author_flair_text", None),
        "created_utc": int(sub.created_utc),
        "created_iso": created_iso,
    }

def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")



def main():
    args = parse_args()
    reddit = load_reddit_client()

    # resolve subreddits source
    if args.subreddits:
        subreddits = args.subreddits
    else:
        subreddits = read_subreddits_from_file(args.subreddits_file)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    prefix = args.prefix or (f"{args.listing}_{args.time_filter}_{ts}" if args.listing == "top" else f"{args.listing}_{ts}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    # resolve list from args or file above this point...
    for raw_name in subreddits:
        name = raw_name.strip().lstrip("r/")

        # verify/normalize the subreddit before scraping
        valid, suggestion = verify_subreddit_exists(reddit, name)
        if not valid:
            msg = f"[skip] r/{name} not found."
            if suggestion:
                msg += f" Did you mean r/{suggestion}?"
            print(msg, file=sys.stderr)
            continue

        rows = []
        try:
            submissions = safe_iter_submissions(
                reddit.subreddit(valid), args.listing, args.limit, args.time_filter
            )
            for sub in tqdm(submissions, desc=f"r/{valid}", unit="post"):
                try:
                    row = normalize(sub)

                    # add top comments if requested
                    if args.include_comments and args.include_comments > 0:
                        try:
                            comments = fetch_top_comments(sub, args.include_comments)
                        except Exception as ce:
                            print(f"[warn] failed to fetch comments for {sub.id}: {ce}", file=sys.stderr)
                            comments = []
                    else:
                        comments = []

                    row["top_comments"] = comments          # list (good for JSONL)
                    row["top_comments_str"] = " || ".join(  # safe for CSV
                        [c.replace("\n", " ").strip() for c in comments]
                    ) if comments else ""

                    rows.append(row)

                except Exception as e:
                    print(f"[warn] skipping {getattr(sub, 'id', 'unknown')}: {e}", file=sys.stderr)

        except RuntimeError as e:
            print(f"[skip] r/{valid}: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"[error] r/{valid}: {e}", file=sys.stderr)
            continue

        write_csv(outdir / f"{prefix}_r_{valid}.csv", rows)
        write_jsonl(outdir / f"{prefix}_r_{valid}.jsonl", rows)
        all_rows.extend(rows)
        if args.sleep > 0:
            time.sleep(args.sleep)
        


    write_csv(outdir / f"{prefix}_combined.csv", all_rows)
    write_jsonl(outdir / f"{prefix}_combined.jsonl", all_rows)
    if args.analyze:
            run_openai_analysis(args, all_rows, outdir, prefix)

    print(f"\nDone. Saved {len(all_rows)} posts total to '{outdir}'.")
    if all_rows:
        print(f"Examples:\n- {outdir / f'{prefix}_combined.csv'}\n- {outdir / f'{prefix}_combined.jsonl'}")

if __name__ == "__main__":
    main()
