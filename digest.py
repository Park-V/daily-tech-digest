#!/usr/bin/env python3
"""
FeedlyVP — Personal Tech News Digest
Fetches RSS feeds, scores articles with Claude, emails the best ones via SendGrid.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import yaml
import anthropic
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
FEEDS_YAML = BASE_DIR / "feeds.yaml"
SEEN_URLS_FILE = BASE_DIR / "seen_urls.json"
DIGEST_LOG_FILE = BASE_DIR / "digest_log.json"

TOP_N = 15
SCORE_THRESHOLD = 7
MODEL = "claude-sonnet-4-20250514"

SCORING_SYSTEM = """You are a relevance filter for a Solution Consultant at Oracle NetSuite \
who pitches and demos ERP and AI-infused enterprise software to existing customers.

Score HIGH (8-10): enterprise AI, agentic workflows, finance automation,
AP/AR/FP&A, ERP trends, competitor moves (Salesforce, SAP, Workday, \
Microsoft Dynamics), AI policy, tech's business impact.

Score LOW (1-4): consumer gadgets, smartphone reviews, gaming, \
celebrity tech, lifestyle.

Return ONLY valid JSON: {"score": <1-10>, "summary": "<2 sentences>", \
"why": "<one short phrase>"}"""


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(FEEDS_YAML) as f:
        return yaml.safe_load(f)


def load_seen_urls() -> set:
    if SEEN_URLS_FILE.exists():
        with open(SEEN_URLS_FILE) as f:
            return set(json.load(f))
    return set()


def save_seen_urls(seen: set) -> None:
    with open(SEEN_URLS_FILE, "w") as f:
        json.dump(sorted(seen), f, indent=2)


def load_digest_log() -> list:
    if DIGEST_LOG_FILE.exists():
        with open(DIGEST_LOG_FILE) as f:
            return json.load(f)
    return []


def save_digest_log(log: list) -> None:
    with open(DIGEST_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def get_stale_warning(log: list, now: datetime) -> str:
    """Return a warning string if the last successful daily run was > 2 days ago."""
    daily_entries = [e for e in log if e.get("type") == "daily" and e.get("date")]
    if not daily_entries:
        return ""
    last_date_str = max(e["date"] for e in daily_entries)
    try:
        last_dt = datetime.strptime(last_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if (now - last_dt).days > 2:
            return (
                "⚠️ Note: No digest was delivered yesterday. "
                "You may have missed some articles."
            )
    except ValueError:
        pass
    return ""


# ---------------------------------------------------------------------------
# Feed fetching
# ---------------------------------------------------------------------------

def _clean_html(text: str, max_len: int = 600) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def fetch_feed(feed_cfg: dict, category: str) -> list | None:
    """Return list of article dicts, or None on hard failure."""
    try:
        parsed = feedparser.parse(feed_cfg["url"])
        max_articles = feed_cfg.get("max_articles", 10)
        articles = []
        for entry in parsed.entries[:max_articles]:
            url = entry.get("link", "").strip()
            if not url:
                continue
            title = entry.get("title", "Untitled").strip()
            raw_summary = (
                getattr(entry, "summary", None)
                or getattr(entry, "description", None)
                or ""
            )
            articles.append(
                {
                    "url": url,
                    "title": title,
                    "excerpt": _clean_html(raw_summary),
                    "feed_name": feed_cfg["name"],
                    "category": category,
                    "weight": float(feed_cfg.get("weight", 1.0)),
                }
            )
        return articles
    except Exception as exc:
        print(f"  [WARN] Skipping feed '{feed_cfg['name']}': {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Claude scoring
# ---------------------------------------------------------------------------

def score_article(client: anthropic.Anthropic, article: dict, config: dict) -> dict | None:
    """Score one article. Retries up to 3 times on failure. Returns None if all attempts fail."""
    hi_kw = config.get("high_priority_keywords", [])
    comp_kw = config.get("competitor_keywords", [])
    kw_ctx = ""
    if hi_kw:
        kw_ctx += f"\nHigh-priority keywords: {', '.join(hi_kw)}"
    if comp_kw:
        kw_ctx += f"\nCompetitor keywords (score high if present): {', '.join(comp_kw)}"

    prompt = (
        f"Article to score:\n"
        f"Title: {article['title']}\n"
        f"Source: {article['feed_name']}\n"
        f"Excerpt: {article['excerpt']}\n"
        f"{kw_ctx}\n\n"
        "Return ONLY valid JSON."
    )

    max_attempts = 3
    retry_delay = 5  # seconds between retries

    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=300,
                system=SCORING_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            # Tolerate ```json ... ``` wrappers
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in response")
            data = json.loads(match.group())
            article["score"] = int(data.get("score", 0))
            article["ai_summary"] = str(data.get("summary", "")).strip()
            article["why"] = str(data.get("why", "")).strip()
            article["weighted_score"] = article["score"] * article["weight"]
            return article
        except Exception as exc:
            print(
                f"  [WARN] Attempt {attempt}/{max_attempts} failed for "
                f"'{article['title'][:50]}': {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if attempt < max_attempts:
                time.sleep(retry_delay)

    return None


# ---------------------------------------------------------------------------
# Big Picture
# ---------------------------------------------------------------------------

def get_big_picture(client: anthropic.Anthropic, articles: list) -> str:
    headlines = "\n".join(
        f"- {a['title']} ({a['feed_name']}): {a['ai_summary']}"
        for a in articles
    )
    prompt = (
        "Here are today's top enterprise tech news stories:\n\n"
        f"{headlines}\n\n"
        "Write a 3-sentence 'Today's Big Picture' paragraph that synthesizes the "
        "overall themes and what they mean for enterprise software buyers and ERP "
        "solution consultants. Be direct and insightful."
    )
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=350,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as exc:
        print(f"  [WARN] Big Picture generation failed: {exc}", file=sys.stderr)
        return (
            "Today's digest spans key developments in enterprise AI and ERP — "
            "worth a careful read for any solution consultant staying ahead of the curve."
        )


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def _score_color(score: int) -> str:
    if score >= 9:
        return "#10b981"  # emerald
    if score >= 7:
        return "#f59e0b"  # amber
    return "#ef4444"  # red


def build_html(articles: list, big_picture: str, run_date: str, stale_warning: str = "") -> str:
    cards = ""
    for article in articles:
        color = _score_color(article["score"])
        cards += f"""
        <div style="background:#ffffff;border-radius:10px;padding:20px 22px;
                    margin-bottom:14px;border-left:4px solid {color};
                    box-shadow:0 1px 4px rgba(0,0,0,0.07);">
          <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:8px;">
            <tr>
              <td style="font-size:11px;color:#888;text-transform:uppercase;
                         letter-spacing:0.6px;">{article['feed_name']}</td>
              <td align="right">
                <span style="background:{color};color:#fff;border-radius:12px;
                             padding:2px 9px;font-size:11px;font-weight:700;
                             white-space:nowrap;">{article['score']}/10</span>
              </td>
            </tr>
          </table>
          <h3 style="margin:0 0 8px;font-size:16px;line-height:1.4;color:#111;">
            <a href="{article['url']}" style="color:#111;text-decoration:none;">
              {article['title']}
            </a>
          </h3>
          <p style="margin:0 0 6px;color:#444;font-size:14px;line-height:1.65;">
            {article['ai_summary']}
          </p>
          <span style="font-size:11px;color:#aaa;font-style:italic;">{article['why']}</span>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>FeedlyVP &mdash; {run_date}</title>
</head>
<body style="margin:0;padding:0;background:#f0f2f5;
             font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,
             'Helvetica Neue',Arial,sans-serif;">
  <div style="max-width:620px;margin:0 auto;padding:20px 12px;">

    <!-- Header -->
    <div style="background:linear-gradient(135deg,#1e1b4b 0%,#312e81 60%,#1e40af 100%);
                border-radius:12px;padding:30px 24px;margin-bottom:20px;text-align:center;">
      <h1 style="margin:0 0 6px;color:#fff;font-size:26px;font-weight:800;
                 letter-spacing:-0.5px;">FeedlyVP</h1>
      <p style="margin:0;color:#a5b4fc;font-size:13px;letter-spacing:0.3px;">
        {run_date} &nbsp;·&nbsp; Enterprise Tech Digest
      </p>
    </div>

    {f'''<!-- Stale warning banner -->
    <div style="background:#fef3c7;border:1px solid #f59e0b;border-radius:8px;
                padding:12px 16px;margin-bottom:16px;font-size:13px;color:#92400e;">
      {stale_warning}
    </div>''' if stale_warning else ''}

    <!-- Big Picture -->
    <div style="background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%);
                border-radius:10px;padding:22px 24px;margin-bottom:22px;">
      <p style="margin:0 0 10px;color:#c7d2fe;font-size:11px;text-transform:uppercase;
                letter-spacing:1.2px;font-weight:700;">Today&rsquo;s Big Picture</p>
      <p style="margin:0;color:#f5f3ff;font-size:15px;line-height:1.75;">
        {big_picture}
      </p>
    </div>

    <!-- Article count label -->
    <p style="margin:0 0 12px;font-size:12px;color:#6b7280;
              text-transform:uppercase;letter-spacing:0.8px;font-weight:600;">
      Top {len(articles)} Stories
    </p>

    <!-- Article cards -->
    {cards}

    <!-- Footer -->
    <div style="text-align:center;padding:22px 0 10px;color:#9ca3af;font-size:11px;">
      <p style="margin:0;">Generated by FeedlyVP &nbsp;&middot;&nbsp; Powered by Claude</p>
    </div>
  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# SendGrid
# ---------------------------------------------------------------------------

def send_email(html: str, subject: str, to_email: str, from_email: str) -> int:
    sg = SendGridAPIClient(os.environ["SENDGRID_API_KEY"])
    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        html_content=html,
    )
    response = sg.send(message)
    return response.status_code


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FeedlyVP Digest")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save email to preview.html and open in browser instead of sending",
    )
    args = parser.parse_args()

    print("=== FeedlyVP Digest ===")
    if args.preview:
        print("  [PREVIEW MODE — email will not be sent]")

    # In preview mode SendGrid credentials are not needed
    required_env: dict[str, str] = {"ANTHROPIC_API_KEY": "Anthropic API key"}
    if not args.preview:
        required_env.update(
            {
                "SENDGRID_API_KEY": "SendGrid API key",
                "FEEDLYVP_TO_EMAIL": "recipient email address",
                "FEEDLYVP_FROM_EMAIL": "sender email address",
            }
        )
    missing = [k for k in required_env if not os.environ.get(k, "").strip()]
    if missing:
        for k in missing:
            print(f"ERROR: {k} ({required_env[k]}) is not set or is empty", file=sys.stderr)
        sys.exit(1)

    # Confirm the key looks plausible before spending time on feed fetching
    api_key = os.environ["ANTHROPIC_API_KEY"].strip()
    if not api_key.startswith("sk-"):
        print(
            f"ERROR: ANTHROPIC_API_KEY does not look valid (got '{api_key[:8]}…'). "
            "Expected a key starting with 'sk-'.",
            file=sys.stderr,
        )
        sys.exit(1)

    now = datetime.now(timezone.utc)
    # Platform-safe date without leading zero on day
    run_date = now.strftime(f"%A, %B {now.day}, %Y")

    config = load_config()
    seen_urls = load_seen_urls()
    client = anthropic.Anthropic(api_key=api_key)

    # Check for stale digest (last successful run > 2 days ago)
    stale_warning = get_stale_warning(load_digest_log(), now)
    if stale_warning:
        print(f"\n[WARN] {stale_warning}")

    # ------------------------------------------------------------------
    # 1. Fetch feeds
    # ------------------------------------------------------------------
    print("\n[1/5] Fetching feeds …")
    all_new_articles: list = []
    feeds_ok = feeds_fail = 0

    for category in config.get("categories", []):
        for feed_cfg in category.get("feeds", []):
            result = fetch_feed(feed_cfg, category["name"])
            if result is None:
                feeds_fail += 1
                continue
            feeds_ok += 1
            new = [a for a in result if a["url"] not in seen_urls]
            all_new_articles.extend(new)
            print(f"  {feed_cfg['name']}: {len(result)} fetched, {len(new)} new")

    print(f"\n  Feeds: {feeds_ok} ok / {feeds_fail} failed")
    print(f"  Total new articles to score: {len(all_new_articles)}")

    if not all_new_articles:
        print("\nNo new articles found. Nothing to send.")
        save_seen_urls(seen_urls)
        print(f"\nSummary: {feeds_ok} feeds fetched, 0 articles scored, 0 sent")
        return

    # Cost guard — abort if article count is suspiciously high
    ARTICLE_LIMIT = 150
    if len(all_new_articles) > ARTICLE_LIMIT:
        alert = (
            f"FeedlyVP aborted — unusually high article count detected: "
            f"{len(all_new_articles)} articles.\n"
            "Check feeds.yaml for a misconfigured feed."
        )
        print(f"\nABORT: {alert}", file=sys.stderr)
        if not args.preview:
            send_email(
                alert,
                "FeedlyVP ABORTED — unusually high article count",
                os.environ["FEEDLYVP_TO_EMAIL"],
                os.environ["FEEDLYVP_FROM_EMAIL"],
            )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Score articles
    # ------------------------------------------------------------------
    print("\n[2/5] Scoring articles with Claude …")
    scored: list = []
    score_fail = 0
    total = len(all_new_articles)

    for idx, article in enumerate(all_new_articles, 1):
        label = article["title"][:58]
        print(f"  [{idx:>3}/{total}] {label:<58}", end="\r", flush=True)
        result = score_article(client, article, config)
        if result:
            scored.append(result)
            seen_urls.add(article["url"])  # mark seen only if successfully processed
        else:
            score_fail += 1
        time.sleep(0.25)  # gentle rate-limit buffer

    print(f"\n  Scored: {len(scored)} / Failed: {score_fail}      ")

    # ------------------------------------------------------------------
    # 3. Filter & rank
    # ------------------------------------------------------------------
    print("\n[3/5] Filtering and ranking …")
    qualified = [a for a in scored if a["score"] >= SCORE_THRESHOLD]
    qualified.sort(key=lambda x: x["weighted_score"], reverse=True)
    top_articles = qualified[:TOP_N]
    print(
        f"  {len(qualified)} articles scored {SCORE_THRESHOLD}+, "
        f"keeping top {len(top_articles)}"
    )

    if not top_articles:
        print("\nNo articles met the score threshold. Nothing to send.")
        save_seen_urls(seen_urls)
        print(f"\nSummary: {feeds_ok} feeds fetched, {len(scored)} articles scored, 0 sent")
        return

    # ------------------------------------------------------------------
    # 4. Big Picture
    # ------------------------------------------------------------------
    print("\n[4/5] Writing Today's Big Picture …")
    big_picture = get_big_picture(client, top_articles)

    # ------------------------------------------------------------------
    # 5. Build HTML and deliver
    # ------------------------------------------------------------------
    html = build_html(top_articles, big_picture, run_date, stale_warning)
    subject = f"FeedlyVP Digest — {run_date}"

    if args.preview:
        print("\n[5/5] Saving preview …")
        preview_path = BASE_DIR / "preview.html"
        preview_path.write_text(html, encoding="utf-8")
        print(f"  Saved → {preview_path}")
        # open in default browser on macOS
        subprocess.run(["open", str(preview_path)], check=False)
        print("  Opened in browser.")
        status = None
    else:
        print("\n[5/5] Building HTML and sending email …")
        status = send_email(
            html,
            subject,
            os.environ["FEEDLYVP_TO_EMAIL"],
            os.environ["FEEDLYVP_FROM_EMAIL"],
        )
        print(f"  SendGrid response: HTTP {status}")

        # Persist state only on a real send (preview is non-destructive)
        save_seen_urls(seen_urls)

        log = load_digest_log()
        log.append(
            {
                "type": "daily",
                "date": now.strftime("%Y-%m-%d"),
                "run_at": now.isoformat(),
                "feeds_fetched": feeds_ok,
                "feeds_failed": feeds_fail,
                "articles_scored": len(scored),
                "articles_sent": len(top_articles),
                "sendgrid_status": status,
                "big_picture": big_picture,
                "articles": [
                    {
                        "title": a["title"],
                        "url": a["url"],
                        "source": a["feed_name"],
                        "score": a["score"],
                        "summary": a["ai_summary"],
                        "category": a["category"],
                    }
                    for a in top_articles
                ],
            }
        )
        save_digest_log(log)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*36}")
    print(f"  Feeds fetched:     {feeds_ok}  ({feeds_fail} failed)")
    print(f"  Articles scored:   {len(scored)}  ({score_fail} failed)")
    print(f"  Articles in email: {len(top_articles)}")
    if args.preview:
        print(f"  Output:            {BASE_DIR / 'preview.html'}")
        print(f"  State files:       unchanged (preview mode)")
    else:
        print(f"  Sent to:           {os.environ['FEEDLYVP_TO_EMAIL']}")
        print(f"  SendGrid status:   {status}")
    print(f"{'='*36}")



if __name__ == "__main__":
    main()
