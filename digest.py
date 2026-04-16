#!/usr/bin/env python3
"""
FeedlyVP — Personal Tech News Digest
Fetches RSS feeds, scores articles with Claude, delivers via Telegram.
"""

import argparse
import asyncio
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
from telegram import Bot
from telegram.constants import ParseMode

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

SCORING_SYSTEM = """You are a relevance filter for a technology and business professional \
who follows tech strategy, AI, and the broader forces reshaping industries and companies.

Score HIGH (8-10):
- AI breakthroughs, agent systems, foundation models, and their real-world deployment
- How technology is changing business models, strategy, or competitive dynamics
- Enterprise software, cloud, and platform shifts (major vendors, M&A, market moves)
- Economic and policy forces acting on tech: regulation, labor, geopolitics, antitrust
- Organizational and leadership change driven by technology
- Substantive analysis from researchers, investors, or operators with original insight

Score MEDIUM (5-7):
- Solid tech or business reporting without a strong strategic angle
- Product launches or funding rounds with broader market implications
- Industry trends that are real but not urgent

Score LOW (1-4): consumer gadgets, smartphone specs, gaming, celebrity or lifestyle \
content, opinion pieces with no new information, and PR-driven announcements.

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

# Category display metadata: emoji + left-border accent colour
_CATEGORY_META: dict[str, tuple[str, str]] = {
    "AI & Research":            ("🤖", "#7c3aed"),
    "Enterprise & ERP":         ("🏢", "#2563eb"),
    "Tech Strategy & Analysis": ("📊", "#0f766e"),
    "Broad Tech News":          ("📰", "#475569"),
    "Community & Practitioner": ("💬", "#ea580c"),
}
_DEFAULT_CATEGORY_META = ("📌", "#6b7280")


def _badge(score: int) -> tuple[str, str]:
    """Return (background, text-color) for a score pill. 9-10 = green, 7-8 = blue."""
    if score >= 9:
        return ("#d1fae5", "#065f46")   # mint chip / deep green text
    return ("#dbeafe", "#1e40af")       # sky chip / deep blue text


def _group_by_category(articles: list) -> dict:
    """Preserve ranked order within each group; preserve first-seen category order."""
    groups: dict[str, list] = {}
    for a in articles:
        groups.setdefault(a.get("category", "Other"), []).append(a)
    return groups


def build_html(
    articles: list,
    big_picture: str,
    run_date: str,
    stale_warning: str = "",
    feeds_ok: int = 0,
) -> str:

    # ── stale warning banner (built before main f-string to avoid nesting) ──
    stale_banner = ""
    if stale_warning:
        stale_banner = f"""
    <div class="stale-banner"
         style="background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;
                padding:12px 16px;margin-bottom:20px;font-size:13px;
                color:#92400e;line-height:1.5;">
      {stale_warning}
    </div>"""

    # ── article sections grouped by category ────────────────────────────────
    sections_html = ""
    for category, cat_articles in _group_by_category(articles).items():
        emoji, accent = _CATEGORY_META.get(category, _DEFAULT_CATEGORY_META)

        cards_html = ""
        for a in cat_articles:
            badge_bg, badge_fg = _badge(a["score"])
            safe_title = (
                a["title"]
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )
            cards_html += f"""
      <div class="card"
           style="background:#ffffff;border-radius:12px;padding:20px 22px;
                  margin-bottom:12px;border:1px solid #f1f5f9;
                  box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(0,0,0,0.04);">
        <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:10px;">
          <tr>
            <td style="font-size:10px;font-variant:small-caps;font-weight:700;
                       color:#94a3b8;letter-spacing:0.6px;vertical-align:middle;">
              {a['feed_name']}
            </td>
            <td align="right" style="vertical-align:middle;width:1%;">
              <span style="display:inline-block;background:{badge_bg};color:{badge_fg};
                           border-radius:20px;padding:3px 11px;font-size:11px;
                           font-weight:800;white-space:nowrap;letter-spacing:0.2px;">
                {a['score']}&thinsp;/&thinsp;10
              </span>
            </td>
          </tr>
        </table>
        <h3 style="margin:0 0 10px;font-size:16px;line-height:1.45;
                   font-weight:700;color:#0f172a;">
          <a href="{a['url']}" class="card-title"
             style="color:#0f172a;text-decoration:none;">{safe_title}</a>
        </h3>
        <p class="summary"
           style="margin:0 0 10px;color:#475569;font-size:14px;line-height:1.72;">
          {a['ai_summary']}
        </p>
        <p class="why"
           style="margin:0;font-size:12px;color:#94a3b8;font-style:italic;line-height:1.5;">
          &#8627;&nbsp;{a['why']}
        </p>
      </div>"""

        sections_html += f"""
    <div style="margin-bottom:32px;">
      <div style="border-left:3px solid {accent};padding:2px 0 2px 14px;margin-bottom:16px;">
        <span class="section-label"
              style="font-size:11px;font-weight:800;color:#374151;
                     text-transform:uppercase;letter-spacing:1px;">
          {emoji}&nbsp; {category}
        </span>
      </div>
      {cards_html}
    </div>"""

    # ── assemble full email ──────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en" xmlns="http://www.w3.org/1999/xhtml">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <meta name="color-scheme" content="light dark">
  <meta name="supported-color-schemes" content="light dark">
  <title>FeedlyVP &mdash; {run_date}</title>
  <style>
    :root {{ color-scheme: light dark; }}

    /* ── Dark mode ───────────────────────────────────────── */
    @media (prefers-color-scheme: dark) {{
      body, .body-bg {{ background-color: #0c1117 !important; }}
      .wrapper       {{ background-color: #0c1117 !important; }}
      .card          {{ background-color: #161d27 !important;
                        border-color: #1e2d3d !important; }}
      .card-title    {{ color: #e2e8f0 !important; }}
      .summary       {{ color: #94a3b8 !important; }}
      .why           {{ color: #64748b !important; }}
      .section-label {{ color: #94a3b8 !important; }}
      .footer-text   {{ color: #475569 !important; }}
      .footer-sub    {{ color: #334155 !important; }}
      .footer-rule   {{ border-color: #1e293b !important; }}
      .stale-banner  {{ background-color: #451a03 !important;
                        border-color: #78350f !important;
                        color: #fcd34d !important; }}
    }}

    /* ── Mobile — 375 px (iPhone SE / iPhone 17 Pro) ────── */
    @media (max-width: 480px) {{
      .wrapper {{ padding: 12px 8px !important; }}
      .hero    {{ padding: 22px 18px !important; }}
      .card    {{ padding: 16px 16px !important; }}
    }}
  </style>
</head>
<body class="body-bg"
      style="margin:0;padding:0;background:#f1f5f9;-webkit-text-size-adjust:100%;
             font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,
             'Helvetica Neue',Arial,sans-serif;">

  <div class="wrapper" style="max-width:600px;margin:0 auto;padding:24px 16px;">

    <!-- ── HEADER ─────────────────────────────────────────────────── -->
    <div style="text-align:center;margin-bottom:28px;">
      <div style="display:inline-block;
                  background:linear-gradient(160deg,#0f172a 0%,#1e293b 100%);
                  border-radius:18px;padding:22px 36px;">
        <p style="margin:0 0 2px;color:#64748b;font-size:10px;font-weight:700;
                  text-transform:uppercase;letter-spacing:2px;">Enterprise Tech</p>
        <h1 style="margin:0 0 4px;color:#f8fafc;font-size:28px;font-weight:800;
                   letter-spacing:-0.5px;">FeedlyVP</h1>
        <p style="margin:0;color:#64748b;font-size:12px;">{run_date}</p>
      </div>
    </div>

    {stale_banner}

    <!-- ── BIG PICTURE HERO ───────────────────────────────────────── -->
    <div class="hero"
         style="background:linear-gradient(145deg,#312e81 0%,#4338ca 50%,#6d28d9 100%);
                border-radius:16px;padding:28px;margin-bottom:32px;
                box-shadow:0 8px 32px rgba(79,70,229,0.22);">
      <div style="display:inline-block;background:rgba(255,255,255,0.12);
                  border-radius:20px;padding:4px 14px;margin-bottom:14px;">
        <span style="color:rgba(255,255,255,0.9);font-size:10px;font-weight:800;
                     text-transform:uppercase;letter-spacing:1.5px;">
          &#10022;&thinsp;Today&rsquo;s Big Picture
        </span>
      </div>
      <p style="margin:0;color:#e0e7ff;font-size:15px;line-height:1.8;">
        {big_picture}
      </p>
    </div>

    <!-- ── ARTICLE SECTIONS ───────────────────────────────────────── -->
    {sections_html}

    <!-- ── FOOTER ─────────────────────────────────────────────────── -->
    <div class="footer-rule"
         style="border-top:1px solid #e2e8f0;padding-top:20px;text-align:center;">
      <p class="footer-text"
         style="margin:0 0 4px;color:#94a3b8;font-size:12px;">
        {run_date}&nbsp;&middot;&nbsp;{len(articles)} articles&nbsp;&middot;&nbsp;{feeds_ok} feeds checked
      </p>
      <p class="footer-sub" style="margin:0;color:#cbd5e1;font-size:11px;">
        FeedlyVP&nbsp;&middot;&nbsp;Powered by Claude
      </p>
    </div>

  </div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Telegram delivery
# ---------------------------------------------------------------------------

def _score_emoji(score: int) -> str:
    if score == 10: return "🔥"
    if score == 9:  return "⭐"
    if score == 8:  return "🔵"
    return "🟢"  # 7


def _tg_escape(text: str) -> str:
    """Escape a string for Telegram MarkdownV2. Backslash must be first."""
    text = text.replace("\\", "\\\\")
    for ch in ("_", "*", "[", "]", "(", ")", "~", "`", ">",
               "#", "+", "-", "=", "|", "{", "}", ".", "!"):
        text = text.replace(ch, f"\\{ch}")
    return text


def _tg_escape_url(url: str) -> str:
    """Escape a URL for use inside a MarkdownV2 inline link [text](url)."""
    return url.replace("\\", "\\\\").replace(")", "\\)")


async def _send_telegram_digest(
    bot_token: str,
    chat_id: str,
    articles: list,
    big_picture: str,
    run_date: str,
    feeds_ok: int,
    stale_warning: str,
) -> int:
    """Send full digest to Telegram. Returns total messages sent."""
    delay = 1  # seconds between messages to avoid rate limits
    sent = 0

    async with Bot(token=bot_token) as bot:
        # ── 1. Hero message ─────────────────────────────────────────────
        stale_line = f"\n\n⚠️ {_tg_escape(stale_warning)}" if stale_warning else ""
        hero = (
            f"🗞 *FeedlyVP — {_tg_escape(run_date)}*"
            f"{stale_line}\n\n"
            f"{_tg_escape(big_picture)}\n\n"
            f"📊 {len(articles)} articles scored 7\\+ from {feeds_ok} feeds today"
        )
        await bot.send_message(chat_id=chat_id, text=hero, parse_mode=ParseMode.MARKDOWN_V2)
        sent += 1
        await asyncio.sleep(delay)

        # ── 2. One message per article ───────────────────────────────────
        for a in articles:
            emoji = _score_emoji(a["score"])
            why_line = (
                f"\n💡 _{_tg_escape(a['why'])}_\n" if a.get("why") else "\n"
            )
            msg = (
                f"{emoji} *{_tg_escape(a['title'])}*\n\n"
                f"🗂 {_tg_escape(a['category'])} · 📰 {_tg_escape(a['feed_name'])}\n\n"
                f"{_tg_escape(a['ai_summary'])}\n"
                f"{why_line}\n"
                f"🔗 [Read Article]({_tg_escape_url(a['url'])})"
            )
            await bot.send_message(
                chat_id=chat_id, text=msg, parse_mode=ParseMode.MARKDOWN_V2
            )
            sent += 1
            await asyncio.sleep(delay)

        # ── 3. Closing message ───────────────────────────────────────────
        closing = (
            f"✅ Digest complete — {len(articles)} articles from {feeds_ok} feeds\n"
            "Next digest tomorrow at 7:30am PT"
        )
        await bot.send_message(chat_id=chat_id, text=closing)
        sent += 1

    return sent


def deliver_telegram(
    bot_token: str,
    chat_id: str,
    articles: list,
    big_picture: str,
    run_date: str,
    feeds_ok: int,
    stale_warning: str = "",
) -> int:
    """Synchronous entry point for Telegram delivery."""
    return asyncio.run(
        _send_telegram_digest(
            bot_token, chat_id, articles, big_picture, run_date, feeds_ok, stale_warning
        )
    )


def send_telegram_alert(bot_token: str, chat_id: str, text: str) -> None:
    """Send a plain-text alert message to Telegram (sync wrapper)."""
    async def _send() -> None:
        async with Bot(token=bot_token) as bot:
            await bot.send_message(chat_id=chat_id, text=text)
    asyncio.run(_send())


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
        print("  [PREVIEW MODE — Telegram will not be used]")

    # In preview mode Telegram credentials are not needed
    required_env: dict[str, str] = {"ANTHROPIC_API_KEY": "Anthropic API key"}
    if not args.preview:
        required_env.update(
            {
                "TELEGRAM_BOT_TOKEN": "Telegram bot token",
                "TELEGRAM_CHAT_ID": "Telegram chat ID",
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
            send_telegram_alert(
                os.environ["TELEGRAM_BOT_TOKEN"],
                os.environ["TELEGRAM_CHAT_ID"],
                f"🚨 {alert}",
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
    # 5. Deliver
    # ------------------------------------------------------------------
    if args.preview:
        print("\n[5/5] Saving preview …")
        html = build_html(top_articles, big_picture, run_date, stale_warning, feeds_ok)
        preview_path = BASE_DIR / "preview.html"
        preview_path.write_text(html, encoding="utf-8")
        print(f"  Saved → {preview_path}")
        # open in default browser on macOS
        subprocess.run(["open", str(preview_path)], check=False)
        print("  Opened in browser.")
        messages_sent = None
    else:
        print("\n[5/5] Delivering to Telegram …")
        messages_sent = deliver_telegram(
            os.environ["TELEGRAM_BOT_TOKEN"],
            os.environ["TELEGRAM_CHAT_ID"],
            top_articles,
            big_picture,
            run_date,
            feeds_ok,
            stale_warning,
        )
        print(f"  Sent {messages_sent} Telegram messages")

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
                "telegram_messages_sent": messages_sent,
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
    print(f"  Articles sent:     {len(top_articles)}")
    if args.preview:
        print(f"  Output:            {BASE_DIR / 'preview.html'}")
        print(f"  State files:       unchanged (preview mode)")
    else:
        print(f"  Telegram messages: {messages_sent}")
    print(f"{'='*36}")



if __name__ == "__main__":
    main()
