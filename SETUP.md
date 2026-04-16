# FeedlyVP — GitHub Actions Setup Guide

Everything you need to get the daily digest running in the cloud,
read logs when something breaks, and tweak your feed list without
ever opening a terminal.

---

## 1. Add secrets to the repo

GitHub Actions reads credentials from **encrypted Secrets** — never from code.

1. Go to your repo on GitHub: `github.com/<you>/feedlydigest`
2. Click **Settings → Secrets and variables → Actions → New repository secret**
3. Add each of these four secrets:

| Secret name | Value |
|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic key (`sk-ant-...`) |
| `SENDGRID_API_KEY` | Your SendGrid key (`SG....`) |
| `FEEDLYDIGEST_FROM_EMAIL` | Verified sender address in SendGrid |
| `FEEDLYDIGEST_TO_EMAIL` | Where you want the digest delivered |

> **SendGrid sender verification:** `FEEDLYDIGEST_FROM_EMAIL` must be a verified
> sender or domain in SendGrid → Settings → Sender Authentication,
> otherwise emails will be silently rejected.

---

## 2. Push the repo

From your local `feedlyVP` folder:

```bash
git init
git remote add origin https://github.com/<you>/feedlydigest.git
git add .
git commit -m "Initial commit"
git push -u origin main
```

The workflow file (`.github/workflows/daily-digest.yml`) is picked up
automatically as soon as it lands on the default branch.

---

## 3. Run a manual test from the GitHub website

You don't need your laptop for this.

1. Open `github.com/<you>/feedlydigest`
2. Click the **Actions** tab (top nav bar)
3. In the left sidebar, click **Daily Digest**
4. Click the **Run workflow** dropdown on the right side
5. Leave the branch as `main` and click the green **Run workflow** button
6. A new run row appears — click it to watch the live log

The digest email will arrive within a few minutes of the run completing.

---

## 4. Read the logs when something goes wrong

**Find the failed run:**
1. **Actions** tab → click the red ✗ run
2. Click the **run-digest** job on the left
3. Expand the failing step to see the full output

**Common problems and fixes:**

| Symptom | Likely cause | Fix |
|---|---|---|
| `ANTHROPIC_API_KEY not set` | Secret name typo or missing | Check Settings → Secrets |
| `400 Bad Request` from SendGrid | From-address not verified | Verify sender in SendGrid dashboard |
| Feed fetch warnings in log | Feed URL changed or site is down | Update `feeds.yaml` (see section 5) |
| `0 articles scored, 0 sent` | All articles already in `seen_urls.json` | Normal if re-run same day; or reset: `echo "[]" > seen_urls.json` and push |
| Workflow never triggers | Repo has no recent commits | GitHub pauses schedules on repos with no activity for 60 days; do a manual run to reactivate |

**Persistent run history** is saved in `digest_log.json` in the repo —
each row has `run_at`, `articles_scored`, `articles_sent`, and `sendgrid_status`.

---

## 5. Add or remove a feed — no local tools needed

You can edit `feeds.yaml` directly in the GitHub web editor:

1. Open `github.com/<you>/feedlydigest`
2. Click `feeds.yaml` in the file list
3. Click the **pencil icon** (Edit this file) in the top-right
4. Make your changes:
   - **Add a feed:** copy an existing block and change `name`, `url`, `weight`, `max_articles`
   - **Remove a feed:** delete its block (the four lines starting with `- name:`)
   - **Change the weight:** edit the `weight` value (higher = ranked first when scores are equal)
5. Scroll down, write a short commit message, and click **Commit changes**

The next scheduled run (or any manual run) picks up the changes automatically.
No pull request needed — committing directly to `main` is fine for a personal digest.

---

## 6. Schedule details

The cron is set to **15:30 UTC**, which equals:

- **7:30 AM PST** (November – March, Pacific Standard Time)
- **8:30 AM PDT** (March – November, Pacific Daylight Time)

The weekly digest runs on the same UTC time every Sunday.

---

## 7. How state is persisted between runs

After each successful run, the workflow automatically commits two files
back to `main`:

- **`seen_urls.json`** — URLs already processed (prevents duplicate articles)
- **`digest_log.json`** — append-only run history

The commit message is `chore: update digest state [skip ci]`.
The `[skip ci]` tag tells GitHub not to treat this as a new trigger.
