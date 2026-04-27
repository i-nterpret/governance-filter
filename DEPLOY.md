# Deployment guide — pushing this repo to GitHub

You have a GitHub account; this repo is staged at
`00_Corpus/Papers/Drafts/Live/governance-filter-repo/`. The path from "files
on disk" to "live repo with sponsorship surface" is below.

---

## Step 1 — Install gh CLI (if not already)

From your Replit terminal (or local terminal):

```bash
# macOS
brew install gh

# Or on Replit / Linux
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list
sudo apt update && sudo apt install gh
```

Authenticate (one-time):

```bash
gh auth login
# choose: GitHub.com -> HTTPS -> authenticate via web browser
```

---

## Step 2 — Create the repo and push (5 commands)

From the staged directory:

```bash
cd "/Users/jamesdunn/Library/Mobile Documents/com~apple~CloudDocs/00_Corpus/Papers/Drafts/Live/governance-filter-repo"

git init
git add .
git commit -m "Initial release: three-substrate governance architecture (v0.1.0)"
gh repo create i-nterpret/governance-filter --public --source=. --remote=origin --push
git tag -a v0.1.0 -m "v0.1.0 — first public release alongside synthesis paper"
git push origin v0.1.0
```

That creates the public repo, pushes the code, and tags v0.1.0.

If the org name `i-nterpret` is taken or unavailable as a GitHub org you own, swap to your personal username:

```bash
gh repo create <YOUR_USERNAME>/governance-filter --public --source=. --remote=origin --push
```

(And update README.md + pyproject.toml + FUNDING.yml URLs accordingly.)

---

## Step 3 — Set up GitHub Sponsors (one-time, web UI)

GitHub Sponsors requires a one-time enrollment via the web UI; it cannot be set up via the API. Walk through:

1. Visit <https://github.com/sponsors>
2. Click "Join the waitlist" or "Set up GitHub Sponsors" (depending on your eligibility)
3. Provide: bank info (Stripe-backed), tax info (W-9 in US), profile copy
4. Define sponsorship tiers (matching the three-tier structure in the paper):
   - **$5/mo** — Research-tier supporter (name in CONTRIBUTORS)
   - **$50/mo** — Commercial-thank-you tier (logo + name)
   - **$500/mo** — Frontier-lab tier (priority issue review + early access to extensions)
   - **$5000/mo** — Sponsoring partner (named in next paper acknowledgments + collaboration call)
5. Approval typically takes 1–7 days

Once approved, GitHub auto-renders the "Sponsor" button on your repo using the `.github/FUNDING.yml` file already in this repo.

---

## Step 4 — Optional: Open Collective + Stripe

If you want broader payment options:

- **Open Collective** — <https://opencollective.com/create> — free for open-source projects, accepts cards + bank transfers, transparent ledger. Setup: ~30 min. Then update `FUNDING.yml` with the slug.
- **Stripe via i-nterpret.com** — embed a payment link or Buy Button on i-nterpret.com/sponsor; redirect from the `custom` URL in `FUNDING.yml`.

---

## Step 5 — Cross-link with Zenodo

After Wave 6 (Mon 2026-05-26) Zenodo DOIs are minted:

1. Add the synthesis-paper DOI to README.md citation block
2. Add the four Zenodo DOIs to a `CITATION.cff` file (GitHub will auto-render a "Cite this repository" button)
3. In Zenodo metadata, add `relatedIdentifiers` linking to the GitHub repo URL
4. Cross-references compile in both directions

---

## Step 6 — Verify the deploy

```bash
gh repo view i-nterpret/governance-filter --web
```

Should show:

- README rendered with code blocks + tables
- LICENSE auto-detected as Apache 2.0
- v0.1.0 release listed
- Sponsor button (after Step 3 completes)
- Languages chart showing Python

Test the package install end-to-end:

```bash
pip install git+https://github.com/i-nterpret/governance-filter.git
python -m governance_filter.benchmark
```

If the benchmark output matches the README's expected table, the deploy is verified.

---

## Optional: PyPI release

If you want `pip install governance-filter` (vs the git+ URL above) to work:

```bash
pip install build twine
python -m build
twine upload dist/*
```

Requires PyPI account + 2FA. Skip until after the Wave 6 launch is settled.

---

## API-only path (if you want to script everything)

Everything above can be done via the GitHub REST API:

```python
import os, requests
TOKEN = os.environ["GH_TOKEN"]
H = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json"}

# Create the repo
r = requests.post(
    "https://api.github.com/user/repos",
    headers=H,
    json={"name": "governance-filter", "description": "...", "private": False, "license_template": "apache-2.0"},
)

# Create a release after pushing
r = requests.post(
    "https://api.github.com/repos/i-nterpret/governance-filter/releases",
    headers=H,
    json={"tag_name": "v0.1.0", "name": "v0.1.0", "body": "Initial release..."},
)
```

But for first-time deploy, the `gh` CLI commands in Step 2 are simpler.
