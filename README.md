# SME Growth Copilot

This is a local Streamlit hackathon prototype. Phase 1 generates private synthetic data; Phase 2 computes deterministic aggregate merchant facts; Phase 3 adds a private retention model, anomaly signal, eligible audience snapshot, and transparent opportunity scenarios. Phase 4 connects these to one merchant screen, exact campaign approval, a local mock partner, and aggregate simulated measurement. Phase 5 adds an optional aggregate-only OpenAI Copilot with validated responses and a local fallback.

The application is simulation-only. It contains no live Visa or issuer integration, no customer contact mechanism, and no real customer data. The architecture and prototype policies are described in [PROJECT_BRIEF.md](PROJECT_BRIEF.md), [JUDGING_CRITERIA.md](JUDGING_CRITERIA.md), and [AGENTS.md](AGENTS.md).

## Public demo deployment

This hackathon prototype can run on Streamlit Community Cloud from a GitHub repository. Deploy the repository's `app.py` entry point with Python 3.12; Community Cloud installs the packages in the root `requirements.txt`. Set the app's sharing mode to public to give viewers a shareable URL.

The first app run creates deterministic synthetic files under Git-ignored `data/generated/`. Do not commit `data/generated/`, `data/private/`, `runtime/`, `.env`, or an API key. The core demo works without an OpenAI key and labels its local Copilot answers. To enable the optional API explanation, add `OPENAI_API_KEY` as a top-level secret in the app's Community Cloud settings; never put its value in this repository.

This is a shared, resettable simulation. The local SQLite files under `runtime/` hold campaign and audit state for the running app instance. Community Cloud does not guarantee persistence of local files, so a restart can reset approvals and results. Public viewers share this demo state; use the local app for a controlled judging run if that shared behavior is unsuitable.

## Local setup on this Windows computer

Open PowerShell in the repository root and run:

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If Python 3.12 is available as `python` instead of `py`, use `python -m venv .venv` for the first command. The remaining commands work from the repository root in PowerShell.

## Launch the merchant demo

```powershell
& .\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL Streamlit prints, normally `http://localhost:8501`. Stop it with Ctrl+C. The screen should show **Visa Versa Cafe** and the **Synthetic data · Simulated activation** label. On first run, the app generates private synthetic fixtures if the Git-ignored `data/generated/` files are absent.

The demo clock is fixed at **2026-10-01**, regardless of today's date. In the app, follow **Diagnose → Audience → Recommend → Activate → Measure**:

1. Read the observed net-sales decline, weekday-lunch contributor and aggregate peer comparison.
2. Open **Why am I seeing this?** for windows, thresholds, model context and scenario assumptions.
3. Review the 126-person lapsed audience and the separate model-selected active-regular audience. The latter is the highest-risk 20% of active regulars before simulated program checks. No membership or individual score is shown.
4. In **Activate**, choose the lapsed win-back or model retention offer, check the offer, dates, simulated issuer-app channel and budget, then click **Review Campaign**. To run both, review and approve each separately.
5. Tick the exact-terms confirmation and click **Approve simulation**. This records approval, then submits once to the local simulator. The receipt says **Submitted to simulator. No customers contacted.** Submission is not delivery.
6. Click **Run simulated dispatch and results** for each approved campaign. The status timeline records acceptance, simulated dispatch and completion separately. Measure shows a privacy-checked whole-campaign aggregate for each. These purchases are campaign-associated, not proven incremental lift.
7. Type a question under **Ask the Copilot**, such as “Why did my revenue drop?”, “Are other cafes seeing the same thing?”, or “Did sales fall because of fewer transactions or smaller tickets?” The app maps paraphrases and simple follow-ups to one of six supported topics. Without a key, it shows a labelled deterministic answer. With a key, it requests a short AI explanation, validates the answer and its evidence references, then attaches numeric evidence from Python. The sales-components question uses one approved read-only function call before the explanation. Invalid or unavailable AI responses fall back to the safe template.

Editing any reviewed offer term, date or budget requires another **Review Campaign** and a new approval. Campaign and simulator state persists in Git-ignored `runtime/demo.sqlite`; repeated clicks reuse the same receipt and events. To start a fresh local demo from scratch, close the app and remove that specific local state file after you no longer need its history.

## Optional AI explanation

The core demo needs no API key. To enable the OpenAI Responses explanation for the current PowerShell session, set `OPENAI_API_KEY` as an environment variable in that session before launching Streamlit. You can optionally set `OPENAI_MODEL`; its default is `gpt-5.6-luna`. Keep the actual key out of `.env.example`, source control, screenshots and logs. The app does not automatically load a `.env` file. The request uses a short timeout and `store: false`; if the service is unavailable, the UI continues with a labelled local template. This integration follows the [OpenAI Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create) and [structured-output format](https://developers.openai.com/api/docs/guides/structured-outputs).

The Copilot accepts varied wording of six supported topics: sales drivers, peer comparisons, customer groups, next action, recommendation evidence, and transaction-count versus average-ticket analysis. Routing happens locally. It sends only the matched intent plus allowlisted aggregate evidence, units, time windows, and observed/estimated metadata. For the new comparison, the model must request `get_sales_components`; Python checks the call and returns the released aggregate sales, count, ticket, and arithmetic-effect evidence. The symmetric arithmetic split adds exactly to the observed sales change and does not establish a behavioral cause. The app does not send raw transaction data, private customer tokens, membership, individual risk scores, campaign state, or the typed message. Simple follow-ups use the prior safe intent; unclear questions ask for a rephrase. Chat history stores canonical topic labels and validated answers, never raw typed messages. Model prose cannot introduce its own numbers; the app attaches numerical evidence from deterministic analytics only after validating cited evidence IDs. It blocks identity requests and attempts to lower thresholds or bypass approval. Audit events in Git-ignored `runtime/copilot_audit.sqlite` hold only intent, outcome, evidence IDs and the fixed as-of date. The AI cannot approve or submit a campaign.

## Run tests

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

## Generate the private demo data

```powershell
& .\.venv\Scripts\python.exe scripts/generate_demo_data.py
```

The generator always uses seed `20261001` and as-of date **2026-10-01**, covering the preceding 365 days. It writes `merchants.parquet`, `transactions.parquet`, `program_eligibility.parquet`, and an aggregate `run_summary.json` under Git-ignored `data/generated/`. If PyArrow is unavailable, it writes CSV instead. Re-running with the same configuration produces the same rows. The transaction and eligibility files contain private synthetic customer tokens: keep them internal and do not pass them to the UI, exports, or an LLM.

The cafe fixture compares the latest 28 complete days with the preceding 28. Its lapsed-candidate qualification window runs from 135 to 45 days before the as-of date. A regular has at least three completed purchases in that window, including two weekday lunches (11:00–15:00); a lapsed regular has no later completed purchase. The targetable subset also has enrollment, granted retention permission for the simulated issuer-app channel, unexpired permission, no revocation, an available frequency cap, and no suppression. Phase 3 applies these checks in the private Audience Builder.

The second campaign ranks active regulars (at least three settled visits in 90 days and a visit within 45 days) by the retention model's predicted 30-day non-return probability. It selects the highest-risk 20%, then applies the same private program checks and 50-person activation minimum. Active and lapsed audiences are disjoint. A recency-baseline fallback does not activate this model-driven audience. The 10% or 15% lunch offer, scenario response rates and estimated sales are reviewed separately; model risk is not a prediction of offer response.

## Print the safe demo facts

After generating the private data, run:

```powershell
& .\.venv\Scripts\python.exe scripts/show_demo_facts.py
```

This prints an allowlisted aggregate object. The comparison uses two complete 28-day windows. Net sales include settled purchases after refunds; declines and full refunds are excluded. The fixed weekday and daypart breakdowns are released only when every cell and its complement has at least 10 distinct contributors in both windows. The lapsed summary requires at least 50 eligible people and a safe ineligible complement. Peer medians require at least 10 comparable merchants with sufficient data. Suppressed values are unavailable, never zero. All changes and drivers are observed descriptions, not proven causes.

## Print the Phase 3 decision objects

```powershell
& .\.venv\Scripts\python.exe scripts/show_phase3_decision.py
```

This command trains a deterministic logistic regression on prior 90-day behavior, evaluates a future 30-day non-return label on later time splits, and compares a recency-only baseline. It prints only aggregate evaluation and cohort explanations. The local Audience Builder persists its private membership and version in Git-ignored `runtime/audiences.sqlite`, then returns only an opaque merchant-scoped handle and safe count. The handle is a reference, not a security credential. No campaign is submitted or customer contacted.

Opportunity ranges are assumption-based scenarios, not confidence intervals or measured incremental lift. Incentive cost includes all assumed qualifying redemptions. Estimated net additional sales equal additional sales minus that cost; operating expenses, profit and ROI are not modeled. The synthetic model evaluation does not establish real-world performance.

## Current boundaries

- `src/sme_growth/contracts.py` defines allowlisted aggregate-facing response shapes and policy minimums. Analytics, audience and whole-campaign outcome release checks are implemented.
- `src/sme_growth/audiences/` privately selects eligible behavior-defined members and persists a versioned local snapshot. `src/sme_growth/measurement/` writes deduplicated private simulated events and releases only safe whole-campaign aggregates.
- `src/sme_growth/activation/` contains a persistent local-only mock adapter and version-bound approval service. There is no live transport, customer messaging, or payment action.
- `src/sme_growth/synthetic/generator.py` creates deterministic private fixtures and an allowlisted aggregate run summary. It is not a production data source.
- `src/sme_growth/analytics/` performs deterministic calculations and returns only safe aggregate contracts; token joins and peer-level results remain internal.
- `src/sme_growth/ml/` contains the private retention model and aggregate anomaly signal; `src/sme_growth/recommendations/` ranks observed-evidence proposals with explicit impact assumptions.
- `data/private/` is for private synthetic fixtures and `runtime/` holds the Audience Builder's private snapshot and local campaign state. Both are ignored by Git.
- `src/sme_growth/copilot/` builds the aggregate-only question payload, validates AI claims and references, supplies local templates, and records safe audit events. It cannot access private transaction frames or perform activation.
- `.env.example` lists optional configuration names only. Do not put real secrets in it or commit a `.env` file.

Merchant-facing responses, exports, and LLM context may contain only released aggregate information and opaque audience/campaign references. They must never contain customer PII, customer IDs or tokens, membership, contact details, or individual scores. The local package boundary is not production security isolation.
