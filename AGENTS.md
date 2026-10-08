# AGENTS.md - SME Growth Copilot


## Mission
Build a reliable hackathon prototype called SME Growth Copilot.


The product is an AI-powered business analyst for SME merchants. It turns synthetic payment data into clear answers to:
1. What happened?
2. Why did it happen?
3. What should I do?


The user journey is:
Diagnose -> Audience -> Recommend -> Activate -> Measure


## Primary Optimization Target
Optimize implementation decisions for the hackathon judging criteria:
- Business Impact: 25 points
- Role of AI: 20 points
- Feasibility: 20 points
- Responsible AI Controls: 15 points
- Prototype Demo: 20 points


When tradeoffs are necessary, prefer a stable, explainable, end-to-end prototype over broader functionality.


## Required Stack
Unless explicitly approved otherwise, use:
- Python
- Streamlit
- pandas / NumPy
- scikit-learn
- Plotly
- CSV or Parquet; local SQLite for mock campaign state
- pytest
- Git


Use an OpenAI API integration only for the Generative AI layer after deterministic analytics are working.


Do not introduce React, FastAPI, PostgreSQL, Docker, cloud infrastructure, microservices, or other major dependencies unless the user explicitly approves them.


## Core Demo Merchant
The primary demo merchant is Visa Versa Cafe.


The synthetic data should create a deterministic scenario with approximately:
- 8-10% recent sales decline
- 20-21% decline in weekday lunch transactions
- peer cafes broadly stable
- weaker repeat-customer activity
- ~126 eligible lapsed regulars after private program checks
- ~AED 21K historical monthly spend from the targetable lapsed group
- a weekday lunch win-back opportunity
- estimated opportunity calculated from explicit assumptions, not a fixed AED 8-12K claim


Small numerical variation is acceptable if the story stays coherent and deterministic.


## Data Rules
- Use synthetic data only.
- Use a fixed seed and explicit as-of date.
- Never create real personal data.
- Customer IDs are private pseudonymous synthetic tokens.
- Merchants never receive customer PII, tokens, member lists, contact details or individual scores.
- Return allowlisted aggregate objects only to the UI, exports and LLM.
- Do not display competitor names or competitor-level performance.
- Peer benchmarks must use an aggregate group.
- Defaults: 10 peer merchants, 50 eligible people for activation.
- Outcome cells need 10 contributors and safe complements; suppress related totals and restrict overlapping report cuts. These are prototype policies.


## Architecture Rules
Use this logical flow:


Synthetic transactions / merchant data
-> deterministic analytics
-> structured facts
-> private ML / Audience Builder / eligibility checks
-> aggregate facts, safe audience handle and opportunity scoring
-> Generative AI explanation and Streamlit review
-> human approval -> mock Visa/issuer activation
-> private simulated events -> aggregate measurement


The LLM must never be the source of truth for numerical business metrics. Keep this as a simple local prototype with internal boundaries; do not claim production isolation or require a new web service. Real deployments need separately enforced permissions.


## Analytics Rules
Core values must be calculated in Python before being passed to AI.


Important analytics include:
- sales change
- transaction change
- average ticket
- daypart change
- repeat/new/lapsed customer analysis
- peer benchmark
- historical targetable spend
- opportunity sizing


Prefer small composable functions with tests.


## ML Rules
ML must have a clear decision purpose.


Recommended components:
- anomaly detection
- retention/churn prediction
- opportunity scoring


Prefer explainable models, time-based validation without future leakage, a recency baseline and honest calibration/limitations. Individual scores remain private. Do not add a model solely to make the project sound more technical.


## Generative AI Rules
The AI may:
- explain facts
- answer merchant questions
- summarize evidence
- explain recommendations
- suggest approved next steps


The AI must not:
- invent numerical values
- calculate business KPIs from raw transactions
- reveal individual competitor performance
- claim an estimate is guaranteed
- approve or execute a campaign
- reveal customer tokens/membership or infer sensitive attributes
- claim a simulated campaign is live
- make unsupported recommendations


AI prompts must use only supplied aggregate facts and approved opportunities. Validate output references and numerical claims; use a labelled template fallback when invalid. User text cannot override privacy or approval rules.


## Responsible AI Rules
Implement and demonstrate:
- synthetic data only
- private synthetic customer IDs and aggregate-only merchant responses
- privacy-safe peer benchmarks
- minimum peer-group threshold
- observed versus estimated labels
- evidence for important recommendations
- confidence indication where practical
- human confirmation before action
- safe fallback when evidence is insufficient
- safe audit references/events without raw prompts, IDs, contacts or secrets


If there is insufficient evidence, return a transparent insufficient-data response instead of guessing.


## Audience and Activation Rules
- Select allowlisted behavior-based cohorts privately; no sensitive or named-competitor targeting.
- Separate behavioral suitability from enrollment and valid purpose/channel permission.
- Unknown, expired, revoked, suppressed or frequency-capped records are ineligible.
- Return a merchant-scoped audience handle/version and safe summary, never membership.
- Require explicit approval of exact audience/offer/dates/channel/budget; edits reset approval.
- Recheck eligibility at dispatch; changed membership requires a new version and approval.
- Persist request idempotency and campaign state; repeated clicks return the same receipt.
- Reconcile an unknown submission before retrying; deduplicate simulated events.
- Implement only a local mock adapter. No real Visa credentials, live sending or payments.
- Distinguish submitted, accepted, simulated dispatched, completed, blocked and failed.
- Receipt: Submitted to simulator. No customers contacted.
- Measure aggregates only. Descriptive sales are not causal lift; use valid holdouts for incremental estimates.
- Public Visa patterns are context, not access guarantees or endorsement.


## UX Rules
The product is for a busy SME owner, not a data scientist.


Prefer:
- concise cards
- plain language
- clear hierarchy
- small number of actionable KPIs
- conversational questions
- evidence hidden behind expandable details


Avoid:
- dense BI dashboards
- unnecessary technical terminology
- excessive charts
- screens that do not serve the core demo flow


## Business Impact Rules
Important recommendations should display quantified impact where supported, such as:
- targetable customers
- historical spend
- estimated reactivation rate
- estimated incremental sales
- campaign cost
- simulated campaign-associated sales; incrementality only with a valid control
- estimated ROI


Always label forecasts as estimates and campaign events as simulated. Include incentive/program costs and margin; profit ROI is not a revenue multiple. Suppressed values are not zero.


## Testing Rules
Test important analytics and demo assumptions, including:
- synthetic data generation is reproducible
- Visa Versa Cafe shows the intended decline pattern
- peer group remains broadly stable
- lapsed customer count is in the intended range
- peer/audience/result privacy thresholds and complementary suppression work
- private fields cannot leak through UI, exports, errors, logs or AI context
- approval changes, expiry, eligibility revocation and duplicate requests are handled
- opportunity engine ranks the intended demo action correctly
- AI input contains only approved structured facts
- AI fallback activates when evidence is insufficient


After material changes:
1. Run the relevant tests.
2. Run the full test suite when practical.
3. Fix failures before adding new features.
4. Report what was changed and how it was verified.


## Coding Practices
- Keep modules small and readable.
- Use clear names and type hints where useful.
- Avoid premature abstraction.
- Avoid unnecessary dependencies.
- Keep secrets in environment variables; never commit API keys.
- Maintain .env.example but not a real .env secret file in Git.
- Prefer deterministic calculations.
- Add comments only where the business logic is not obvious.


## Change Discipline
- Read PROJECT_BRIEF.md and JUDGING_CRITERIA.md when relevant.
- Inspect existing implementation before changing it.
- Make the smallest coherent change that completes the requested task.
- Do not refactor unrelated working code.
- Do not add new features unless requested.
- Run tests and explain verification.

## Current Phase Boundaries
- Keep `app.py` focused on one Visa Versa Cafe journey, with synthetic/simulated labels and only released aggregate data. Streamlit session state may hold safe campaign references, canonical Copilot topics and validated aggregate answers, never raw typed messages or private customer data.
- Keep `src/sme_growth/contracts.py` as the allowlisted merchant-facing contract boundary. Audience references and campaign references are opaque; customer PII, customer IDs or tokens, membership, contacts and individual scores cannot cross it.
- Keep `src/sme_growth/audiences/` and `src/sme_growth/measurement/` private internally. Release only allowlisted whole-campaign aggregates after privacy checks. The minimums are 10 peer merchants, 50 eligible people for activation, and 10 contributors plus safe complements for outcome cells.
- Keep `src/sme_growth/activation/` mock-only, with exact version-bound approval and persistent idempotent local state. No live Visa/issuer connection, message sending, payments or customer contact.
- Keep private fixtures under `data/private/` and local campaign state under `runtime/`; both directories are ignored by Git.
- Phase 1 adds deterministic synthetic merchant, transaction, and private program-eligibility generation in `src/sme_growth/synthetic/`. Store generated files under Git-ignored `data/generated/` and print only allowlisted aggregate run metadata. Use a fixed as-of date; never derive the demo from the wall clock.
- Phase 2 adds deterministic analytics in `src/sme_growth/analytics/`. The only public output is allowlisted aggregate contracts with explicit units, windows, as-of date and evidence type. Keep all customer and peer joins internal. Fixed breakdowns are released only if every cell and complement is safe; no arbitrary drill-downs.
- Phase 3 adds private temporal retention modelling, a transparent merchant anomaly signal, a local Audience Builder with private SQLite membership, and assumption-based opportunity ranking. Public model output is cohort-level only; audience handles are scoped references, not authorization tokens. Estimated impact cannot be presented as measured incrementality.
- Phase 4 adds the single-merchant Streamlit journey, exact human approval before local mock submission, private deduplicated simulation events, and one privacy-safe aggregate result. Submission, acceptance, simulated dispatch, completion, blocked and failed states remain distinct. Simulated campaign-associated sales are descriptive without a holdout.
- Phase 5 adds the optional OpenAI Responses Copilot over allowlisted aggregate facts only. Locally route typed paraphrases and simple follow-ups to five supported intents; send the intent without raw user text. Numeric evidence is rendered from deterministic analytics after model-output validation. No key, insufficient evidence or invalid output uses a labelled safe template. Audit only event codes and evidence references, never raw prompts, outputs, customer data or secrets. The LLM has no campaign approval or dispatch authority.
- Do not add real partner access or customer contact until separately requested.


## Definition of Done for the MVP
The project is MVP-complete only when a user can:
1. Open the Streamlit app.
2. See Visa Versa Cafe performance.
3. See a detected performance issue.
4. Ask why performance changed.
5. Receive an evidence-grounded explanation.
6. See a privacy-safe peer benchmark.
7. See a ranked recommended action.
8. See quantified estimated opportunity.
9. Review evidence and responsible-AI information.
10. Review an eligible audience and explicitly approve simulated partner activation.
11. See safe aggregate simulated measurement with honest attribution labels.
12. Run the test suite successfully.


## Stop Conditions
Seek approval for material scope, architecture, dependency or privacy-control changes outside the authorized task.
