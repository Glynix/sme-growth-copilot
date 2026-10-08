# SME Growth Copilot - Project Brief


## 1. Hackathon Theme
Merchant & SME Growth


## 2. Product Name
SME Growth Copilot


## 3. One-line proposition
An AI-powered business analyst that turns payment data into simple growth decisions, privacy-safe audiences and approved campaign activation for SME owners.


## 4. Business Problem
SMEs have data, but often lack the analytics capability to turn it into decisions.


An SME owner can usually see how much they sold, but often cannot answer:
- Why did my sales fall?
- Am I losing customers, or is the whole market slowing?
- How do I compare with similar businesses?
- What should I actually do about it?


Large businesses can use dedicated analytics teams for this type of diagnosis. The goal of SME Growth Copilot is to make similar decision support accessible to smaller merchants through a simple conversational product.


## 5. Proposed Solution
SME Growth Copilot turns payment and merchant data into three simple answers:


1. WHAT happened?
2. WHY did it happen?
3. WHAT should I do?


The Copilot should also benchmark a merchant against a privacy-safe peer group so the SME can understand whether a change is business-specific or part of a broader market trend. It then connects the insight to an eligible audience and an approved offer. Merchants never receive customer PII, customer IDs or audience membership. The prototype simulates matching and activation through a Visa/issuer channel; actual access and partner participation are not guaranteed.


## 6. Core User Journey
The product experience follows:


Diagnose -> Audience -> Recommend -> Activate -> Measure


### Diagnose
Show recent performance at a glance and explain the main observed contributors using analytical evidence and safe peer comparisons.


### Audience
Select a behavior-based audience inside the private partner simulator, check enrollment and current purpose/channel permission, and show only a safe summary and audience reference.


### Recommend
Identify and rank actionable growth opportunities.


### Activate
Let the merchant review the audience, offer, dates, channel and budget. Require approval of that exact version, then submit to a mock Visa/issuer adapter. No customer is contacted. A change invalidates approval; retrying must not create a duplicate campaign.


### Measure
Show aggregate simulated campaign outcomes, costs and clearly labelled estimates. Submission is not delivery or redemption. Without a valid control, show attributed outcomes rather than claiming incremental lift.


## 7. MVP Demo Merchant
Use one synthetic merchant called Visa Versa Cafe.


The demo data should intentionally create the following story:
- Recent sales decline of approximately 8-10%.
- Weekday lunch transactions decline by approximately 20-21%.
- Similar cafes are broadly stable.
- Repeat-customer activity has weakened.
- Approximately 126 lapsed regulars pass the simulated campaign eligibility checks after no observed visit for 45+ days.
- Their historical spend is approximately AED 21K per month.
- The highest-priority recommendation is a weekday lunch win-back campaign.
- The merchant approves a simulated partner campaign and receives aggregate results.
- Estimated opportunity is calculated from explicit reach, response, visit and spend assumptions; it is not fixed to AED 8-12K.


These values are demo targets, not real-world claims. They may be adjusted slightly during implementation as long as the story remains consistent and deterministic. Use a fixed seed and explicit as-of date. Define regular, lapsed, baseline and lunch windows in code; distinguish all lapsed candidates from the eligible audience.


## 8. Example Merchant Questions
- Why were my sales down last month?
- Is the whole market slowing, or is this specific to my business?
- Which customer groups am I losing?
- What promotion should I run?
- Where is my biggest growth opportunity?
- What evidence supports this recommendation?
- How can I reach this audience without receiving their identities?
- Was the campaign submitted, and what do the aggregate results show?


## 9. Example Copilot Response
Question: Why were my sales down last month?


Expected response structure:
- State the sales change.
- Identify the primary analytical driver.
- Compare the merchant with the peer group.
- Mention repeat-customer/lapsed-customer evidence if relevant.
- Do not invent values.
- Distinguish observed facts from estimates.


Example:
"Sales were down 8.2% versus the recent baseline. The largest deterioration came from weekday lunch, where transactions fell 21.1%. Similar cafes were broadly stable, which suggests the decline is more specific to your business than to the wider category. The partner simulator also identified an eligible audience of 126 lapsed regulars. You can review a win-back offer without receiving customer identities."


## 10. Recommended Action Example
Recommended action:
Review a 15% Monday-to-Thursday lunch win-back offer for previously active lunch customers who have become inactive and pass program eligibility. Set dates, budget and redemption cap before approving simulated partner activation.


Illustrative impact block:
- Eligible audience: 126; identities stay with the partner simulator
- Historical monthly spend: AED 21,000, as context only
- Assumed incremental return rate: 8-12%
- Assumed additional visits: 2; spend per visit: AED 60
- Estimated incremental sales: 126 x 8-12% x 2 x 60 = AED 1,210-1,814


These scenario assumptions are not guaranteed results or a confidence interval. Show campaign discount cost and estimated net additional sales after those discounts. Do not assume a contribution margin or report profit or ROI. Include incentives on all qualifying redemptions, without double-counting discounts.


The mock adapter returns an aggregate receipt: Submitted to simulator. No customers contacted. It tracks submission, simulated dispatch and outcomes separately. Example simulated results may show 18 purchasers, AED 2,160 campaign-associated sales and AED 400 cost. These descriptive figures do not establish incremental sales or ROI without a valid holdout. A holdout extension must assign groups before exposure, use all assigned people in denominators and report uncertainty.


## 11. Data Sources
The hackathon prototype uses synthetic data only. Customer-level records belong to private internal modules and never appear in merchant-facing outputs, exports, logs or LLM context. This logical local boundary is not production security isolation.


### Payment / Transaction Data
- transaction_id
- transaction_datetime
- amount, currency, transaction_status, refund_amount
- merchant_id
- merchant_category / MCC
- synthetic_customer_id (private pseudonymous token, not an anonymous-data guarantee)
- card_origin: domestic / international
- merchant_location


### Merchant Data
- merchant_id
- merchant_category
- merchant_location
- merchant_size
- opening_date / tenure


### Private Eligibility and Audience Data
- synthetic_customer_id, enrolled, permission_state, purpose, channel
- valid_until, revoked_at, frequency_cap_ok, suppressed
- audience_id, audience_version, private_membership, snapshot, expires_at
- Unknown, expired or revoked permission is not eligible.


### Safe Campaign and Measurement Data
- audience_id/version, safe_definition, eligible_count (no member list)
- campaign_id/version, merchant_id, offer_terms, dates, budget, channel
- approval_id, approved_definition_hash, idempotency_key, status
- simulation=true, aggregate outcomes, reporting window, suppression flags
- Private events may link tokens internally; public results must not.


### Optional External Data
- public holidays
- local events
- weather
- tourism trends


External data is optional and should not be added unless the core MVP already works.


## 12. Suggested Synthetic Data Scale
- ~300 merchants
- ~20,000 private synthetic customer tokens
- 12 months of history
- ~300,000 to 500,000 transactions
- Fixed random seed and explicit as-of date for reproducibility


## 13. Analytics Requirements
The deterministic analytics layer should be able to calculate:
- Sales performance versus prior period / baseline
- Transaction count change
- Average ticket change
- Day-of-week performance
- Daypart performance
- New versus repeat customer trends
- Lapsed regular customers
- Peer-group benchmark
- Historical spend of targetable customers
- Candidate growth opportunities
- Estimated opportunity size


Suggested analytical functions:
- get_sales_performance()
- get_daypart_analysis()
- get_repeat_customer_analysis()
- get_peer_benchmark()
- get_lapsed_customer_summary()
- build_eligible_audience() (private membership; safe summary only)
- calculate_opportunity()


## 14. Machine Learning Requirements
Use ML only where it improves decision-making.


Recommended MVP components:
1. Anomaly detection for unusual merchant performance.
2. Customer retention/churn prediction for previously regular customers.
3. Opportunity scoring to rank possible merchant actions.


Prefer explainable and maintainable models such as logistic regression or gradient boosting. Model complexity is not a goal by itself. Define the future non-return label, use temporal splits without future leakage, compare with a recency baseline and report calibration/limitations. Individual scores remain private; synthetic evaluation does not prove real-world accuracy.


## 15. Opportunity Engine
The recommendation engine should first generate structured candidate opportunities from analytics.


Examples:
- Recover lapsed weekday-lunch customers.
- Improve underperforming dayparts.
- Offer an approved incentive to an eligible behavior-defined group.


Audience membership is built privately using allowlisted rules. No sensitive traits, inferred residence or named competitor targeting. Acquisition is an optional later scenario requiring permitted category-level data and program access.


A simple scoring concept is sufficient:


Opportunity Score = Potential Revenue x Confidence x Actionability


The highest-ranked approved opportunity is passed to the Generative AI layer for explanation.


## 16. Generative AI Role
Generative AI is a language and decision-support layer over structured evidence.


The AI should:
- Explain analytical results in simple language.
- Answer merchant questions conversationally.
- Explain why an action is recommended.
- Summarize supporting evidence.
- Suggest practical next steps.


The AI should NOT:
- Calculate core metrics from raw data itself.
- Invent transaction values, percentages, or customer counts.
- Reveal competitor-specific information.
- Approve or execute campaigns.
- Reveal customer identities, tokens, member lists or individual scores.
- Invent customer attributes or claim real Visa/issuer access.
- Present estimates as guaranteed outcomes.


## 17. Responsible AI Requirements
- Synthetic data only for the prototype.
- No personally identifiable information.
- Keep pseudonymous synthetic customer IDs and membership private; expose safe audience references only.
- Do not show competitor names or merchant-level competitor performance.
- Apply a minimum peer-group threshold before showing benchmarks.
- Defaults: at least 10 peer merchants and 50 eligible people for activation.
- Report outcome cells only when at least 10 contributors and safe complements remain; suppress related totals and restrict overlapping report cuts. These are prototype policies, not Visa standards.
- Numerical claims must come from deterministic analytics or approved model outputs.
- Predictions and business impact values must be clearly labelled as estimates.
- Merchant approval must match the exact campaign/audience version.
- Recheck eligibility before simulated dispatch; changed membership requires revised approval.
- Persist campaign state and idempotency keys so duplicate clicks/retries cannot repeat execution.
- Provide evidence for important recommendations.
- Return an insufficient-evidence message when the data does not support a confident answer.
- Log safe evidence references and events without raw prompts, private IDs, contacts or secrets.
- Validate model output and test privacy across UI, exports, browser state, errors and AI context.
- Map data permissions, roles, retention, consumer rights and marketing rules for a future pilot; do not claim compliance certification.


## 18. Prototype UX
The prototype should be a simple Streamlit web app.


Main merchant screen:
- Merchant name
- Sales KPI and change
- Main reason identified
- Peer benchmark
- Eligible audience summary and privacy status
- Opportunity identified
- Recommended action
- Estimated opportunity
- Review Campaign -> Approve simulation -> aggregate receipt
- Persistent Synthetic data / Simulated activation label
- Aggregate Measure view with honest missing/suppressed results
- Conversational Ask the Copilot area


Important recommendations should have a "Why am I seeing this?" or evidence panel.


## 19. Suggested Technology Stack
- Python
- Streamlit
- pandas / NumPy
- scikit-learn
- Plotly
- CSV or Parquet; local SQLite for persistent mock campaign state
- Internal mock Visa/issuer adapter, with no live transport
- OpenAI API for the Generative AI layer
- pytest
- Git


Keep the architecture simple for the hackathon.


## 20. Definition of MVP Success
The prototype must demonstrate one complete end-to-end merchant journey:
1. Detect a performance issue.
2. Explain what changed.
3. Explain why it changed.
4. Benchmark against peers.
5. Identify an eligible privacy-safe audience and growth opportunity.
6. Recommend an action.
7. Quantify the estimated business opportunity.
8. Require version-bound approval and submit a simulated partner campaign.
9. Show aggregate simulated measurement, with privacy suppression and honest attribution.
10. Allow conversational questions grounded in the same evidence.


## 21. Demo Principle
The demo must be deterministic and stable. Prefer a smaller working product with clear evidence over a wider product with partially working features.


Visa/issuer activation follows publicly described patterns, not guaranteed project access. Validate data rights, program participation, available channels, API/commercial arrangements and security/privacy obligations before any real pilot. No real data, customer outreach or money movement is part of this MVP.
