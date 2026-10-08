# Hackathon Judging Criteria


## Session Format
Each team has 10 minutes total, including:
- Presentation
- Prototype demo
- Q&A


## Recommended Presentation Structure


### Slide 1 - Team Introduction
Recommended details:
- Team name
- Company name
- Team member names and job titles
- Photos are optional


### Slide 2 - Overview of the Idea
Recommended details:
- Theme
- Track
- Business problem being solved
- Proposed solution
- Platform
- Data sources


## Scored Criteria - 100 Points Total


### 1. Business Impact - 25 points
Standard:
Strong business value with measurable benefits for stakeholders.


Recommended evidence:
Include key metrics that quantify impact.


For SME Growth Copilot, show:
- Addressable merchant opportunity
- Eligible audience reach without customer identity disclosure
- Estimated incremental sales
- Potential retention/reactivation
- Transparent cost/margin assumptions and aggregate simulated outcomes
- Incremental impact only with a valid experimental comparison


### 2. Role of AI - 20 points
Standard:
AI is central to the solution and contributes through meaningful decisioning.


Recommended evidence:
Show AI answering:
- What happened?
- Why did it happen?
- What should you do?


For SME Growth Copilot, AI should sit on top of deterministic analytics and ML evidence, explain findings, prioritize approved opportunities, and support merchant decision-making.


### 3. Feasibility - 20 points
Standard:
Realistic data and architecture assumptions.


For SME Growth Copilot, demonstrate:
- Synthetic but realistic payment and merchant data
- Simple reproducible architecture
- Fast local processing
- Clear separation between analytics, private audiences, mock activation, AI and UI
- Visa/issuer access and channels described as assumptions to validate
- No unnecessary infrastructure dependencies


### 4. Responsible AI Controls - 15 points
Standard:
Guardrails for privacy, human review, and risk management are addressed; regulatory considerations are mapped.


For SME Growth Copilot, demonstrate:
- Synthetic data only in the prototype
- Private synthetic tokens; merchants never receive customer PII or membership
- Minimum peer/audience thresholds and safe aggregate result release
- No competitor identity disclosure
- No LLM-invented metrics
- Clear observed-versus-estimated labelling
- Version-bound human approval before simulated partner submission
- Evidence / confidence display
- Safe audit logging and eligibility checks without private-data leakage


### 5. Prototype Demo - 20 points
Standard:
Functional prototype that explains the end-to-end workflow and the output clearly.


Recommended demo path:
Diagnose -> Audience -> Recommend -> Activate -> Measure


The demo should prove the product works as one coherent workflow rather than a collection of disconnected screens.


## Build Priority by Score
1. Business Impact - 25
2. Role of AI - 20
3. Feasibility - 20
4. Prototype Demo - 20
5. Responsible AI Controls - 15


## Demo Goal
In 2-3 minutes, demonstrate enough of the product to visibly touch every scoring category, leaving time for the short presentation and Q&A.
