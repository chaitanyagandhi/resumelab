# ResumeLab

A local CLI research prototype that studies how AI resume-tailoring systems transform a
candidate's resume in response to a target job description.

Developed as part of USC research.

---

## Research disclaimer

**ResumeLab intentionally reproduces aggressive, job-description-conditioned resume
transformation.** Generated resumes may present technologies, metrics, architectures, and
project framings that do not appear in the source profile. That is the phenomenon under
study, not a defect.

This is **not** a resume verification tool, a factuality checker, or a system for
producing resumes to submit to employers. It is an instrument for measuring what
commercial AI tailoring systems do to a candidate's stated technical identity.

The source profile is never modified, so every run can be analyzed as:

```
ORIGINAL PROFILE   vs   TARGET JOB DESCRIPTION   vs   GENERATED RESUME
```

---

## Research objective

Commercial resume tailoring products appear to do far more than keyword matching. Given a
job description, they rewrite summaries, reframe experience, reposition projects,
introduce technologies associated with the target role, generate quantified impact, and
change the technical vocabulary throughout — in effect inferring the engineer the posting
is looking for and presenting the candidate as that engineer.

ResumeLab reproduces that behavior in a controlled setting so it can be measured. The
central mechanism is **`technical_identity`**: a description of the engineer that would
look maximally aligned with a posting, extracted during analysis. Every later stage
transforms the candidate toward it. The same profile against three different postings
should produce three recognisably different engineers.

Deliberately absent: there is no factuality guardrail. Constraining the model to claims
supported by the source profile would suppress exactly the behavior being studied.

---

## Setup

Requires Python 3.12+.

```bash
make install                                                       # venv + dependencies
cp .env.example .env                                               # add an API key
cp data/candidate_profile.example.yaml data/candidate_profile.yaml # add your profile
```

Both `.env` and `data/candidate_profile.yaml` are git-ignored; only their `.example`
counterparts are tracked. Populate the profile with real source material — the template
is intentionally unpopulated and will fail validation until you fill it in.

You need a key for **one** provider, not both.

---

## Usage

```bash
# Read a posting without generating anything
resumelab analyze --jd examples/sample_jd.txt

# Generate a tailored resume
resumelab generate --jd examples/sample_jd.txt

# Put the PDF somewhere convenient
resumelab generate --jd examples/sample_jd.txt --output output/crusoe_resume.pdf

# Supply the posting directly
resumelab generate --jd-text "Senior Go engineer, distributed storage..."

# Or paste the link to a posting and let ResumeLab read it
resumelab generate --jd-url https://job-boards.greenhouse.io/acme/jobs/8077887

# Force a provider for one run
resumelab generate --jd examples/sample_jd.txt --provider anthropic
```

Or via make:

```bash
make analyze  JD=examples/sample_jd.txt
make generate JD=examples/sample_jd.txt
```

| Flag | Applies to | Meaning |
|---|---|---|
| `--jd PATH` | both | Job description file. |
| `--jd-text TEXT` | both | Job description supplied directly. |
| `--jd-url URL` | both | Link to a posting, fetched and reduced to text. |
| `--provider` | both | `openai` or `anthropic`, overriding configuration. |
| `--output`, `-o` | both | `analyze`: write the analysis as JSON. `generate`: also write the PDF here. |
| `--debug` | both | Log at DEBUG and show tracebacks instead of one-line errors. |

Exactly one of `--jd`, `--jd-text`, and `--jd-url` may be given.

`analyze` exists because the analysis conditions everything downstream. When a generated
resume disappoints, reading the analysis is how you tell a bad *reading of the posting*
from a bad *plan*.

When both providers are configured and you are at a terminal, `generate` asks which to
use. Scripts and batch runs use the configured provider without prompting.

---

## Reading a posting from a URL

`--jd-url` turns a link into the same text you would otherwise have pasted. Where the
applicant tracking system publishes the posting as structured data, that is read
instead of the rendered page — the text arrives already free of navigation and
boilerplate, and the title, company, and location are fields rather than guesses.

| Source | How it is read |
|---|---|
| Greenhouse | Board API. Its `content` field is entity-escaped twice; that is undone. |
| Lever | Postings API, reassembled from the four fields it splits a posting across. |
| Ashby | Job-board API, filtered to the posting the URL names. |
| Workday | The `/wday/cxs/` JSON behind the page, which is otherwise JavaScript-rendered. |
| Anything else | A schema.org `JobPosting` block if the page has one, else its `<main>` text. |

The fetched posting is written to the run's `jd.txt`, so what was analyzed is always
recoverable — including when a fallback read carried a cookie banner along with it.

**What does not work.** Sites behind bot protection (Indeed, among others) refuse
automated requests and are reported as doing so. The user agent identifies ResumeLab
rather than imitating a browser, and working around bot detection is out of scope; use
`--jd-text` or `--jd` for those. A page whose posting is rendered entirely by
JavaScript and carries no structured block is reported as such rather than analyzed as
if the empty shell were a posting.

A fetched posting is untrusted in exactly the way a pasted one is: it is fenced as data
in every prompt, and scanned for instruction-like content, which is logged.

---

## Architecture

```
data/candidate_profile.yaml ─┐
                             ├─► JD analysis ─► transformation strategy ─┐
job description ─────────────┘                                           │
  (file, text, or URL)                                                   │
                                                                         ▼
                            summary ─ experience ─ projects ─ skills (per-section rewrite)
                                                                         │
                                                                         ▼
                                        assembly ─► validation ─► PDF ─► run artifacts
```

| Module | Responsibility |
|---|---|
| `models/` | Pydantic domain models: source profile, job description, analysis, strategy, generated resume, run metadata |
| `loaders/` | Reading and validating the profile and the posting |
| `fetching/` | Retrieving a posting from a URL: board adapters, then the page itself |
| `llm/` | Provider abstraction, OpenAI and Anthropic adapters, retry policy, versioned prompts |
| `pipeline/` | One module per stage, plus the generator that orders them |
| `validation/` | Deterministic pre-render checks over the finished resume |
| `rendering/` | ReportLab layout, with every measurement in `styles.py` |
| `experiment/` | Per-run directories and metadata |

Three structural rules hold throughout:

- **The model never produces the document.** It returns structured content; the renderer
  owns every layout decision. Any visual difference between two runs is a content
  difference.
- **Stages depend on an `LLMClient` protocol**, never a provider client, so a stage can be
  tested against a fake and a second provider is a new file rather than an edit.
- **Personal details never reach a model.** Name, email, phone, and profile links are
  excluded from every prompt and copied onto the resume at assembly.

---

## Pipeline stages

| # | Stage | What it does |
|---|---|---|
| 1 | Load candidate | Reads and validates the immutable source profile |
| 2 | Analyze JD | Extracts the role, its technology surface, and `technical_identity` |
| 3 | Build strategy | One global plan: target identity, per-entry directions, skills priority, tone |
| 4 | Generate summary | Executes the plan's `summary_direction` |
| 5 | Transform experience | Rewrites bullets per role; anchors are never generated |
| 6 | Transform projects | Regenerates subtitle, stack, and bullets; only the name is carried over |
| 7 | Transform skills | Rebuilds the section, choosing groupings for this role |
| 8 | Assemble & validate | Combines with identity and education, then checks it is fit to render |
| 9 | Render | ReportLab PDF, tightening the layout to fit one page where it can |

Stages 5–7 receive everything written before them, so sections do not converge on the
same verbs or restate each other's claims. The strategy exists so the finished resume
reads as one engineer rather than four independently-tailored fragments.

**Anchors versus framing.** Company, job title, dates, location, education, and project
names are copied from the source profile and are never generated — they are what makes a
generated resume comparable to its source. Everything else is open.

---

## Candidate profile

`data/candidate_profile.yaml`, validated on load. See
[`data/candidate_profile.example.yaml`](data/candidate_profile.example.yaml) for the full
template.

```yaml
personal:      {name, email, phone, linkedin, github, location}
education:     [{institution, degree, field, location, start_date, end_date, gpa, coursework}]
experiences:   [{company, title, location, start_date, end_date, description, bullets}]
projects:      [{name, subtitle, date, technologies, description, bullets}]
skills:        {programming_languages, frameworks, databases, cloud_devops, ai_ml, other}
achievements:  []
```

The research design fixes **exactly three projects with exactly three bullets each**;
those counts are `REQUIRED_PROJECT_COUNT` and `REQUIRED_PROJECT_BULLET_COUNT` in
`models/candidate.py`, so an ablation changes them in one place. Experience bullet counts
are not fixed at the source — record as much material as you have.

The file is read-only to the pipeline. A mistyped key is an error rather than a silently
dropped section.

---

## Configuration

All settings come from the environment or a local `.env`. See
[`.env.example`](.env.example) for the documented set.

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | inferred | `openai` or `anthropic`; inferred from whichever key is set |
| `LLM_MAX_RETRIES` | `3` | Retry budget, with exponential backoff |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-request timeout |
| `OPENAI_API_KEY` | — | Required for the OpenAI provider |
| `OPENAI_MODEL` | `gpt-4o` | |
| `OPENAI_TEMPERATURE` | `0.2` | Low by default so runs stay comparable |
| `ANTHROPIC_API_KEY` | — | Required for the Anthropic provider |
| `ANTHROPIC_MODEL` | `claude-opus-5` | |
| `ANTHROPIC_MAX_TOKENS` | `16000` | Covers reasoning as well as the response |
| `ANTHROPIC_EFFORT` | `high` | Current Claude models reject `temperature`; this is the equivalent dial |
| `SUMMARY_MAX_CHARACTERS` | `300` | Checked before rendering |
| `BULLET_MAX_CHARACTERS` | `220` | Checked before rendering |
| `EXPERIENCE_BULLET_COUNT` | `3` | Also enforced by the response schema — see below |
| `PROJECT_BULLET_COUNT` | `3` | Same |
| `CANDIDATE_PROFILE_PATH` | `data/candidate_profile.yaml` | |
| `OUTPUT_DIR` | `output` | Runs land in `<OUTPUT_DIR>/runs/` |
| `LOG_LEVEL` | `INFO` | |

Secrets are held as `SecretStr`, redacted in reprs and tracebacks, scrubbed from provider
error messages, and never written into a run directory.

---

## Experiment artifacts

Every generation writes a self-contained directory:

```
output/runs/2026-08-10T153000_crusoe/
├── jd.txt                        the posting exactly as analyzed
├── jd_analysis.json              structured reading of the posting
├── transformation_strategy.json  the plan every rewrite followed
├── generated_resume.json         the resume as structured data
├── metadata.json                 how the run was produced
└── resume.pdf
```

Intermediates are written as they are produced, so a run that fails at any stage still
leaves behind the reasoning that led there.

`metadata.json` records the provider, model, temperature or effort, both prompt versions,
the **SHA-256 of the source profile**, the job description's provenance — including the
URL, when it was fetched — call count, token usage, duration, and the layout outcome. The profile hash is what lets two runs be shown
to have shared an input.

Prompts are versioned in two families — `JD_ANALYSIS_PROMPT_VERSION` and
`TRANSFORMATION_PROMPT_VERSION` — because reading a posting and rewriting a candidate
change for different reasons. The transformation prompts are tuned as one set and move
together.

---

## Testing

```bash
make check        # ruff lint + format, mypy strict, pytest with coverage
pytest -m e2e     # the only tests that call a real provider, and spend money
```

Everything except the `e2e` test runs offline. Unit tests script a fake client; the
integration suite drives a **deterministic fake that reads its prompts** — it parses the
profile to learn which entries exist and draws vocabulary from the posting, so the
couplings between stages are genuinely exercised and two postings really do produce two
different resumes.

Rendered PDFs are verified by reading them back: text extraction must return the resume in
reading order, with every bullet intact.

---

## Limitations

- **Single candidate, single profile.** No upload, parsing, database, or multi-user
  support; the profile is a file you edit.
- **No evaluation metrics yet.** The artifacts support transformation-magnitude,
  keyword-coverage, and unsupported-claim measurement, but none of that is implemented.
- **No batch mode.** One posting per invocation.
- **Bullet counts are coupled.** `EXPERIENCE_BULLET_COUNT` and `PROJECT_BULLET_COUNT` are
  validated deterministically *and* enforced by the response schemas. Changing the
  environment variable alone fails the run loudly rather than being ignored; change the
  matching constant in `models/` too.
- **Right-aligned dates are not supported.** The only clean way to right-align them is a
  table, and tables damage text extraction, so dates sit inline.
- **One page is a target, not a guarantee.** The renderer tightens the layout within
  conservative limits and will condense once; content that still overflows is written as a
  readable two pages rather than shrunk into an unreadable one.
- **Posting retrieval does not defeat bot protection.** Sites that refuse automated
  requests are reported as refusing them; paste the posting instead.
- **Injection detection is a signal, not a defence.** The defence is the fencing and the
  system prompt; the detector only makes a suspicious posting visible in the log.
- **Model behavior is not reproducible across providers or model versions**, even at a
  fixed temperature. Metadata records what produced each run so results stay
  interpretable, not identical.

---

## Development

```bash
make help      # list targets
make format    # apply ruff formatting and fixes
make check     # lint, typecheck, test
```

Python 3.12+, full type hints, mypy strict, ruff, and pytest with 100% line and branch
coverage.
