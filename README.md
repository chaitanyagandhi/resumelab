# ResumeLab

A local CLI research prototype that studies how AI resume-tailoring systems transform a
candidate's resume in response to a target job description.

ResumeLab does not perform keyword matching. It infers the **technical identity implied by a
job description** and repositions a fixed candidate profile toward that identity — rewriting
summaries, experience bullets, project framing, and skills. The source candidate profile is
immutable, so every run can be compared as *original profile* vs *target JD* vs
*generated resume*.

> **Research disclaimer.** This system intentionally reproduces aggressive JD-conditioned
> transformation. Generated resumes may contain technologies, metrics, and project framings
> that are not present in the source profile. It is a research instrument, not a factual
> resume editor or verification tool.

Developed as part of USC research.

## Status

Under active construction, built in the numbered steps defined by the project's master build
prompt. Full documentation — architecture, setup, environment variables, candidate profile
format, CLI usage, pipeline stages, experiment artifacts, and limitations — is written in the
final documentation step.

## Development

```bash
make install   # create .venv and install the package with dev dependencies
make check     # ruff lint + format check, mypy strict, pytest with coverage
```

Run `make help` to list all targets.

## Providers

ResumeLab runs against **OpenAI** or **Anthropic**. The pipeline depends only on an
`LLMClient` protocol, so the provider is a configuration choice rather than a code
change, and both are recorded in each run's metadata for comparison.

Set `LLM_PROVIDER` to pick one explicitly; leave it blank and the provider is inferred
from whichever API key is configured. Note that current Claude models reject
`temperature`, so the Anthropic adapter uses `ANTHROPIC_EFFORT` as the equivalent
quality/cost dial.

## Local setup

Both files below hold private data and are git-ignored; only their `.example`
counterparts are tracked.

```bash
cp .env.example .env                                              # add your API key
cp data/candidate_profile.example.yaml data/candidate_profile.yaml # add your profile
```
