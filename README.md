# FLAME Framework

We introduce **F**ine-grained, **LA**rge-scale **M**odel **E**valuation (FLAME), a framework for automated, comprehensive benchmark generation grounded in external knowledge sources such as textbooks and technical references

![ACE pipeline overview](src/assets/ACE_pipeline.png)

## Repository Layout

```text
src/
  cfg/task_generation/
    agent_config.yaml       # Model/provider and deduplication settings
    pipeline_config.yaml    # Corpus, output, resume, and loop settings
  schemas/                  # Lightweight task-generation schemas and I/O
  task_generation/
    runner.py               # Main CLI entrypoint
    agentic_pipeline.py     # Generation, repair, verification loop
    designer_agent.py       # Designer model wrapper
    verifier_agent.py       # Verifier model wrapper
    prompts.py              # Prompt templates
    blueprints/
      blueprints.json       # Difficulty and Bloom's-level combinations
    sample_chapter_text_files/
      *.txt                 # Example chapter corpus
  utils/
    model_client_utils.py   # Model client construction and retry helpers
```

## Requirements

- Python 3.10 or newer
- Runtime dependencies from `pyproject.toml`
- API credentials for the providers configured in
  `src/cfg/task_generation/agent_config.yaml`

Common environment variables are:

```text
GOOGLE_API_KEY
ANTHROPIC_API_KEY
OPENAI_API_KEY
```

`OPENAI_API_KEY` is only required when deduplication is enabled.

## Configuration

Pipeline behavior is controlled by:

```text
src/cfg/task_generation/pipeline_config.yaml
```

Important fields:

- `experiment_id`: output experiment folder name
- `output_base_dir`: root directory for generated artifacts
- `book_chapter_dir`: chapter corpus directory under `src/task_generation/`
- `blueprints_file`: blueprint JSON file under `src/task_generation/blueprints/`
- `max_retries`: retry budget for failed candidates
- `num_tasks_per_combo`: fallback task count when a blueprint omits `num_tasks`
- `checkpoint`: incremental save and resume settings

Model behavior is controlled by:

```text
src/cfg/task_generation/agent_config.yaml
```

This file configures the designer model, verifier model, and optional
embedding-based deduplication.

Task difficulty and Bloom's-level coverage are controlled by:

```text
src/task_generation/blueprints/blueprints.json
```

Each blueprint combination can define:

- `difficulty`
- `blooms_level`
- `num_tasks`

## Run

From the repository root:

```bash
python3 -m src.task_generation.runner
```

The runner creates a fresh task tag automatically.

## Resume Or Reuse A Task Tag

Pass an existing or desired task generation task tag:

```bash
python3 -m src.task_generation.runner --tasks-tag _YYYYMMDD_HHMMSS
```

When resuming, the runner skips generation units whose `tasks.json` output
already exists and can also resume from per-combination checkpoints when
checkpointing is enabled.

## Parallel Run

Use the helper script:

```bash
bash src/task_generation/run_parallel_task_gen.sh ''
```

The first argument is an optional task tag. Leave it empty to create a fresh tag.

You can control the number of workers with `WORKER_COUNT`:

```bash
WORKER_COUNT=4 bash src/task_generation/run_parallel_task_gen.sh ''
```

Equivalent direct worker commands:

```bash
python3 -m src.task_generation.runner --tasks-tag _YYYYMMDD_HHMMSS --worker-index 0 --worker-count 4
python3 -m src.task_generation.runner --tasks-tag _YYYYMMDD_HHMMSS --worker-index 1 --worker-count 4
python3 -m src.task_generation.runner --tasks-tag _YYYYMMDD_HHMMSS --worker-index 2 --worker-count 4
python3 -m src.task_generation.runner --tasks-tag _YYYYMMDD_HHMMSS --worker-index 3 --worker-count 4
```

## Outputs

Generated tasks are written to:

```text
<output_base_dir>/<experiment_id>/tasks/<tasks_tag>/<area_id>/<capability_id>/tasks.json
```

Each generation-unit directory may also contain:

```text
chapter_summary.json
verification_stats.json
token_stats.json
dedup_report.json
discarded_tasks.json
embedding_cache.json
checkpoints/
```

## Placeholder Metadata

The runner creates schema-compatible placeholder metadata directly from each
chapter file. For every chapter, it builds:

- a `Domain` from the blueprint domain,
- an `Area` with a deterministic chapter-derived `area_id`,
- a `Capability` with a deterministic chapter-derived `capability_id`.

These placeholder objects satisfy the same schema chain used by generated task
artifacts:

```text
Domain -> Area -> Capability -> Task
```

Later on, the capability and areas can be included manually.

## Notes

- Keep source text files under the configured `book_chapter_dir`.
- Keep blueprint counts small for smoke tests.
- Disable `dedup.enabled` in `agent_config.yaml` if you want to test generation
  without embedding calls.
