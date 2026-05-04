"""Script to run the task generation pipeline over a corpus of book chapters."""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from autogen_core.models import ChatCompletionClient
from dotenv import load_dotenv

from src.schemas.area_schemas import Area
from src.schemas.capability_schemas import Capability
from src.schemas.domain_schemas import Domain
from src.schemas.task_schemas import Task
from src.task_generation.agentic_pipeline import run_task_generation_loop
from src.task_generation.dedup_utils import (
    assign_chapter_level_task_ids,
    deduplicate_tasks_for_chapter,
    mark_discarded_metadata,
)
from src.task_generation.designer_agent import DesignerAgent
from src.task_generation.output_writer import (
    build_checkpoint_path,
    build_pipeline_metadata,
    build_task_output_path,
    save_task_outputs,
    write_dedup_report,
    write_json_artifact,
    write_token_stats,
    write_verification_stats,
)
from src.task_generation.verifier_agent import VerifierAgent
from src.utils.model_client_utils import get_standard_model_client


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT_DIR / "src" / "task_generation" / "logs"
logger = logging.getLogger(__name__)


@dataclass
class GenerationUnit:
    """One chapter-backed task generation unit."""

    capability: Capability
    context_text: str
    chapter_id: str
    chapter_relpath: str
    book_name: str


def configure_logging() -> None:
    """Configure logging for CLI runs without overriding existing app logging."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"task_gen_{run_id}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """
    Load YAML config from file, expanding environment variables.

    Args:
        path: Path to the YAML config file.

    Returns
    -------
        Parsed YAML content as a dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = os.path.expandvars(f.read())

    obj: Any = yaml.safe_load(content)
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(obj)} in {path}")
    return obj


def create_tag(dt: datetime) -> str:
    """
    Make a tag string from datetime in format: _YYYYMMDD_HHMMSS.

    Args:
        dt: The datetime object.

    Returns
    -------
        The formatted tag string.
    """
    return "_" + dt.strftime("%Y%m%d_%H%M%S")


def check_tag(s: str) -> bool:
    """
    Check if string looks like a tag in format: _YYYYMMDD_HHMMSS.

    Args:
        s: The string to check.

    Returns
    -------
        True if it looks like a tag, False otherwise.
    """
    if not isinstance(s, str) or len(s) != 16 or not s.startswith("_"):
        return False
    return s[1:9].isdigit() and s[10:16].isdigit() and s[9] == "_"


def create_diff_blueprint_combo(difficulty: str, blooms: str) -> str:
    """
    Create a combination slug from difficulty and blooms level.

    Args:
        difficulty: Difficulty string.
        blooms: Bloom's level string.

    Returns
    -------
        Slugified string.
    """

    def _primary_label(x: str) -> str:
        text = x.strip()
        if " - " in text:
            head = text.split(" - ", 1)[0].strip()
            if head:
                return head
        return text

    def _clean(x: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in x.strip()).strip("_")

    cleaned_difficulty = _clean(_primary_label(difficulty))
    cleaned_blooms = _clean(_primary_label(blooms))
    return f"{cleaned_difficulty}_{cleaned_blooms}"


def _stable_slug(value: str) -> str:
    """Create a compact slug for deterministic placeholder ids."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return cleaned or "unknown"


def _stable_short_hash(value: str, length: int = 8) -> str:
    """Create a deterministic short hash for collision-resistant ids."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def _book_name_from_relpath(chapter_relpath: str) -> str:
    """Return the top-level corpus folder from a chapter relative path."""
    parts = chapter_relpath.split("/")
    return parts[0] if len(parts) > 1 else "root_corpus"


def build_placeholder_capability(
    *,
    chapter_id: str,
    chapter_index: int,
    chapter_relpath: str,
    domain_name: str,
) -> Capability:
    """Build schema-valid placeholder domain, area, and capability objects."""
    slug = _stable_slug(chapter_id)
    short_hash = _stable_short_hash(chapter_relpath)

    domain = Domain(
        domain_name=domain_name or "unknown_domain",
        domain_id="domain_000",
        domain_description="Placeholder domain created by the standalone runner.",
    )
    area = Area(
        area_name=f"placeholder_area_{chapter_id}",
        area_id=f"area_ch_{slug}_{short_hash}",
        domain=domain,
        area_description="Placeholder area created from the source chapter.",
        generation_metadata={
            "lineage_mode": "chapter_placeholder",
            "source_chapter_id": chapter_id,
            "source_chapter_relpath": chapter_relpath,
            "source_chapter_index": chapter_index,
        },
    )
    return Capability(
        capability_name=f"placeholder_capability_{chapter_id}",
        capability_id=f"cap_ch_{slug}_{short_hash}",
        area=area,
        capability_description="Placeholder capability created from the source chapter.",
        generation_metadata={
            "lineage_mode": "chapter_placeholder",
            "source_chapter_id": chapter_id,
            "source_chapter_relpath": chapter_relpath,
            "source_chapter_index": chapter_index,
        },
    )


def prepare_generation_units(
    *,
    chapter_files: List[Path],
    chapter_root_dir: Path,
    blueprint_domain: str,
) -> List[GenerationUnit]:
    """Create one schema-compatible generation unit per chapter file."""
    generation_units: List[GenerationUnit] = []
    for chapter_idx, chapter_path in enumerate(chapter_files):
        chapter_relpath = chapter_path.relative_to(chapter_root_dir).as_posix()
        chapter_id = chapter_path.stem
        generation_units.append(
            GenerationUnit(
                capability=build_placeholder_capability(
                    chapter_id=chapter_id,
                    chapter_index=chapter_idx,
                    chapter_relpath=chapter_relpath,
                    domain_name=blueprint_domain,
                ),
                context_text=chapter_path.read_text(encoding="utf-8"),
                chapter_id=chapter_id,
                chapter_relpath=chapter_relpath,
                book_name=_book_name_from_relpath(chapter_relpath),
            )
        )
    return generation_units


def load_runner_configs(config_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load runner agent and pipeline configs."""
    agent_cfg = load_yaml_config(config_path / "agent_config.yaml")
    pipeline_cfg = load_yaml_config(config_path / "pipeline_config.yaml")
    return agent_cfg, pipeline_cfg


def load_blueprints(blueprints_path: Path) -> Tuple[List[Dict[str, Any]], str]:
    """Load blueprint combinations and domain from blueprint file."""
    if not blueprints_path.exists():
        raise FileNotFoundError(f"Blueprints JSON not found: {blueprints_path}")

    blueprints_obj: Dict[str, Any] = json.loads(
        blueprints_path.read_text(encoding="utf-8")
    )
    combinations = blueprints_obj.get("combinations", [])
    if not isinstance(combinations, list) or not combinations:
        raise ValueError(
            f"Blueprints JSON must contain a non-empty 'combinations' list: {blueprints_path}"
        )
    blueprint_domain = str(blueprints_obj.get("domain"))
    return combinations, blueprint_domain


def init_model_clients(
    agent_cfg: Dict[str, Any],
) -> Tuple[ChatCompletionClient, ChatCompletionClient]:
    """Initialize designer/verifier model clients."""
    designer_model_cfg = agent_cfg["agents"]["designer"]["model_config"]["config_list"][
        0
    ]
    verifier_model_cfg = agent_cfg["agents"]["verifier"]["model_config"]["config_list"][
        0
    ]
    designer_agent_cfg = agent_cfg["agents"]["designer"]["model_config"]
    verifier_agent_cfg = agent_cfg["agents"]["verifier"]["model_config"]

    designer_client = get_standard_model_client(
        designer_model_cfg["model"],
        seed=designer_agent_cfg.get("cache_seed"),
        temperature=designer_agent_cfg.get("temperature"),
        top_p=designer_agent_cfg.get("top_p"),
        timeout=designer_agent_cfg.get("timeout"),
        api_type=designer_model_cfg.get("api_type"),
        api_key=designer_model_cfg.get("api_key"),
        base_url=designer_model_cfg.get("base_url"),
    )

    verifier_client = get_standard_model_client(
        verifier_model_cfg["model"],
        seed=verifier_agent_cfg.get("cache_seed"),
        temperature=verifier_agent_cfg.get("temperature"),
        timeout=verifier_agent_cfg.get("timeout"),
        api_type=verifier_model_cfg.get("api_type"),
        api_key=verifier_model_cfg.get("api_key"),
        base_url=verifier_model_cfg.get("base_url"),
    )

    logger.info(f"Designer Agent loaded with config: {designer_model_cfg}")
    logger.info(f"Verifier Agent loaded with config: {verifier_model_cfg}")
    return designer_client, verifier_client


def shard_generation_units(
    generation_units: List[GenerationUnit],
    *,
    worker_index: Optional[int],
    worker_count: Optional[int],
) -> List[GenerationUnit]:
    """Select the generation units owned by this worker shard."""
    if worker_index is None and worker_count is None:
        return generation_units
    if worker_index is None or worker_count is None:
        raise ValueError("worker_index and worker_count must be provided together.")
    if worker_count < 1:
        raise ValueError(f"worker_count must be >= 1. Got: {worker_count}")
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError(
            f"worker_index must be in [0, worker_count). Got worker_index={worker_index}, "
            f"worker_count={worker_count}"
        )
    return [
        unit
        for idx, unit in enumerate(generation_units)
        if idx % worker_count == worker_index
    ]


async def run_pipeline(
    *,
    experiment_id_override: Optional[str] = None,
    output_base_dir_override: Optional[Path] = None,
    tasks_tag_override: Optional[str] = None,
    worker_index: Optional[int] = None,
    worker_count: Optional[int] = None,
) -> str:
    """Run task generation pipeline and return the tasks tag."""
    configure_logging()
    config_path = ROOT_DIR / "src" / "cfg" / "task_generation"
    agent_cfg, pipeline_cfg = load_runner_configs(config_path)

    # ---- required pipeline config ----
    experiment_id = experiment_id_override or pipeline_cfg["pipeline"]["experiment_id"]
    output_base_dir = output_base_dir_override or Path(
        pipeline_cfg["pipeline"]["output_base_dir"]
    )

    # ---- chapter corpus root ----
    book_chapter_dir = pipeline_cfg["pipeline"].get(
        "book_chapter_dir", "book_chapter_text_files"
    )
    chapter_root_dir = (
        ROOT_DIR / "src" / "task_generation" / book_chapter_dir
    ).resolve()
    if not chapter_root_dir.exists():
        raise FileNotFoundError(f"book_chapter_dir not found: {chapter_root_dir}")

    # ---- blueprints ----
    blueprints_file = pipeline_cfg["pipeline"].get(
        "blueprints_file", "finance_blueprints.json"
    )
    blueprints_path = (
        ROOT_DIR / "src" / "task_generation" / "blueprints" / blueprints_file
    ).resolve()
    combinations, blueprint_domain = load_blueprints(blueprints_path)
    logger.info(f"Using blueprints file: {blueprints_path}")

    # ---- runtime params ----
    max_retries = int(pipeline_cfg["pipeline"].get("max_retries", 3))
    default_num_tasks_per_combo = int(
        pipeline_cfg["pipeline"].get("num_tasks_per_combo", 5)
    )
    configured_num_tasks = pipeline_cfg["pipeline"].get("num_tasks")
    global_seed_num_tasks_override = (
        int(configured_num_tasks) if configured_num_tasks is not None else None
    )
    checkpoint_cfg = pipeline_cfg["pipeline"].get("checkpoint", {}) or {}
    checkpoint_enabled = bool(checkpoint_cfg.get("enabled", True))
    checkpoint_resume = bool(checkpoint_cfg.get("resume_from_checkpoint", True))
    checkpoint_every = int(checkpoint_cfg.get("every", 2))
    checkpoint_dir_name = str(checkpoint_cfg.get("dir_name", "checkpoints")).strip()
    checkpoint_file_name = str(
        checkpoint_cfg.get("file_name", "passed_tasks_checkpoint.json")
    ).strip()
    checkpoint_every = max(checkpoint_every, 0)

    is_resume = tasks_tag_override is not None
    out_tag = tasks_tag_override or create_tag(datetime.now())
    designer_client, verifier_client = init_model_clients(agent_cfg)

    try:
        # ---- discover chapters on disk ----
        chapter_files = sorted(chapter_root_dir.rglob("*.txt"))
        if not chapter_files:
            logger.error(f"No chapter .txt files found under: {chapter_root_dir}")
            return out_tag

        logger.info(
            f"Found {len(chapter_files)} chapter files under {chapter_root_dir}"
        )
        logger.info(f"Found {len(combinations)} blueprint combinations")
        logger.info("Capability source mode: chapter-derived placeholders")
        if blueprint_domain:
            logger.info(f"Blueprint domain (informational): {blueprint_domain}")

        def make_designer_agent() -> DesignerAgent:
            return DesignerAgent(name="Designer", model_client=designer_client)

        def make_verifier_agent() -> VerifierAgent:
            return VerifierAgent(name="Verifier", model_client=verifier_client)

        generation_units: List[GenerationUnit] = prepare_generation_units(
            chapter_files=chapter_files,
            chapter_root_dir=chapter_root_dir,
            blueprint_domain=blueprint_domain,
        )
        total_generation_units = len(generation_units)
        generation_units = shard_generation_units(
            generation_units,
            worker_index=worker_index,
            worker_count=worker_count,
        )

        if worker_index is not None and worker_count is not None:
            logger.info(
                "Prepared %s generation unit(s); worker %s/%s owns %s unit(s).",
                total_generation_units,
                worker_index,
                worker_count,
                len(generation_units),
            )
        else:
            logger.info("Prepared %s generation unit(s).", len(generation_units))

        for unit_idx, unit in enumerate(generation_units):
            current_capability = unit.capability
            context_text = unit.context_text
            chapter_id = unit.chapter_id
            chapter_relpath = unit.chapter_relpath
            book_name = unit.book_name

            # compatible path: tasks/<tag>/<area_id>/<capability_id>/tasks.json
            chapter_out_path = build_task_output_path(
                output_base_dir=output_base_dir,
                experiment_id=experiment_id,
                out_tag=out_tag,
                capability=current_capability,
            )
            chapter_out_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint_metadata = build_pipeline_metadata(
                experiment_id=experiment_id,
                output_base_dir=output_base_dir,
                input_stage_tag="chapter_placeholders",
                output_stage_tag=out_tag,
                resume=is_resume,
            )

            if is_resume and chapter_out_path.exists():
                logger.info(
                    f"Skipping (resume) {current_capability.area.area_id}/"
                    f"{current_capability.capability_id} ({book_name}/{chapter_id}) "
                    "because tasks.json exists."
                )
                continue

            logger.info(
                f"Generation unit {unit_idx + 1}/{len(generation_units)}: {chapter_relpath}"
            )

            all_tasks: List[Task] = []
            chapter_q_counter = 0
            chapter_verification_logs: List[Dict[str, Any]] = []
            chapter_token_usage_logs: List[Dict[str, Any]] = []
            chapter_summary_path = chapter_out_path.parent / "chapter_summary.json"

            # ---- chapter knowledge extraction (ONCE per chapter) ----
            summary_usage_record: Dict[str, Any]
            if is_resume and chapter_summary_path.exists():
                summary_payload = json.loads(
                    chapter_summary_path.read_text(encoding="utf-8")
                )
                chapter_knowledge_obj = summary_payload.get("chapter_knowledge_obj")
                summary_usage_record = dict(
                    summary_payload.get("summary_usage_record") or {}
                )
            else:
                designer = make_designer_agent()
                (
                    chapter_knowledge_obj,
                    _chapter_knowledge_prompt,
                    chapter_summary_usage,
                ) = await designer.summarize_chapter_knowledge(
                    chapter_excerpts=context_text,
                )
                summary_usage_record = {
                    "stage": "summarize_chapter_knowledge",
                    "model_role": "designer",
                    "chapter_id": chapter_id,
                    "chapter_relpath": chapter_relpath,
                    "blueprint_key": None,
                    "difficulty": "",
                    "blooms_level": "",
                    "seed_generation_index": None,
                    "candidate_label": None,
                    "candidate_index_within_seed": None,
                    "attempt_index": None,
                    "input_tokens": chapter_summary_usage.get("input_tokens"),
                    "output_tokens": chapter_summary_usage.get("output_tokens"),
                    "total_tokens": chapter_summary_usage.get("total_tokens"),
                    "usage_available": bool(
                        chapter_summary_usage.get("usage_available", False)
                    ),
                }
                write_json_artifact(
                    chapter_summary_path,
                    {
                        "chapter_id": chapter_id,
                        "chapter_relpath": chapter_relpath,
                        "book_name": book_name,
                        "chapter_knowledge_obj": chapter_knowledge_obj,
                        "summary_usage_record": summary_usage_record,
                    },
                )
            chapter_token_usage_logs.append(summary_usage_record)
            if not isinstance(chapter_knowledge_obj, dict):
                logger.warning(
                    "Chapter knowledge summary not dict; using raw text block."
                )
            chapter_knowledge_text = (
                json.dumps(chapter_knowledge_obj, indent=2, ensure_ascii=False)
                if isinstance(chapter_knowledge_obj, dict)
                else str(chapter_knowledge_obj)
            )
            previous_questions_by_combo: Dict[str, List[str]] = {}

            for combination in combinations:
                difficulty = str(combination.get("difficulty", "")).strip()
                blooms_level = str(combination.get("blooms_level", "")).strip()
                diff_bpt_combo = create_diff_blueprint_combo(difficulty, blooms_level)
                seed_num_tasks = int(
                    global_seed_num_tasks_override
                    if global_seed_num_tasks_override is not None
                    else combination.get("num_tasks", default_num_tasks_per_combo)
                )
                combo_checkpoint_path = build_checkpoint_path(
                    chapter_out_path=chapter_out_path,
                    checkpoint_enabled=checkpoint_enabled,
                    checkpoint_dir_name=checkpoint_dir_name,
                    checkpoint_file_name=f"{diff_bpt_combo}_{checkpoint_file_name}",
                )
                combo_previous_questions = previous_questions_by_combo.setdefault(
                    diff_bpt_combo, []
                )
                combo_verification_logs: List[Dict[str, Any]] = []
                combo_token_usage_logs: List[Dict[str, Any]] = []

                logger.info(
                    "Generating tasks for %s | combo=%s with seed_tasks=%s",
                    chapter_id,
                    diff_bpt_combo,
                    seed_num_tasks,
                )

                tasks: Optional[List[Task]] = await run_task_generation_loop(
                    designer_factory=make_designer_agent,
                    verifier_factory=make_verifier_agent,
                    capability=current_capability,
                    capability_source_mode="chapter_placeholder",
                    domain=blueprint_domain,
                    context_text=context_text,
                    chapter_knowledge_text=chapter_knowledge_text,
                    max_retries=max_retries,
                    difficulty=difficulty,
                    blooms_level=blooms_level,
                    num_tasks=seed_num_tasks,
                    seed_generation_target=seed_num_tasks,
                    chapter_id=chapter_id,
                    chapter_relpath=chapter_relpath,
                    blueprint_key=diff_bpt_combo,
                    chapter_q_start=chapter_q_counter,
                    verification_log=combo_verification_logs,
                    token_usage_log=combo_token_usage_logs,
                    previous_questions=combo_previous_questions,
                    checkpoint_path=combo_checkpoint_path,
                    checkpoint_every=checkpoint_every,
                    checkpoint_metadata=checkpoint_metadata,
                    resume_from_checkpoint=checkpoint_enabled and checkpoint_resume,
                )
                chapter_verification_logs.extend(combo_verification_logs)
                chapter_token_usage_logs.extend(combo_token_usage_logs)

                if not tasks:
                    logger.error(
                        "Failed to generate tasks for %s | %s",
                        chapter_id,
                        diff_bpt_combo,
                    )
                    continue

                chapter_q_counter += len(tasks)
                all_tasks.extend(tasks)
                logger.info(
                    "Accumulated %s total tasks so far for %s after combo=%s",
                    len(all_tasks),
                    chapter_id,
                    diff_bpt_combo,
                )

            if not all_tasks:
                logger.error(
                    f"No tasks generated for chapter {book_name}/{chapter_id}; nothing to save."
                )
                continue

            # ---- deduplication ----
            dedup_cfg = agent_cfg.get("dedup", {}) or {}
            if bool(dedup_cfg.get("enabled", False)):
                threshold = float(dedup_cfg.get("threshold", 0.90))
                embedding_model = str(
                    dedup_cfg.get("embedding_model", "text-embedding-3-small")
                )
                keep_policy = str(dedup_cfg.get("keep_policy", "first"))
                cache_embeddings = bool(dedup_cfg.get("cache_embeddings", True))
                save_discarded = bool(dedup_cfg.get("save_discarded", True))

                cache_path = None
                if cache_embeddings:
                    cache_path = chapter_out_path.parent / "embedding_cache.json"

                kept, discarded, report = deduplicate_tasks_for_chapter(
                    tasks=all_tasks,
                    chapter_id=current_capability.capability_id,
                    embedding_model=embedding_model,
                    threshold=threshold,
                    keep_policy=keep_policy,
                    cache_path=cache_path,
                )

                kept = assign_chapter_level_task_ids(
                    kept_tasks=kept,
                    chapter_id=current_capability.capability_id,
                )

                report_path = write_dedup_report(
                    chapter_out_path=chapter_out_path,
                    report=report,
                )
                logger.info(
                    f"Dedup: kept {len(kept)}/{len(all_tasks)} tasks. Report → {report_path}"
                )

                if save_discarded and discarded:
                    discarded = mark_discarded_metadata(
                        discarded_tasks=discarded,
                        chapter_id=current_capability.capability_id,
                        dedup_report=report,
                    )
                    discarded_tasks_to_save = discarded
                else:
                    discarded_tasks_to_save = None

                all_tasks = kept
            else:
                discarded_tasks_to_save = None

            metadata = build_pipeline_metadata(
                experiment_id=experiment_id,
                output_base_dir=output_base_dir,
                input_stage_tag="chapter_placeholders",
                output_stage_tag=out_tag,
                resume=is_resume,
            )

            stats_path = write_verification_stats(
                chapter_out_path=chapter_out_path,
                chapter_id=chapter_id,
                chapter_relpath=chapter_relpath,
                book_name=book_name,
                capability_id=current_capability.capability_id,
                area_id=current_capability.area.area_id,
                verification_logs=chapter_verification_logs,
            )
            logger.info(f"Saved verification stats → {stats_path}")

            token_stats_path = write_token_stats(
                chapter_out_path=chapter_out_path,
                chapter_id=chapter_id,
                chapter_relpath=chapter_relpath,
                book_name=book_name,
                capability_id=current_capability.capability_id,
                area_id=current_capability.area.area_id,
                token_usage_logs=chapter_token_usage_logs,
            )
            logger.info(f"Saved token stats → {token_stats_path}")

            saved_path, saved_discarded_path = save_task_outputs(
                tasks=all_tasks,
                discarded_tasks=discarded_tasks_to_save,
                metadata=metadata,
                chapter_out_path=chapter_out_path,
            )
            logger.info(f"Saved {len(all_tasks)} tasks → {saved_path}")

            if saved_discarded_path is not None:
                logger.info(
                    f"Saved {len(discarded_tasks_to_save or [])} discarded tasks → {saved_discarded_path}"
                )
    finally:
        await designer_client.close()
        await verifier_client.close()
    logger.info("Done.")
    return out_tag


def run_from_stage3(
    *,
    experiment_id: str,
    output_base_dir: Path,
    tasks_tag: Optional[str] = None,
    worker_index: Optional[int] = None,
    worker_count: Optional[int] = None,
) -> str:
    """Run standalone task generation and return tasks_tag."""
    return asyncio.run(
        run_pipeline(
            experiment_id_override=experiment_id,
            output_base_dir_override=output_base_dir,
            tasks_tag_override=tasks_tag,
            worker_index=worker_index,
            worker_count=worker_count,
        )
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI args for direct runner execution."""
    parser = argparse.ArgumentParser(description="Run agentic task generation.")
    parser.add_argument(
        "--tasks-tag",
        dest="tasks_tag",
        help="Existing or desired task tag. Provide this to resume or to coordinate parallel workers.",
    )
    parser.add_argument(
        "--worker-index",
        dest="worker_index",
        type=int,
        help="Zero-based shard index for chapter-level parallel workers.",
    )
    parser.add_argument(
        "--worker-count",
        dest="worker_count",
        type=int,
        help="Total number of chapter-level parallel workers.",
    )
    return parser.parse_args()


async def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    await run_pipeline(
        tasks_tag_override=args.tasks_tag,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
    )


if __name__ == "__main__":
    asyncio.run(main())
