"""
运行 Bidirectional Feedback EDC (BF-EDC) 框架

用法:
python run_bf_edc.py --dataset webnlg --output_dir output/webnlg_bf_edc
python run_bf_edc.py --dataset webnlg --log_file_path logs/webnlg_bf_edc.log ...
"""

import argparse
import json
import logging
import os
import pathlib
from contextlib import redirect_stderr, redirect_stdout

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from edc.bidirectional_feedback_edc import BidirectionalFeedbackEDC
import edc.utils.llm_utils as llm_utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def run_bf_edc(args):
    """Run BF-EDC extraction with parsed CLI arguments."""
    # 根据消融模式调整开关
    if args.ablation_mode == "baseline":
        args.enable_ca_oie = False
        args.enable_sd_ac = False
        args.enable_bi_sc = False
        args.enable_gca = False
    elif args.ablation_mode == "ca_oie_only":
        args.enable_ca_oie = True
        args.enable_sd_ac = False
        args.enable_bi_sc = False
        args.enable_gca = False
    elif args.ablation_mode == "sd_ac_only":
        args.enable_ca_oie = True
        args.enable_sd_ac = True
        args.enable_bi_sc = False
        args.enable_gca = False
    elif args.ablation_mode == "bi_sc_only":
        args.enable_ca_oie = True
        args.enable_sd_ac = True
        args.enable_bi_sc = True
        args.enable_gca = False
    elif args.ablation_mode == "gca_only":
        args.enable_ca_oie = True
        args.enable_sd_ac = True
        args.enable_bi_sc = True
        args.enable_gca = True
        args.max_feedback_iterations = 1  # 只GCA，不需要多次反馈

    # Explicit leave-one-out switches are applied after the legacy additive
    # ablation modes so that `--ablation_mode full --disable_*` is unambiguous.
    if args.disable_ca_oie:
        args.enable_ca_oie = False
    if args.disable_sd_ac:
        args.enable_sd_ac = False
    if args.disable_bi_sc:
        args.enable_bi_sc = False
    if args.disable_gca:
        args.enable_gca = False
    if args.disable_relation_classifier:
        args.enable_relation_classifier = False
    if args.disable_relation_reranker:
        args.enable_relation_reranker = False
    # full模式使用默认参数

    logger.info("=" * 70)
    logger.info("BF-EDC Configuration")
    logger.info("=" * 70)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Cheap Model: {args.cheap_model}")
    logger.info(f"Mid Model: {args.mid_model}")
    logger.info(f"Ablation Mode: {args.ablation_mode}")
    logger.info(f"CA-OIE: {args.enable_ca_oie}")
    logger.info(f"SD-AC: {args.enable_sd_ac}")
    logger.info(f"Bi-SC: {args.enable_bi_sc}")
    logger.info(f"GCA: {args.enable_gca}")
    logger.info(f"Relation Classifier: {args.enable_relation_classifier}")
    if args.enable_relation_classifier:
        logger.info(f"  Model: {args.relation_classifier_model}")
        logger.info(f"  Threshold: {args.relation_classifier_threshold}")
    logger.info(f"Relation Reranker: {args.enable_relation_reranker}")
    if args.enable_relation_reranker:
        logger.info(f"  Model: {args.relation_classifier_model}")
        logger.info(f"  Top-K: {args.relation_reranker_top_k}")
        logger.info(f"  Min prob: {args.relation_reranker_min_prob}")
        logger.info(f"  Alias min prob: {args.relation_reranker_alias_min_prob}")
        logger.info(f"  Margin: {args.relation_reranker_margin}")
    logger.info(f"Entity Aligner: {args.enable_entity_aligner}")
    if args.enable_entity_aligner:
        logger.info(f"  Model: {args.entity_aligner_model}")
        logger.info(f"  Threshold: {args.entity_aligner_threshold}")
        logger.info(f"  Max candidates: {args.entity_aligner_max_candidates}")
        logger.info(f"  Data dir: {args.entity_aligner_data_dir}")
    logger.info(f"Output Postprocess: {args.enable_output_postprocess}")
    if args.enable_output_postprocess:
        logger.info(f"  Profile: {args.output_postprocess_profile}")
        logger.info(f"  Normalize objects: {not args.disable_output_object_normalization}")
        logger.info(f"  Repair relations: {not args.disable_output_relation_postprocess}")
        logger.info(f"  Dedupe: {not args.disable_output_dedupe}")
        logger.info(f"  Rule set: {args.output_postprocess_rule_set}")
    logger.info(f"Max Feedback Iterations: {args.max_feedback_iterations}")
    logger.info(f"OIE Batch Size: {args.oie_batch_size}")
    logger.info(f"OIE Prompt Template: {args.oie_prompt_template}")
    logger.info("=" * 70)

    # 加载数据
    dataset_path = args.dataset_file or f"datasets/{args.dataset}.txt"
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return

    with open(dataset_path, 'r', encoding='utf-8') as f:
        input_texts = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(input_texts)} texts from {dataset_path}")

    schema_path = args.schema_file or f"schemas/{args.dataset}_schema.csv"
    few_shot_dir = args.few_shot_dir or f"few_shot_examples/{args.dataset}"
    reference_path = args.reference_file or f"evaluate/references/{args.dataset}.txt"

    required_files = [
        schema_path,
        args.oie_prompt_template,
        os.path.join(few_shot_dir, "oie_few_shot_examples.txt"),
        os.path.join(few_shot_dir, "sd_few_shot_examples.txt"),
        os.path.join(few_shot_dir, "oie_few_shot_refine_examples.txt"),
        os.path.join(few_shot_dir, "ee_few_shot_examples.txt"),
    ]
    missing_files = [path for path in required_files if not os.path.isfile(path)]
    if missing_files:
        logger.error("Required dataset resources are missing: %s", missing_files)
        return

    logger.info(f"Schema: {schema_path}")
    logger.info(f"Few-shot directory: {few_shot_dir}")
    logger.info(f"Reference: {reference_path}")

    # 加载模型配置
    model_config_path = "model_api_config.json"
    if not os.path.exists(model_config_path):
        logger.error(f"Model config not found: {model_config_path}")
        return

    # 设置模型注册表
    llm_utils.set_model_registry(model_config_path)

    # 构建EDC配置
    edc_config = {
        # OIE配置
        "oie_llm": args.cheap_model,
        "oie_prompt_template_file_path": args.oie_prompt_template,
        "oie_few_shot_example_file_path": os.path.join(few_shot_dir, "oie_few_shot_examples.txt"),

        # SD配置
        "sd_llm": args.mid_model,
        "sd_prompt_template_file_path": "prompt_templates/sd_template.txt",
        "sd_few_shot_example_file_path": os.path.join(few_shot_dir, "sd_few_shot_examples.txt"),

        # SC配置
        "sc_llm": args.mid_model,
        "sc_embedder": args.sc_embedder,
        "sc_prompt_template_file_path": "prompt_templates/sc_template.txt",

        # Schema配置
        "target_schema_path": schema_path,
        "enrich_schema": False,

        # Refinement配置（不使用）
        "sr_adapter_path": None,
        "sr_embedder": "sentence-transformers/all-MiniLM-L6-v2",
        "oie_refine_prompt_template_file_path": "prompt_templates/oie_refine_template.txt",
        "oie_refine_few_shot_example_file_path": os.path.join(few_shot_dir, "oie_few_shot_refine_examples.txt"),
        "ee_llm": args.mid_model,
        "ee_prompt_template_file_path": "prompt_templates/ee_template.txt",
        "ee_few_shot_example_file_path": os.path.join(few_shot_dir, "ee_few_shot_examples.txt"),
        "em_prompt_template_file_path": "prompt_templates/em_template.txt",

        # BF-EDC特定配置
        "enable_bf_edc": True,
        "enable_ca_oie": args.enable_ca_oie,
        "enable_sd_ac": args.enable_sd_ac,
        "enable_bi_sc": args.enable_bi_sc,
        "enable_gca": args.enable_gca,
        "max_feedback_iterations": args.max_feedback_iterations,
        "oie_batch_size": args.oie_batch_size,

        # 成本优化配置
        "enable_cost_aware": True,
        "cheap_model": args.cheap_model,
        "mid_model": args.mid_model,
        
        # 关系分类器配置（新增）
        "enable_relation_classifier": args.enable_relation_classifier,
        "relation_classifier_model": args.relation_classifier_model,
        "relation_classifier_threshold": args.relation_classifier_threshold,
        "enable_relation_reranker": args.enable_relation_reranker,
        "relation_reranker_top_k": args.relation_reranker_top_k,
        "relation_reranker_min_prob": args.relation_reranker_min_prob,
        "relation_reranker_alias_min_prob": args.relation_reranker_alias_min_prob,
        "relation_reranker_margin": args.relation_reranker_margin,
        "relation_reranker_mode": args.relation_reranker_mode,

        # Optional neural entity aligner inside GCA/PostProcessor.
        "enable_entity_aligner": args.enable_entity_aligner,
        "entity_aligner_model": args.entity_aligner_model,
        "entity_aligner_threshold": args.entity_aligner_threshold,
        "entity_aligner_max_candidates": args.entity_aligner_max_candidates,
        "entity_aligner_data_dir": args.entity_aligner_data_dir,

        # Final output post-processing. When enabled, canon_kg.raw.txt keeps
        # the pre-postprocess output and canon_kg.txt contains the final output.
        "enable_output_postprocess": args.enable_output_postprocess,
        "output_postprocess_normalize_objects": not args.disable_output_object_normalization,
        "output_postprocess_repair_relations": not args.disable_output_relation_postprocess,
        "output_postprocess_dedupe": not args.disable_output_dedupe,
        "output_postprocess_rule_set": args.output_postprocess_rule_set,
        "output_postprocess_profile": args.output_postprocess_profile,

        # 日志配置
        "loglevel": logging.INFO
    }

    # 初始化BF-EDC
    logger.info("Initializing BF-EDC framework...")
    bf_edc = BidirectionalFeedbackEDC(**edc_config)

    # 运行抽取
    logger.info("Starting extraction...")
    results = bf_edc.extract_kg(
        input_text_list=input_texts,
        output_dir=args.output_dir
    )

    logger.info(f"\n{'=' * 70}")
    logger.info("Extraction completed!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Total triples extracted: {sum(len(r) for r in results)}")
    logger.info(f"{'=' * 70}")

    # 打印评估命令
    logger.info("\nTo evaluate the results, run:")
    logger.info("python evaluate/evaluation_script.py \\")
    logger.info(f"  --edc_output {args.output_dir}/iter0/canon_kg.txt \\")
    logger.info(f"  --reference {reference_path}")


def main():
    parser = argparse.ArgumentParser(description="Run BF-EDC Framework")

    # 数据集参数
    parser.add_argument("--dataset", type=str, default="webnlg",
                        help="Dataset name (webnlg, webnlg_10, webnlg_50, etc.)")
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=None,
        help="Explicit dataset text file; defaults to datasets/{dataset}.txt",
    )
    parser.add_argument(
        "--schema_file",
        type=str,
        default=None,
        help="Explicit schema CSV; defaults to schemas/{dataset}_schema.csv",
    )
    parser.add_argument(
        "--few_shot_dir",
        type=str,
        default=None,
        help="Explicit few-shot directory; defaults to few_shot_examples/{dataset}",
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        default=None,
        help="Reference path printed in the evaluation command",
    )
    parser.add_argument("--output_dir", type=str, default="output/webnlg_bf_edc",
                        help="Output directory")
    parser.add_argument(
        "--oie_prompt_template",
        type=str,
        default="prompt_templates/oie_template.txt",
        help="Explicit OIE prompt template path",
    )

    # 模型参数
    parser.add_argument("--cheap_model", type=str, default="ollama_qwen32b",
                        help="Cheap model for initial extraction")
    parser.add_argument("--mid_model", type=str, default="ollama_qwen72b",
                        help="Mid-tier model for refinement")
    parser.add_argument("--sc_embedder", type=str, default="e5_mistral_7b_instruct",
                        help="Embedder alias or model path for schema canonicalization")

    # BF-EDC开关
    parser.add_argument("--enable_ca_oie", action="store_true", default=True,
                        help="Enable Consistency-Aware OIE")
    parser.add_argument("--disable_ca_oie", action="store_true", default=False,
                        help="Disable CA-OIE for a leave-one-out ablation")
    parser.add_argument("--enable_sd_ac", action="store_true", default=True,
                        help="Enable Schema Definition with Alignment")
    parser.add_argument("--disable_sd_ac", action="store_true", default=False,
                        help="Disable SD-AC for a leave-one-out ablation")
    parser.add_argument("--enable_bi_sc", action="store_true", default=True,
                        help="Enable Bidirectional Schema Canonicalization")
    parser.add_argument("--disable_bi_sc", action="store_true", default=False,
                        help="Disable Bi-SC while retaining the original SC")
    parser.add_argument("--enable_gca", action="store_true", default=True,
                        help="Enable Global Consistency Alignment")
    parser.add_argument("--disable_gca", action="store_true", default=False,
                        help="Disable GCA while retaining entity-only alignment when enabled")
    
    # 关系分类器参数（新增）
    parser.add_argument("--enable_relation_classifier", action="store_true", default=False,
                        help="Enable relation classifier for relation alignment")
    parser.add_argument("--disable_relation_classifier", action="store_true", default=False,
                        help="Disable the relation classifier for a leave-one-out ablation")
    parser.add_argument("--relation_classifier_model", type=str, default=None,
                        help="Path to trained relation classifier model (e.g., relation_classifier_models/webnlg/checkpoint_epoch_10_best.pt)")
    parser.add_argument("--relation_classifier_threshold", type=float, default=0.5,
                        help="Confidence threshold for relation classifier (default: 0.5)")
    parser.add_argument("--enable_relation_reranker", action="store_true", default=False,
                        help="Enable conservative schema-aware relation reranking")
    parser.add_argument("--disable_relation_reranker", action="store_true", default=False,
                        help="Disable the relation reranker for a leave-one-out ablation")
    parser.add_argument("--relation_reranker_top_k", type=int, default=5,
                        help="Number of classifier candidates used by the relation reranker")
    parser.add_argument("--relation_reranker_min_prob", type=float, default=0.70,
                        help="Minimum candidate probability required by the relation reranker")
    parser.add_argument("--relation_reranker_alias_min_prob", type=float, default=0.0,
                        help="Minimum probability for strong OIE relation alias corrections")
    parser.add_argument("--relation_reranker_margin", type=float, default=0.20,
                        help="Minimum score margin over the original relation")
    parser.add_argument("--relation_reranker_mode", type=str, default="conservative",
                        choices=["conservative"],
                        help="Relation reranker rule profile")

    # 效率参数
    # Final output post-processing
    # Entity aligner parameters
    parser.add_argument("--enable_entity_aligner", action="store_true", default=False,
                        help="Enable neural entity aligner inside the GCA/PostProcessor stage")
    parser.add_argument("--entity_aligner_model", type=str,
                        default="entity_aligner_models_v4/entity_aligner_best.pt",
                        help="Path to trained entity aligner checkpoint")
    parser.add_argument("--entity_aligner_threshold", type=float, default=None,
                        help="Override entity aligner threshold; default uses checkpoint optimal_threshold")
    parser.add_argument("--entity_aligner_max_candidates", type=int, default=2000,
                        help="Maximum lexical candidates scored per unseen entity")
    parser.add_argument("--entity_aligner_data_dir", type=str, default="entity_alignment_data_v4",
                        help="Directory containing train/val/test JSON files used to expand entity aligner candidates")

    parser.add_argument("--enable_output_postprocess", action="store_true", default=False,
                        help="Enable final object normalization and conservative output post-processing")
    parser.add_argument(
        "--output_postprocess_profile",
        type=str,
        default="webnlg_conservative",
        help="Dataset-specific postprocess profile name or JSON path",
    )
    parser.add_argument("--disable_output_object_normalization", action="store_true", default=False,
                        help="Disable final object/surface normalization when output postprocess is enabled")
    parser.add_argument("--disable_output_relation_postprocess", action="store_true", default=False,
                        help="Disable final conservative relation repairs when output postprocess is enabled")
    parser.add_argument("--disable_output_dedupe", action="store_true", default=False,
                        help="Disable final duplicate triple removal when output postprocess is enabled")
    parser.add_argument("--output_postprocess_rule_set", type=str, default="conservative",
                        choices=["conservative"],
                        help="Final output postprocess rule profile")

    parser.add_argument("--max_feedback_iterations", type=int, default=2,
                        help="Maximum feedback iterations (default: 2 for efficiency)")
    parser.add_argument("--oie_batch_size", type=int, default=1,
                        help="Batch size for OIE API extraction. 1 keeps the original per-line behavior")
    parser.add_argument("--bi_sc_max_iterations", type=int, default=3,
                        help="Max iterations for Bi-SC (default: 3)")
    parser.add_argument("--gca_max_iterations", type=int, default=3,
                        help="Max iterations for GCA (default: 3)")

    # 消融实验模式
    parser.add_argument("--ablation_mode", type=str, default="full",
                        choices=["baseline", "ca_oie_only", "sd_ac_only", "bi_sc_only", "gca_only", "full"],
                        help="Ablation study mode")

    # 日志（与 run.py 一致）
    parser.add_argument(
        "--log_file_path",
        default=None,
        help="Optional path to a log file. When set, stdout and stderr are redirected to this file.",
    )
    parser.add_argument("--logging_verbose", action="store_const", dest="loglevel", const=logging.INFO)
    parser.add_argument("--logging_debug", action="store_const", dest="loglevel", const=logging.DEBUG)

    args = parser.parse_args()

    if getattr(args, "loglevel", None) is not None:
        logging.getLogger().setLevel(args.loglevel)

    log_file_path = args.log_file_path
    if log_file_path:
        log_path = pathlib.Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log_file:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                run_bf_edc(args)
    else:
        run_bf_edc(args)


if __name__ == "__main__":
    main()
