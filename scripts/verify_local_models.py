import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "generator_protocol": ROOT / "models" / "Qwen" / "Qwen2.5-7B-Instruct",
    "generator_fallback": Path("D:/local_qwen_models/Qwen3.5-4B/Qwen3.5-4B"),
    "dense_retriever": ROOT / "models" / "BAAI" / "bge-base-en-v1.5",
    "reranker": ROOT / "models" / "BAAI" / "bge-reranker-base",
}


def inspect_model(path: Path) -> dict:
    result = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    try:
        cfg = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
        result["config_ok"] = True
        result["model_type"] = getattr(cfg, "model_type", None)
        result["architectures"] = getattr(cfg, "architectures", None)
    except Exception as exc:
        result["config_ok"] = False
        result["config_error"] = repr(exc)
    try:
        tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
        result["tokenizer_ok"] = True
        result["tokenizer_class"] = type(tok).__name__
    except Exception as exc:
        result["tokenizer_ok"] = False
        result["tokenizer_error"] = repr(exc)
    return result


def main() -> int:
    report = {name: inspect_model(path) for name, path in MODELS.items()}
    out = ROOT / "models" / "local_model_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
