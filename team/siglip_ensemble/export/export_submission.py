"""Stage the selected trained heads and a pinned, offline SigLIP snapshot.

Export runs on CPU and does not train or allocate a CUDA model. A submission ZIP
is only created after a separate raw-input verification has passed and bound
every artifact file by SHA256.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile


MODEL_ID = "google/siglip-base-patch16-224"
MODEL_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
SOURCE_FILES = ("submission.py", "sentence_models.py", "text_tokenization.py")
SNAPSHOT_FILES = (
    "config.json", "model.safetensors", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def verify_file(path, expected):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing or symlink artifact: {path}")
    if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
        raise ValueError(f"Artifact differs from the supplied completed run: {path}")
    return path


def _safe_relative(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe relative artifact path: {relative}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes the run directory: {relative}")
    return path


def artifact_hashes(root):
    """Hash all final artifact files; importing Python may create ignored bytecode."""
    root = Path(root).resolve()
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"Do not package symbolic links: {relative}")
        if path.is_file():
            result[relative.as_posix()] = sha256_file(path)
    return result


def _checked_inputs(run_dir, payload_dir):
    selection_path = payload_dir / "selection.json"
    selection = read_json(selection_path)
    if (selection.get("schema") != 1 or selection.get("model_id") != MODEL_ID or
            selection.get("model_revision") != MODEL_REVISION):
        raise ValueError("Selection does not identify the fixed permitted SigLIP model")
    for name in SOURCE_FILES:
        if not (payload_dir / name).is_file():
            raise FileNotFoundError(f"Export helper needs its companion file: {name}")
    for name, key in (("request.json", "request"), ("comparison.json", "comparison")):
        verify_file(run_dir / name, selection[key])
    request = read_json(run_dir / "request.json")
    comparison = read_json(run_dir / "comparison.json")
    if (request.get("smoke") is not False or comparison.get("smoke") is not False or
            comparison.get("result_status") != "complete" or
            comparison.get("run_hash") != selection["run_hash"] or
            request.get("model") != selection["model_config"]):
        raise ValueError("Run identity, model configuration, or completion status differs")
    heads = selection.get("heads", [])
    if len(heads) != 3 or sorted(item["seed"] for item in heads) != [2026, 2027, 2028]:
        raise ValueError("Selected export requires exactly seeds 2026, 2027, 2028")
    for item in heads:
        verify_file(_safe_relative(run_dir, item["path"]), item)
    return selection, request, comparison


def _check_checkpoint(checkpoint, selection, item, payload_dir):
    import torch

    expected = {"kind": "sentence", "training_mode": "mixed", "seed": item["seed"],
                "epoch": item["epoch"], "model_config": selection["model_config"],
                "run_hash": selection["head_run_hash"]}
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Checkpoint identity differs for seed {item['seed']}")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint has no trained state_dict")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid trained tensor: {name}")
    specification = importlib.util.spec_from_file_location(
        "export_checked_sentence_model", payload_dir / "sentence_models.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    with torch.random.fork_rng(devices=[]):
        model = module.FrozenSiglipResidual(kind="sentence", **selection["model_config"])
    model.load_state_dict(state, strict=True)
    del model


def export_submission(run_dir, output_dir, payload_dir=None):
    """Stage an offline inference directory; return its path, never a final ZIP.

    ``run_dir`` is the user's completed Drive minimal-training run. Model files
    are fetched at their pinned revision into the normal Hugging Face cache if
    absent. Exporting the encoder uses CPU only. Existing outputs are refused.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    payload_dir = Path(payload_dir or Path(__file__).parent).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory; export never overwrites: {output_dir}")
    selection, request, comparison = _checked_inputs(run_dir, payload_dir)

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, PreTrainedTokenizerFast, SiglipModel

    # Check the small learned heads before downloading/copying the backbone.
    checkpoints = []
    for item in selection["heads"]:
        path = _safe_relative(run_dir, item["path"])
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        _check_checkpoint(checkpoint, selection, item, payload_dir)
        checkpoints.append((item, path))
        del checkpoint

    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "export_status.json"
    write_json(status_path, {"status": "running", "verified_for_submission": False})
    try:
        for name in SOURCE_FILES:
            shutil.copyfile(payload_dir / name, output_dir / name)
        (output_dir / "weights").mkdir()
        heads = []
        for item, path in checkpoints:
            relative = f"weights/seed{item['seed']}.pt"
            destination = output_dir / relative
            shutil.copyfile(path, destination)
            verify_file(destination, item)
            heads.append({"path": relative, "seed": item["seed"], "epoch": item["epoch"],
                          "sha256": item["sha256"], "bytes": item["bytes"]})

        snapshot = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION,
                                         allow_patterns=list(SNAPSHOT_FILES)))
        for required in ("config.json", "model.safetensors", "preprocessor_config.json",
                         "tokenizer.json", "tokenizer_config.json"):
            if not (snapshot / required).is_file():
                raise FileNotFoundError(f"Pinned SigLIP snapshot is incomplete: {required}")
        encoder_dir = output_dir / "models" / "siglip"
        encoder_dir.mkdir(parents=True)
        # No .cuda(), automatic device mapping, or training is used during export.
        encoder = SiglipModel.from_pretrained(str(snapshot), local_files_only=True,
                                             dtype=torch.float16, attn_implementation="eager")
        encoder.requires_grad_(False).to("cpu").eval()
        if (encoder.config.text_config.max_position_embeddings != 64 or
                encoder.config.text_config.hidden_size != selection["model_config"]["feature_dim"]):
            raise ValueError("Pinned model shape does not match the trained residual")
        if any(parameter.device.type != "cpu" for parameter in encoder.parameters()):
            raise ValueError("Export must keep every backbone parameter on CPU")
        encoder.save_pretrained(encoder_dir, safe_serialization=True)
        del encoder

        processor = AutoImageProcessor.from_pretrained(str(snapshot), local_files_only=True,
                                                       use_fast=False)
        processor.save_pretrained(encoder_dir)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(snapshot), local_files_only=True)
        if (not tokenizer.is_fast or tokenizer.eos_token_id != tokenizer.pad_token_id or
                tokenizer.num_special_tokens_to_add(pair=False) != 1):
            raise ValueError("Tokenizer differs from the fixed SigLIP EOS-padding contract")
        tokenizer.save_pretrained(encoder_dir)
        write_json(encoder_dir / "baseline_source.json", {
            "model_id": MODEL_ID, "revision": MODEL_REVISION,
            "storage_dtype": "float16", "export_device": "cpu",
            "source_snapshot_files": {p: sha256_file(snapshot / p) for p in SNAPSHOT_FILES
                                      if (snapshot / p).is_file()},
        })
        write_json(output_dir / "submission.json", {"api_version": 2})
        write_json(output_dir / "inference_config.json", {
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "encoder_path": "models/siglip", "model_config": selection["model_config"],
            "heads": heads, "image_batch_size": 8, "text_batch_size": 16,
            "cpu_threads": 4, "ensemble": "arithmetic_mean_of_three_final_scores",
            "attention_implementation": "eager", "encoder_storage_dtype": "float16",
        })
        write_json(output_dir / "training_info.json", {
            "selection_sha256": sha256_file(payload_dir / "selection.json"),
            "run_hash": selection["run_hash"], "head_run_hash": selection["head_run_hash"],
            "request": selection["request"], "comparison": selection["comparison"],
            "heads": heads, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "source_files": {name: sha256_file(output_dir / name) for name in SOURCE_FILES},
            "train_episodes": comparison.get("full_train_episodes"),
            "validation_episodes": comparison.get("full_validation_episodes"),
            "training_augmentation": request.get("training_augmentation"),
            "note": "Three independently trained E0 minimal-negative heads, averaged at inference. "
                    "No test input is used to fit or update parameters. Full reproduction sources "
                    "and generated training data remain in the training run, outside this inference artifact.",
        })
        write_json(status_path, {"status": "complete", "verified_for_submission": False,
                                "note": "Staging complete; separate raw-input offline verification is required before packaging."})
        write_json(output_dir / "export_manifest.json", {
            "schema": 1, "files": artifact_hashes(output_dir),
            "excluded_from_own_hash": ["export_manifest.json"],
        })
        print(f"Staged offline model at {output_dir}. No submission ZIP has been created.", flush=True)
        return output_dir
    except BaseException as error:
        write_json(status_path, {"status": "failed", "verified_for_submission": False,
                                "error": str(error)})
        raise


def package_submission(artifact_dir, output_zip, verification_path):
    """Package only an independently verified, unchanged inference directory."""
    artifact_dir = Path(artifact_dir).resolve()
    output_zip = Path(output_zip).resolve()
    verification_path = Path(verification_path).resolve()
    if output_zip.exists():
        raise FileExistsError(f"Will not overwrite ZIP: {output_zip}")
    if output_zip.is_relative_to(artifact_dir) or verification_path.is_relative_to(artifact_dir):
        raise ValueError("The ZIP and verification report must be outside the inference artifact directory")
    if read_json(artifact_dir / "export_status.json").get("status") != "complete":
        raise ValueError("Export staging did not finish")
    verification = read_json(verification_path)
    actual = artifact_hashes(artifact_dir)
    if verification.get("passed") is not True or verification.get("artifact_hashes") != actual:
        raise ValueError("Raw-input verification did not pass or the verified artifact has changed")
    required = {"submission.py", "submission.json", "inference_config.json", "sentence_models.py",
                "text_tokenization.py", "models/siglip/model.safetensors", "models/siglip/config.json",
                "models/siglip/preprocessor_config.json", "models/siglip/tokenizer.json"}
    if not required <= set(actual):
        raise ValueError(f"Incomplete inference payload: {sorted(required - set(actual))}")
    total_size = sum((artifact_dir / name).stat().st_size for name in actual)
    if len(actual) > 10000 or total_size > 1024 ** 3:
        raise ValueError("Payload exceeds the starter kit reference limit (10,000 files / 1 GiB expanded)")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation ensures a concurrent process cannot be overwritten.
    with zipfile.ZipFile(output_zip, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name in sorted(actual):
            archive.write(artifact_dir / name, name)
    with zipfile.ZipFile(output_zip) as archive:
        if set(archive.namelist()) != set(actual) or archive.testzip() is not None:
            raise ValueError("Written submission ZIP failed its integrity check")
        for name, expected in actual.items():
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"Packaged file differs from verified input: {name}")
    print(f"Created verified inference ZIP: {output_zip} "
          f"({output_zip.stat().st_size / 2**20:.1f} MiB, expanded {total_size / 2**20:.1f} MiB)", flush=True)
    return output_zip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--payload-dir", default=None)
    args = parser.parse_args()
    export_submission(args.run_dir, args.output_dir, args.payload_dir)


if __name__ == "__main__":
    main()
