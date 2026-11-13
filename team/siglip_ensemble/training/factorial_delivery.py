"""Display and package local experiment evidence, without any server submission."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
import zipfile

from experiment_io import read_json, verified_file


def show_and_package(run_dir, *, team_validation_dir=None, sentence_feature_dir=None):
    run_dir = Path(run_dir)
    folder = run_dir / "comparison"
    status = read_json(folder / "status.json")
    if status.get("status") != "complete":
        raise RuntimeError("E0~E3 비교가 아직 완료되지 않았습니다.")
    for name, record in status.get("artifacts", {}).items():
        verified_file(folder / name, record)
    report = read_json(folder / "comparison.json")
    if report.get("smoke_subset") or report.get("validation_episodes") != 834:
        raise RuntimeError("공유 결과는 834개 전체 검증이어야 합니다.")
    if report.get("run_hash") != status.get("run_hash"):
        raise RuntimeError("결과와 완료 기록이 일치하지 않습니다.")
    with (folder / "comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    table = "<table style='border-collapse:collapse'>"
    for i, row in enumerate(rows):
        tag = "th" if i == 0 else "td"
        values = []
        for value in row:
            if i and "." in value:
                try:
                    value = f"{float(value):.6f}"
                except ValueError:
                    pass
            values.append(f"<{tag} style='border:1px solid #ccc;padding:7px'>{html.escape(value)}</{tag}>")
        table += "<tr>" + "".join(values) + "</tr>"
    try:
        from IPython.display import HTML, display
        display(HTML(table + "</table>"))
    except ImportError:
        print((folder / "comparison.csv").read_text(encoding="utf-8-sig"))
    for key in ("effects", "factorial_effects", "interaction", "reuse_summary"):
        if key in report:
            print(key + ":", json.dumps(report[key], ensure_ascii=False, indent=2))
    print("기존 random/team 점수와 순서 보조 평가는 별도입니다. 리더보드 점수가 아닙니다.")
    destination = run_dir / "factorial_results.zip"
    temporary = destination.with_suffix(".zip.part")
    written = set()
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        def add(path, name):
            if name not in written:
                archive.write(path, name)
                written.add(name)
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".csv", ".gz"}:
                add(path, path.relative_to(run_dir).as_posix())
        for path in sorted((run_dir / "source").rglob("*.py")):
            add(path, path.relative_to(run_dir).as_posix())
        for label, directory in (("original_team_validation", team_validation_dir),
                                 ("original_sentence_features", sentence_feature_dir)):
            if directory:
                for path in sorted(Path(directory).glob("*.json")):
                    add(path, label + "/" + path.name)
        for name in ("environment.json", "requirements-colab.txt", "requirements-installed.txt",
                     "feature_source.json", "delivery.json"):
            path = run_dir / name
            if path.is_file():
                add(path, name)
    temporary.replace(destination)
    print("결과 폴더:", folder)
    print("공유할 파일:", destination)
    print("실제 증강 문장·오답 스케줄·학습 기록·코드가 포함됩니다. 가중치와 특징 배열은 Drive에 남습니다.")
    print("채점 사이트에는 제출하지 않았습니다.")
    return destination
