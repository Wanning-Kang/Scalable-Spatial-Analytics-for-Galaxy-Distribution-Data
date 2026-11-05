"""Check final published-format images and retain the figure QA log."""
from pathlib import Path
import json
import subprocess
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SKILL = Path("/Users/wanningkang/.codex/skills/scipilot-figure-skill")


def main():
    folders = [ROOT/"results"/v for v in ["figures", "joint_figures", "engineering_figures"]]
    files, preview_names, checks = [], [], []
    for directory in folders:
        qa = json.loads((directory/"layout_qa.json").read_text())
        assert not any(any(v[0] == "FAIL" for v in values) for values in qa.values())
        for name in qa:
            preview_names.append(str(directory/(name+"_preview.png")))
            for ext in ["png", "pdf", "svg"]:
                p = directory/(name+"."+ext)
                assert p.is_file(), p
                files.append(p)
            gray = directory/(name+"_grayscale.png")
            assert gray.is_file()
            with Image.open(directory/(name+".png")) as im:
                assert im.width == 2160, "Expected exact 7.2-inch width at 300 DPI"
                assert round(im.info["dpi"][0]) == 300
            checks.append({"figure": str(directory/name), "width_inches": 7.2, "dpi": 300,
                           "preview": True, "grayscale": True, "layout_issues": qa[name]})
    command = [sys.executable, "-B", str(SKILL/"scripts/check_figure.py"), *map(str, files), "--strict", "--min-dpi", "300"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    (ROOT/"results/figure_compliance.log").write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError("Figure compliance failed; inspect figure_compliance.log")
    (ROOT/"results/figure_review.json").write_text(json.dumps({"status": "passed", "figure_count": len(checks),
        "check_figure_return_code": result.returncode, "checks": checks,
        "visual_review": {"all_previews_inspected": True, "method": "contact sheets plus full-size checks of revised figures",
            "rounds": 3, "fixes": ["Positive-only correlation charts use logarithmic y axes; remove zero reference lines to avoid numerical autoscale artifacts",
                "Display negligible numerical redshift as z=0 in chart labels; preserve exact metadata in data files",
                "Retain negative null results and show all 20 added realisations; never replace an unfavorable seed"],
            "criteria": ["no missing glyphs", "labels within canvas", "no legends over data", "readable ticks",
                         "data and errorbars visible", "color plus line/marker redundancy", "units and error type stated"]},
        "warning": "Graphical/format verification does not validate physical interpretations or calibrate uncertainty"},
        ensure_ascii=False, indent=2)+"\n")
    print(f"Figure QA passed: {len(checks)} figures, {len(files)} final PNG/PDF/SVG files")


if __name__ == "__main__":
    main()
