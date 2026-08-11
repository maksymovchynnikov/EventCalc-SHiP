"""Download FORESEE ALP constraint polygons into the analysis output namespace."""

from argparse import ArgumentParser
import json
from urllib.request import urlopen

from alp_discrimination.cache import atomic_output_path
from alp_discrimination.config import PROFILES
from alp_discrimination.paths import profile_output_dir

FORESEE_MODELS = {"alp_photon": "ALP-photon", "alp_su2l": "ALP-W"}


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=FORESEE_MODELS)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="production")
    args = parser.parse_args()
    url = (
        "https://api.github.com/repos/KlingFelix/FORESEE/contents/"
        f"Models/{FORESEE_MODELS[args.model]}/model/lines"
    )
    with urlopen(url) as response:
        files = json.load(response)
    selected = [
        item for item in files if item["type"] == "file"
        and item["name"].startswith("bounds_") and item["name"].endswith(".txt")
    ]
    output_dir = profile_output_dir(args.profile, "constraints") / "raw" / args.model
    for item in selected:
        with urlopen(item["download_url"]) as response:
            content = response.read()
        with atomic_output_path(output_dir / item["name"]) as temporary:
            temporary.write_bytes(content)
        print(f"Downloaded {item['name']}")
    print(f"Downloaded {len(selected)} constraint files to {output_dir}")


if __name__ == "__main__":
    main()
