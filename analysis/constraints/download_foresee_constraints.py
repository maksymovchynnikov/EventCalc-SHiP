from argparse import ArgumentParser
from pathlib import Path
from urllib.request import urlopen
import json


BASE_DIR = Path(__file__).resolve().parent

FORESEE_MODELS = {
    "alp_photon": "ALP-photon",
    "alp_su2l": "ALP-W",
}


def parse_arguments():
    parser = ArgumentParser(description=("Download existing FORESEE ALP constraints."))
    parser.add_argument(
        "model",
        choices=FORESEE_MODELS,
        help="Constraint model to download.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    foresee_model = FORESEE_MODELS[arguments.model]

    api_url = (
        "https://api.github.com/repos/"
        "KlingFelix/FORESEE/contents/"
        f"Models/{foresee_model}/model/lines"
    )

    output_dir = BASE_DIR / "raw" / arguments.model
    output_dir.mkdir(parents=True, exist_ok=True)

    with urlopen(api_url) as response:
        files = json.load(response)

    selected_files = [
        file_info
        for file_info in files
        if (
            file_info["type"] == "file"
            and file_info["name"].startswith("bounds_")
            and file_info["name"].endswith(".txt")
        )
    ]

    for file_info in selected_files:
        output_path = output_dir / file_info["name"]

        with urlopen(file_info["download_url"]) as response:
            output_path.write_bytes(response.read())

        print(f"Downloaded {output_path}")

    print(f"\nDownloaded {len(selected_files)} {arguments.model} constraint files.")


if __name__ == "__main__":
    main()
