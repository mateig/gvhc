"""Run the SAM3 stage on the converted video npz."""

from scripts.config import cfg
from src.sam3 import stage as sam3


def main() -> None:
    data = cfg["data"] + "/"
    print("[run] sam3")
    sam3.run(
        video=data + cfg["video"]["npz"],
        output=data + cfg["sam3"]["run"]["npz"],
        prompt=cfg["sam3"]["run"]["prompt"],
    )


if __name__ == "__main__":
    main()
