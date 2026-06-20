"""Dispatch render stages per config."""

from scripts.config import cfg


def main() -> None:
    data = cfg["data"] + "/"
    skip = cfg["skip"]["render"]
    fps = cfg["video"]["fps"]

    if "sam3" not in skip:
        print("[render] sam3")
        from src.sam3 import render as sam3

        sam3.render(
            video=data + cfg["video"]["npz"],
            masks=data + cfg["sam3"]["run"]["npz"],
            output=data + cfg["sam3"]["render"]["output"],
            fps=fps,
            alpha=cfg["sam3"]["render"]["alpha"],
        )

    if "moge" not in skip:
        print("[render] moge")
        from src.moge import render as moge

        moge.render(
            video=data + cfg["video"]["npz"],
            points=data + cfg["moge"]["run"]["npz"],
            output=data + cfg["moge"]["render"]["output"],
            fps=fps,
            point_size=cfg["moge"]["render"]["point_size"],
            camera=data + cfg["moge"]["render"]["camera"],
        )

    if "geocrafter" not in skip:
        print("[render] geocrafter")
        from src.geocrafter import render as geocrafter

        geocrafter.render(
            video=data + cfg["video"]["npz"],
            points=data + cfg["geocrafter"]["run"]["npz"],
            output=data + cfg["geocrafter"]["render"]["output"],
            fps=fps,
            point_size=cfg["geocrafter"]["render"]["point_size"],
            camera=data + cfg["geocrafter"]["render"]["camera"],
        )

    if "sfm" not in skip:
        print("[render] sfm")
        from src.sfm import render as sfm

        sfm.render(
            video=data + cfg["video"]["npz"],
            points=data + cfg["sfm"]["run"]["npz"],
            output=data + cfg["sfm"]["render"]["output"],
            fps=fps,
            point_size=cfg["sfm"]["render"]["point_size"],
            camera=data + cfg["sfm"]["render"]["camera"],
        )

    if "alignment" not in skip:
        print("[render] alignment")
        from src.alignment import render as alignment

        alignment.render(
            video=data + cfg["video"]["npz"],
            points=data + cfg["alignment"]["run"]["npz"],
            output=data + cfg["alignment"]["render"]["output"],
            fps=fps,
            point_size=cfg["alignment"]["render"]["point_size"],
            axis_length=cfg["alignment"]["render"]["axis_length"],
            camera=data + cfg["alignment"]["render"]["camera"],
            offset=cfg["alignment"]["run"]["offset"],
        )


if __name__ == "__main__":
    main()
