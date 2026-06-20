"""Dispatch run stages per config."""

from scripts.config import cfg


def main() -> None:
    data = cfg["data"] + "/"
    skip = cfg["skip"]["run"]

    if "moge" not in skip:
        print("[run] moge")
        from src.moge import stage as moge

        moge.run(
            video=data + cfg["video"]["npz"],
            output=data + cfg["moge"]["run"]["npz"],
            weights=cfg["moge"]["run"]["weights"],
            num_tokens=cfg["moge"]["run"]["num_tokens"],
            batch_size=cfg["moge"]["run"]["batch_size"],
        )

    if "geocrafter" not in skip:
        print("[run] geocrafter")
        from src.geocrafter import stage as geocrafter

        geocrafter.run(
            video=data + cfg["video"]["npz"],
            moge=data + cfg["moge"]["run"]["npz"],
            output=data + cfg["geocrafter"]["run"]["npz"],
            weights=cfg["geocrafter"]["run"]["weights"],
            process_resolution=cfg["geocrafter"]["run"]["process_resolution"],
            steps=cfg["geocrafter"]["run"]["steps"],
            window_size=cfg["geocrafter"]["run"]["window_size"],
            overlap=cfg["geocrafter"]["run"]["overlap"],
            batch_size=cfg["geocrafter"]["run"]["batch_size"],
        )

    if "sfm" not in skip:
        print("[run] sfm")
        from src.sfm import stage as sfm

        sfm.run(
            video=data + cfg["video"]["npz"],
            points=data + cfg["geocrafter"]["run"]["npz"],
            masks=data + cfg["sam3"]["run"]["npz"],
            output=data + cfg["sfm"]["run"]["npz"],
            gaps=cfg["sfm"]["run"]["gaps"],
            dilate_ksize=cfg["sfm"]["run"]["dilate_ksize"],
            depth_percentile=cfg["sfm"]["run"]["depth_percentile"],
            sample=cfg["sfm"]["run"]["sample"],
            cycle_threshold=cfg["sfm"]["run"]["cycle_threshold"],
            ransac_threshold=cfg["sfm"]["run"]["ransac_threshold"],
            ransac_iters=cfg["sfm"]["run"]["ransac_iters"],
            min_inliers=cfg["sfm"]["run"]["min_inliers"],
            smooth_window=cfg["sfm"]["run"]["smooth_window"],
            smooth_polyorder=cfg["sfm"]["run"]["smooth_polyorder"],
        )

    if "alignment" not in skip:
        print("[run] alignment")
        from src.alignment import stage as alignment

        alignment.run(
            keypoints=data + cfg["keypoints"]["npz"],
            sfm=data + cfg["sfm"]["run"]["npz"],
            output=data + cfg["alignment"]["run"]["npz"],
            p1=cfg["alignment"]["run"]["p1"],
            p2=cfg["alignment"]["run"]["p2"],
            p3=cfg["alignment"]["run"]["p3"],
            known_distance=cfg["alignment"]["run"]["known_distance"],
            offset=cfg["alignment"]["run"]["offset"],
        )

    if "mesh" not in skip:
        print("[run] mesh")
        from src.mesh import stage as mesh

        mesh.run(
            alignment=data + cfg["alignment"]["run"]["npz"],
            masks=data + cfg["sam3"]["run"]["npz"],
            output=data + cfg["mesh"]["run"]["stl"],
            dilate_ksize=cfg["mesh"]["run"]["dilate_ksize"],
            radius=cfg["mesh"]["run"]["radius"],
            voxel_size=cfg["mesh"]["run"]["voxel_size"],
            outlier_nb_neighbors=cfg["mesh"]["run"]["outlier_nb_neighbors"],
            outlier_std_ratio=cfg["mesh"]["run"]["outlier_std_ratio"],
            normal_radius=cfg["mesh"]["run"]["normal_radius"],
            normal_max_nn=cfg["mesh"]["run"]["normal_max_nn"],
            poisson_depth=cfg["mesh"]["run"]["poisson_depth"],
            density_quantile=cfg["mesh"]["run"]["density_quantile"],
            smooth_iters=cfg["mesh"]["run"]["smooth_iters"],
        )


if __name__ == "__main__":
    main()
