from tracker import Tracker, TrackerConfig

cfg = TrackerConfig(
    model_path=r".\best.pt",
    tracker_yaml=r".\yaml\botsort_hornets.yaml",
    target_classes=[2],
    gmm_downscale=0.5
)

result = Tracker(cfg).run(r"B:\School\Masterproef\cleaned\cropped_vids\GX010172_1280x720_full_joined.mp4", output_path="./results/test_org2.mp4", write_video=True, vector_csv="exits.csv")
print(result.exit_vector)