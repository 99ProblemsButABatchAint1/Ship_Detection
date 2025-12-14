from app.services.detector import ShipDetector, DetectorConfig

detector = ShipDetector(DetectorConfig(
    classifier_path="models/best_classifier.pth",
    segmenter_path="models/best_ship_segmenter.pth",
    device="cpu",  # later "cuda" if you want GPU
))
