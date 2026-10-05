import unittest

try:
    import numpy as np
    import torch
    from lpr.images import blue_plate_box, crop_box, normalize_plate
    from lpr.ccpd import VOCABS
    from lpr.models import PlateCNN, predicted_indices, recognition_loss
    from lpr.predict import infer
    from lpr.runtime import box_iou
    from lpr.train import score_recognizer
except ImportError:
    np = None


@unittest.skipIf(np is None, "install project dependencies to run pipeline tests")
class PipelineTests(unittest.TestCase):
    def test_image_crop_and_normalization(self):
        image = np.zeros((20, 40, 3), dtype=np.uint8)
        image[5:15, 10:30] = 255
        crop = crop_box(image, (10, 5, 29, 14), inclusive=True)
        self.assertEqual(crop.shape, (10, 20, 3))
        self.assertEqual(normalize_plate(crop).shape, (64, 192, 3))
        with self.assertRaises(ValueError):
            crop_box(image, (0, 0, 100, 100))

    def test_model_output_and_gradient(self):
        model = PlateCNN()
        images = torch.rand(2, 3, 64, 192)
        labels = torch.zeros(2, 7, dtype=torch.long)
        logits = model(images)
        self.assertEqual(len(logits), 7)
        self.assertEqual(predicted_indices(logits).shape, (2, 7))
        loss = recognition_loss(logits, labels)
        loss.backward()
        self.assertIsNotNone(model.features[0].weight.grad)

    def test_iou(self):
        self.assertEqual(box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_recognizer_reports_minority_province_accuracy(self):
        class FixedModel(torch.nn.Module):
            def forward(self, images):
                predictions = torch.tensor([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
                return tuple(torch.nn.functional.one_hot(predictions[:, i], num_classes=33).float() for i in range(7))

        labels = torch.tensor([[0, 0, 0, 0, 0, 0, 0], [19, 0, 0, 0, 0, 0, 0]])
        metrics = score_recognizer(FixedModel(), [(torch.zeros(2, 3, 64, 192), labels)], "cpu")
        self.assertEqual(metrics["province_accuracy"], 0.5)
        self.assertEqual(metrics["non_anhui_province_accuracy"], 0.0)
        self.assertEqual(metrics["plate_accuracy"], 0.5)

    def test_low_confidence_province_is_marked_for_review(self):
        class FixedModel(torch.nn.Module):
            def __init__(self, confident):
                super().__init__()
                self.confident = confident

            def forward(self, images):
                logits = [torch.zeros(len(vocab)).unsqueeze(0) for vocab in VOCABS]
                if self.confident:
                    logits[0][0, 0] = 10
                return logits

        image = np.zeros((64, 192, 3), dtype=np.uint8)
        self.assertTrue(infer(image, FixedModel(False))["province_uncertain"])
        self.assertFalse(infer(image, FixedModel(True))["province_uncertain"])

    def test_blue_plate_fallback_crops_when_detector_misses(self):
        image = np.zeros((200, 500, 3), dtype=np.uint8)
        image[50:150, 100:400] = (0, 0, 180)
        image[65:135, 200:210] = 255

        class MissedDetector(torch.nn.Module):
            def forward(self, images):
                return [{"scores": torch.tensor([0.4]), "boxes": torch.tensor([[110, 55, 365, 145]])}]

        class FixedRecognizer(torch.nn.Module):
            def forward(self, images):
                return [torch.zeros(1, len(vocab)) for vocab in VOCABS]

        box = blue_plate_box(image, (110, 55, 365, 145))
        self.assertIsNotNone(box)
        self.assertTrue(100 <= box[0] < box[2] <= 400)
        self.assertTrue(50 <= box[1] < box[3] <= 150)
        result = infer(image, FixedRecognizer(), MissedDetector())
        self.assertEqual(result["detection_source"], "blue_color")
        self.assertIsNone(result["detection_score"])
        self.assertEqual(result["box"], list(box))
        self.assertIsNone(blue_plate_box(np.zeros_like(image), (110, 55, 365, 145)))

    def test_blue_plate_fallback_rejects_unrelated_blue_region(self):
        image = np.zeros((300, 500, 3), dtype=np.uint8)
        image[30:145, 80:420] = (0, 0, 180)  # Larger grille-shaped blue region.
        image[205:265, 160:360] = (0, 0, 180)  # Smaller plate near the detector hint.

        class Detector(torch.nn.Module):
            def __init__(self, scores, boxes):
                super().__init__()
                self.scores = scores
                self.boxes = boxes

            def forward(self, images):
                return [{"scores": torch.tensor(self.scores), "boxes": torch.tensor(self.boxes).reshape(-1, 4)}]

        class FixedRecognizer(torch.nn.Module):
            def forward(self, images):
                return [torch.zeros(1, len(vocab)) for vocab in VOCABS]

        recognizer = FixedRecognizer()
        supported = infer(image, recognizer, Detector([0.4], [[170, 210, 340, 260]]))
        self.assertEqual(supported["status"], "ok")
        self.assertEqual(supported["detection_source"], "blue_color")
        self.assertGreater(supported["box"][1], 200)
        self.assertEqual(infer(image, recognizer, Detector([0.4], [[0, 0, 40, 40]]))["status"], "no_plate")
        self.assertEqual(infer(image, recognizer, Detector([], []))["status"], "no_plate")

    def test_blue_region_extends_truncated_detector_box(self):
        image = np.zeros((200, 500, 3), dtype=np.uint8)
        image[50:150, 100:400] = (0, 0, 180)
        image[10:45, 20:180] = (0, 0, 180)

        class PartialDetector(torch.nn.Module):
            def forward(self, images):
                return [{"scores": torch.tensor([0.9]), "boxes": torch.tensor([[110, 55, 365, 145]])}]

        class FixedRecognizer(torch.nn.Module):
            def forward(self, images):
                return [torch.zeros(1, len(vocab)) for vocab in VOCABS]

        result = infer(image, FixedRecognizer(), PartialDetector())
        self.assertEqual(result["detection_source"], "cnn_blue_refined")
        self.assertGreater(result["box"][2], 380)
        self.assertEqual(result["box"][1:4:2], [55, 145])


if __name__ == "__main__":
    unittest.main()
