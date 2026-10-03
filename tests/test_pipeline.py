import unittest

try:
    import numpy as np
    import torch
    from lpr.images import crop_box, normalize_plate
    from lpr.models import PlateCNN, predicted_indices, recognition_loss
    from lpr.runtime import box_iou
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


if __name__ == "__main__":
    unittest.main()
