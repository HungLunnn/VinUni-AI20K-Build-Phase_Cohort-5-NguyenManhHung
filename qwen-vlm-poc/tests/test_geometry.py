import unittest

from PIL import Image

from vlm.geometry import clamp_bbox, crop_with_padding, iou, mark_bbox, touches_border


class GeometryTests(unittest.TestCase):
    def test_touches_border(self):
        size = (2000, 1360)
        self.assertEqual(touches_border((1832, 565, 2000, 698), size), ["right"])
        self.assertEqual(touches_border((0, 0, 50, 50), size), ["left", "top"])
        self.assertEqual(touches_border((100, 100, 200, 200), size), [])
        self.assertEqual(touches_border((100, 1300, 200, 1360), size), ["bottom"])

    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3, places=3)

    def test_clamp(self):
        self.assertEqual(clamp_bbox((-5, -5, 3000, 2000), (2000, 1360)), (0.0, 0.0, 2000.0, 1360.0))

    def test_crop_stays_inside_image_and_has_min_side(self):
        img = Image.new("RGB", (640, 480))
        for bbox in [(0, 0, 20, 20), (620, 460, 640, 480), (300, 200, 310, 210)]:
            crop, box = crop_with_padding(img, bbox, pad_ratio=0.3, min_side=224)
            self.assertGreaterEqual(box[0], 0)
            self.assertGreaterEqual(box[1], 0)
            self.assertLessEqual(box[2], 640)
            self.assertLessEqual(box[3], 480)
            self.assertGreaterEqual(crop.size[0], 200)
            self.assertGreaterEqual(crop.size[1], 200)

    def test_mark_does_not_mutate_original(self):
        img = Image.new("RGB", (100, 100), "white")
        marked = mark_bbox(img, (10, 10, 50, 50), (0, 0, 100, 100))
        self.assertEqual(img.getpixel((10, 10)), (255, 255, 255))
        self.assertNotEqual(marked.getpixel((10, 10)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
