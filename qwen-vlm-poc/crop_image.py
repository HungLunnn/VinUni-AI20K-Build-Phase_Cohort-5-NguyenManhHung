from PIL import Image

image = Image.open("images/test2.jpg")

crop_box = (0, 0, 1000, 1000)
cropped = image.crop(crop_box)

cropped.save("images/crop_test.jpg")

print("Original:", image.size)
print("Crop:", cropped.size)
print("Saved: images/crop_test.jpg")