from image_stitching import ImageStitching

images: list = [
    "images/img1.png",
    "images/img2.png",
    "images/img3.png",
]


def main():
    app = ImageStitching()
    app.run(images)


if __name__ == "__main__":
    main()
