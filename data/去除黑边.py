from PIL import Image, ImageFilter
import cv2


def crop_edges_PIL():
    # Open the image
    img = Image.open("image.bmp")

    # Convert the image to grayscale
    img_gray = img.convert("L")

    # Get the edges of the image
    edges = img_gray.filter(ImageFilter.FIND_EDGES)

    # Crop the image to remove the black border
    cropped = img.crop(edges.getbbox())

    # Save the cropped image
    cropped.save("cropped_image.bmp")

def crop_edges_CV():
    # Read the image
    img = cv2.imread("image.bmp")

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get the edges of the image
    edges = cv2.Canny(img_gray, 10, 10)

    cv2.imwrite("edges.bmp", edges)
    # Crop the image to remove the black border
    x, y, w, h = cv2.boundingRect(edges)
    print(x, y, w, h)
    cropped = img[y:y + h, x:x + w]

    # Save the cropped image
    cv2.imwrite("cropped_image.bmp", cropped)

if __name__ == '__main__':
    crop_edges_CV()
