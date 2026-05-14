import os
import cv2
import numpy as np
from pathlib import Path


# =========================
# SETTINGS TO PERSONALIZE
# =========================

RAW_TRAINING_DIR = "data/raw_training_pages"      # EDIT: folder containing your exported iPad training pages
OUTPUT_DIR = "data/cropped_digits"               # EDIT: folder where cropped digit images will be saved

NUM_ROWS = 10                                    # EDIT only if your page does not have 10 digit rows
NUM_COLS = 10                                    # EDIT only if each row does not have 10 digit examples

IGNORE_LEFT_FRACTION = 0.20                      # EDIT: fraction of page width to ignore for row labels
TOP_MARGIN_FRACTION = 0.03                       # EDIT: increase if first row gets cut off or includes title space
BOTTOM_MARGIN_FRACTION = 0.03                    # EDIT: increase if bottom area includes extra marks
RIGHT_MARGIN_FRACTION = 0.03                     # EDIT: increase if right edge has extra marks

FINAL_IMAGE_SIZE = 28                            # EDIT only if your model expects a different image size
PADDING = 8                                      # EDIT: margin added around digit before resizing

SAVE_DEBUG_CELLS = True                          # EDIT: set False once cropping works
DEBUG_DIR = "data/debug_cells"                   # EDIT: folder for checking raw cell crops


# =========================
# HELPER FUNCTIONS
# =========================

def make_output_folders():
    """
    Creates folders:
    data/cropped_digits/0
    data/cropped_digits/1
    ...
    data/cropped_digits/9
    """
    for digit in range(10):
        Path(os.path.join(OUTPUT_DIR, str(digit))).mkdir(parents=True, exist_ok=True)

    if SAVE_DEBUG_CELLS:
        Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)


def load_image(image_path):
    """
    Loads one training page image.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return image


def preprocess_cell(cell):
    """
    Converts one cropped cell into a clean 28x28 digit image.

    Output format:
    - black background
    - white handwriting
    - centered digit
    - 28x28 pixels
    """

    # Convert to grayscale
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # Slight blur helps remove tiny artifacts
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Convert to black background and white handwriting
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,                                      # EDIT if thresholding looks bad
        15                                       # EDIT if digits are too faint or background appears
    )

    # Find all white handwriting pixels
    ys, xs = np.where(binary > 0)

    # If the crop is empty, skip it
    if len(xs) == 0 or len(ys) == 0:
        return None

    # Crop tightly around the handwriting
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    digit = binary[y_min:y_max + 1, x_min:x_max + 1]

    # Pad to square so the digit does not get stretched
    h, w = digit.shape
    size = max(h, w)

    square = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    square[y_offset:y_offset + h, x_offset:x_offset + w] = digit

    # Add margin around the digit
    padded = cv2.copyMakeBorder(
        square,
        PADDING,
        PADDING,
        PADDING,
        PADDING,
        cv2.BORDER_CONSTANT,
        value=0
    )

    # Resize to model input size
    resized = cv2.resize(
        padded,
        (FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return resized


def crop_page_into_digits(image, page_stem, starting_counts):
    """
    Splits one full training page into 10 rows and 10 columns.
    Saves each cropped digit into the correct folder based on row number.
    """

    height, width, _ = image.shape

    # Define the usable area of the page
    x_start = int(width * IGNORE_LEFT_FRACTION)       # EDIT via IGNORE_LEFT_FRACTION if labels are included
    x_end = int(width * (1 - RIGHT_MARGIN_FRACTION))  # EDIT via RIGHT_MARGIN_FRACTION if right side is cut badly

    y_start = int(height * TOP_MARGIN_FRACTION)       # EDIT via TOP_MARGIN_FRACTION if top row is wrong
    y_end = int(height * (1 - BOTTOM_MARGIN_FRACTION))# EDIT via BOTTOM_MARGIN_FRACTION if bottom row is wrong

    digit_area = image[y_start:y_end, x_start:x_end]

    area_height, area_width, _ = digit_area.shape

    cell_height = area_height // NUM_ROWS
    cell_width = area_width // NUM_COLS

    for row in range(NUM_ROWS):
        label = row  # row 0 is digit 0, row 1 is digit 1, etc.

        for col in range(NUM_COLS):
            cell_y1 = row * cell_height
            cell_y2 = (row + 1) * cell_height

            cell_x1 = col * cell_width
            cell_x2 = (col + 1) * cell_width

            cell = digit_area[cell_y1:cell_y2, cell_x1:cell_x2]

            # Save raw cell crop for debugging
            if SAVE_DEBUG_CELLS:
                debug_name = f"{page_stem}_row{row}_col{col}.png"
                debug_path = os.path.join(DEBUG_DIR, debug_name)
                cv2.imwrite(debug_path, cell)

            processed_digit = preprocess_cell(cell)

            if processed_digit is None:
                print(f"Skipped empty crop: page={page_stem}, row={row}, col={col}")
                continue

            # Count existing images for this label
            starting_counts[label] += 1
            digit_number = starting_counts[label]

            output_name = f"{label}_{digit_number:04d}.png"
            output_path = os.path.join(OUTPUT_DIR, str(label), output_name)

            cv2.imwrite(output_path, processed_digit)

    return starting_counts


def get_existing_counts():
    """
    Counts existing saved images so filenames do not overwrite old crops.
    """
    counts = {}

    for digit in range(10):
        folder = os.path.join(OUTPUT_DIR, str(digit))
        Path(folder).mkdir(parents=True, exist_ok=True)

        existing_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(".png")
        ]

        counts[digit] = len(existing_files)

    return counts


def main():
    make_output_folders()

    image_extensions = [".png", ".jpg", ".jpeg"]      # EDIT if your app exports a different image type

    raw_dir = Path(RAW_TRAINING_DIR)

    image_paths = [
        path for path in raw_dir.iterdir()
        if path.suffix.lower() in image_extensions
    ]

    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        print(f"No images found in {RAW_TRAINING_DIR}")
        print("Export your iPad pages as PNG or JPG and put them in that folder.")
        return

    counts = get_existing_counts()

    for image_path in image_paths:
        print(f"Processing {image_path.name}...")

        image = load_image(image_path)
        page_stem = image_path.stem

        counts = crop_page_into_digits(
            image=image,
            page_stem=page_stem,
            starting_counts=counts
        )

    print("\nDone cropping training digits.")
    print(f"Cropped digits saved in: {OUTPUT_DIR}")

    if SAVE_DEBUG_CELLS:
        print(f"Debug cell crops saved in: {DEBUG_DIR}")
        print("Open this folder first to check whether the grid split is correct.")


if __name__ == "__main__":
    main()