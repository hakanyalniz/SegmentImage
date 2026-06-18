import cv2
import numpy as np
from ultralytics import SAM


def main():
    # Provide the path to your image
    image_path = "./test/test_image2.jpg"

    # Load and display the image for coordinate selection
    # Also check if the image is available
    img = cv2.imread(image_path)
    if img is None:
        print(
            f"Error: Could not load image from path '{image_path}'. Check the filename."
        )
        return

    # Create a window, show the img. Set up the mouse click event, wait infinitely, then exit
    print(
        "Click and drag a box around the entire character, then press ENTER or SPACE."
    )
    roi = cv2.selectROI("Select Character", img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # roi gives coordinates on the display image: (x, y, w, h)
    w, h = roi

    # Check if the selection is empty (width or height is 0)
    if w == 0 or h == 0:
        print("Selection cancelled or invalid box drawn. Exiting.")
        return

    # roi returns: (x_start, y_start, width, height)
    # SAM 2 expects: [xmin, ymin, xmax, ymax]
    bbox = [int(roi[0]), int(roi[1]), int(roi[0] + roi[2]), int(roi[1] + roi[3])]

    print("Loading SAM 2 model and generating mask...")

    # Load the SAM 2 model
    # 'sam2_b.pt' is the base model
    model = SAM("sam2_b.pt")

    # Run inference using your clicked coordinate point
    # labels=[1] tells SAM 2 that the point represents the foreground object.
    results = model(image_path, bboxes=[bbox], labels=[1])

    # Extract and process the generated binary mask
    result = results[0]
    if result.masks is not None:
        # Convert the tensor format mask to a binary NumPy array (0 or 255)
        # result.masks.data[0] is the result tensor data. cpu() moves the data to RAM. numpy() this formats the tensor to array
        # We multiply by 255 so the 0.0 and 1.0 of the tensor can range from 0 to 255
        # We need to turn this into an image. Right now they are floats inside the array. So turn them into integers
        mask_array = result.masks.data[0].cpu().numpy() * 255
        mask_array = mask_array.astype(np.uint8)

        # Resize mask back to match original image dimensions if necessary
        mask_img = cv2.resize(mask_array, (img.shape[1], img.shape[0]))

        # Save the crisp binary mask
        output_filename = "./test/output_shadow_mask.png"
        cv2.imwrite(output_filename, mask_img)
        print(f"Success! Mask saved as '{output_filename}'")

        # Show a quick preview of the generated mask
        cv2.imshow("Generated Mask Preview (Press 0 to close)", mask_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("SAM 2 was unable to compute a distinct mask based on that point.")


if __name__ == "__main__":
    main()
