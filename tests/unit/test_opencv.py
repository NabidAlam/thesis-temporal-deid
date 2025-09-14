import cv2
import numpy as np
import os

# Test 1: Create a simple mask
print("Test 1: Creating simple mask")
mask = np.ones((1080, 1920), dtype=np.uint8) * 255
print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
print(f"Mask max: {np.max(mask)}, min: {np.min(mask)}")
print(f"Mask sum: {np.sum(mask)}")

# Test 2: Save mask with cv2.imwrite
print("\nTest 2: Saving mask with cv2.imwrite")
result = cv2.imwrite('test_mask_cv2.png', mask)
print(f"cv2.imwrite result: {result}")

# Test 3: Check if file was created
print("\nTest 3: Checking file")
if os.path.exists('test_mask_cv2.png'):
    file_size = os.path.getsize('test_mask_cv2.png')
    print(f"File created, size: {file_size} bytes")
else:
    print("File not created")

# Test 4: Read back the saved mask
print("\nTest 4: Reading saved mask")
saved_mask = cv2.imread('test_mask_cv2.png', cv2.IMREAD_GRAYSCALE)
if saved_mask is not None:
    print(f"Saved mask shape: {saved_mask.shape}")
    print(f"Saved mask max: {np.max(saved_mask)}, min: {np.min(saved_mask)}")
    print(f"Saved mask sum: {np.sum(saved_mask)}")
else:
    print("Failed to read saved mask")

# Test 5: Try with PIL as alternative
print("\nTest 5: Testing PIL alternative")
try:
    from PIL import Image
    pil_mask = Image.fromarray(mask)
    pil_mask.save('test_mask_pil.png')
    print("PIL save successful")
    
    # Check PIL file
    if os.path.exists('test_mask_pil.png'):
        pil_file_size = os.path.getsize('test_mask_pil.png')
        print(f"PIL file size: {pil_file_size} bytes")
        
        # Read PIL file
        pil_saved = Image.open('test_mask_pil.png')
        pil_array = np.array(pil_saved)
        print(f"PIL saved mask shape: {pil_array.shape}")
        print(f"PIL saved mask max: {np.max(pil_array)}, min: {np.min(pil_array)}")
        print(f"PIL saved mask sum: {np.sum(pil_array)}")
    else:
        print("PIL file not created")
        
except ImportError:
    print("PIL not available")

print("\nTest completed!")
