from PIL import Image
import pillow_heif
import os

pillow_heif.register_heif_opener()

# folder = "C:\\Users\\Husam\\Desktop\\Projects\\CS117\\CS117-ComputerVisionProject\\photos\\Calibration"
# for file in os.listdir(folder):
#     if file.endswith(".HEIC"):
#         heic_path = os.path.join(folder, file)
#         jpg_path = os.path.splitext(heic_path)[0] + ".jpg"
#         with Image.open(heic_path) as img:
#             img.convert("RGB").save(jpg_path, "JPEG")
#         os.remove(heic_path)
# folder = "C:\\Users\\Husam\\Desktop\\Projects\\CS117\\CS117-ComputerVisionProject\\photos\\Captures\\grab5"
# files = sorted(os.listdir(folder))
# pair_count = 0

# for i in range(0, len(files), 2):
#     if i + 1 < len(files):
#         file1 = os.path.join(folder, files[i])
#         file2 = os.path.join(folder, files[i + 1])
        
#         new_name1 = os.path.join(folder, f"frame_C0_{pair_count:02d}.jpg")
#         new_name2 = os.path.join(folder, f"frame_C1_{pair_count:02d}.jpg")
        
#         os.rename(file1, new_name1)
#         os.rename(file2, new_name2)
        
#         print(f"Renamed: {file1} -> {new_name1}")
#         print(f"Renamed: {file2} -> {new_name2}")
        
#         pair_count += 1

# print("Renaming complete!")



folder = "C:\\Users\\Husam\\Desktop\\Projects\\CS117\\CS117-ComputerVisionProject\\photos\\Captures"

for subfold in os.listdir(folder):
    for file in os.listdir(folder + "\\" + subfold):
        if file.endswith(".jpg"):
            jpg_path = os.path.join(folder + "\\" + subfold, file)
            png_path = os.path.splitext(jpg_path)[0] + ".png"
            with Image.open(jpg_path) as img:
                img.convert("RGB").save(png_path, "PNG")
            os.remove(jpg_path)


print("success")
