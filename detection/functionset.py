def load_image(image_path):
    try:
        # 이미지를 읽어와서 RGB 형식으로 변환합니다.
        img_arr = cv2.imread(image_path)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        return img_arr
    except Exception as e:
        raise Exception(f"Error loading image {image_path}: {e}")