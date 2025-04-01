import cv2
import os
from deepface import DeepFace
import faiss
import numpy as np
import tempfile
import tensorflow as tf

# Kiểm tra xem có GPU hay không
gpus = tf.config.list_physical_devices('GPU')
print("Sử dụng GPU:", gpus)  # Thêm dòng này để in danh sách GPU

if gpus:
    for gpu in gpus:
        tf.config.set_logical_device_configuration(
            gpu,
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        )
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("GPU đã sẵn sàng và đang được sử dụng!")


def get_all_subfolders(folder):
    """
    Lấy danh sách tất cả các thư mục con bên trong một thư mục.
    """
    subfolders = [
        os.path.join(folder, d)
        for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d))
    ]
    return subfolders

VECTOR_SIZE = 128

index = faiss.IndexFlatIP(VECTOR_SIZE)

face_database = {}
face_ids = []

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def check_image_folders(folder_paths):
    """
    Kiểm tra danh sách thư mục có chứa ảnh không
    """
    valid_folders = []
    for folder_path in folder_paths:
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print(f"Thư mục {folder_path} không tồn tại.")
            continue

        image_files = [
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            print(f"Không tìm thấy ảnh trong thư mục {folder_path}.")
            continue

        valid_folders.append(folder_path)

    return valid_folders

def preprocess_face(img, is_path=False):
    """
    Hàm tiền xử lý khuôn mặt bằng OpenCV Haarcascade
    """
    if is_path:
        img = cv2.imread(img)
        if img is None:
            return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, w, h in faces:
        face = img[y : y + h, x : x + w]
        if face.size == 0:
            continue
        face = cv2.resize(face, (224, 224))
        face = face.astype(np.float32) / 255.0
        return face, (x, y, w, h)
    print("Không phát hiện khuôn mặt")
    return None, None

def get_face_embedding(face):
    """
    Lấy embedding vector từ DeepFace
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
        temp_path = tmpfile.name
        cv2.imwrite(temp_path, (face * 255).astype(np.uint8))
    embedding_result = DeepFace.represent(
        img_path=temp_path, model_name="Facenet", enforce_detection=False
    )
    os.remove(temp_path)
    embedding = np.array(embedding_result[0]["embedding"], dtype=np.float32).flatten()
    return embedding / np.linalg.norm(embedding)

def register_faces(root_folder):
    """
    Đăng ký khuôn mặt từ một thư mục tổng chứa nhiều thư mục con
    """
    folder_paths = get_all_subfolders(root_folder)
    valid_folders = check_image_folders(folder_paths)
    global index, face_database, face_ids

    for folder_path in valid_folders:
        face_vectors = []
        folder_name = os.path.basename(folder_path)
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                image_path = os.path.join(folder_path, filename)
                processed_face, _ = preprocess_face(image_path, is_path=True)

                if processed_face is None:
                    print(f"Không thể phát hiện khuôn mặt trong ảnh: {filename}")
                    continue

                vector = get_face_embedding(processed_face)

                if vector is not None:
                    face_vectors.append(vector)
                    print(f"Đã thêm ảnh vào vector chung: {filename}")

        if face_vectors:
            aggregated_vector = np.mean(face_vectors, axis=0)
            aggregated_vector = aggregated_vector / np.linalg.norm(aggregated_vector)
            face_ids.append(folder_name)
            face_database[folder_name] = aggregated_vector
            index.add(aggregated_vector.reshape(1, -1))
            print(f"Đã tổng hợp vector cho thư mục: {folder_name}")

def recognize_face(frame):
    """
    Nhận diện khuôn mặt trong frame từ camera
    """
    processed_face, face_coords = preprocess_face(frame, is_path=False)

    if processed_face is None:
        return None, frame

    face_embedding = get_face_embedding(processed_face)

    if face_embedding is None:
        return None, frame

    if face_coords is not None:
        x, y, w, h = face_coords
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if index.ntotal > 0:
        face_embedding = face_embedding.reshape(1, -1)
        D, I = index.search(face_embedding, 1)
        confidence = D[0][0]

        if confidence > 0.8:
            uid = face_ids[I[0][0]]
            print(f"Đã phát hiện khuôn mặt [{uid}] với độ tin cậy: {confidence}")
            cv2.putText(
                frame, uid, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

            return uid, frame
    return None, frame

register_faces("./face_images")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detected_face_id, processed_frame = recognize_face(frame)
    cv2.imshow("Face Recognition", processed_frame)
    if detected_face_id:
        print(f"Đã phát hiện khuôn mặt [{detected_face_id}]")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
