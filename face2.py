import cv2
import os
from deepface import DeepFace
import faiss
import uuid
import numpy as np
import tempfile

# Kích thước vector đặc trưng
VECTOR_SIZE = 128

# Tạo FAISS index sử dụng khoảng cách L2
index = faiss.IndexFlatL2(VECTOR_SIZE)

# Dictionary lưu trữ mapping từ face_id đến vector
face_database = {}
face_ids = []  # Danh sách các face_id theo thứ tự thêm vào index

# Tải bộ nhận diện khuôn mặt Haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Hàm tiền xử lý khuôn mặt
def preprocess_face(img, is_path=False):
    """
    Hàm tiền xử lý khuôn mặt bằng OpenCV Haarcascade
    """

    if is_path: # Nếu là đường dẫn ảnh
        img = cv2.imread(img) # Đọc ảnh từ đường dẫn
        if img is None: # Nếu không đọc được ảnh
            return None, None # Trả về None
    # Chuyển ảnh sang ảnh xám 
    # ## sử dụng ảnh xám để tăng tốc độ xử lý, 
    # giảm kích thước dữ liệu, dùng thư viện cv2 không thể sử dụng ảnh màu
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # ##Hàm detectMultiScale để phát hiện khuôn mặt trong ảnh, nguồn gốc cv2
    # scaleFactor: hệ số giảm kích thước ảnh, minNeighbors: số lần phát hiện khuôn mặt xung quanh 1 khuôn mặt
    # minSize: kích thước tối thiểu của khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Vòng lặp qua các khuôn mặt phát hiện được
    # x y w h là tọa độ và kích thước của khuôn mặt
    # x y là tọa độ 2 bên trái phải, w h là chiều rộng và chiều cao
    for (x, y, w, h) in faces: 
        # cắt khuân mặt từ ảnh được phát hiện từ ảnh gốc
        # chứa khuân mặt được phát hiện từ detectMultiScale trước đó
        face = img[y:y+h, x:x+w]
        #kiểm tra xem khuôn mặt có tồn tại không bằng cách kiểm tra x.y.size có bằng 0 không
        if face.size == 0:
            #nếu không tồn tại thì bỏ qua
            continue
        #resize khuôn mặt về kích thước 224x224 vì 224x224 là kích thước chuẩn của nhiều mô hình
        #lấy từ thư viên cv2
        face = cv2.resize(face, (224, 224))
        #chuyển đổi dữ liệu sang dạng float32 và /255.0 để chuẩn hóa dữ liệu về dạng 0-1 để giảm kích thước dữ liệu
        face = face.astype(np.float32) / 255.0
        #trả về khuôn mặt và tọa độ
        return face, (x, y, w, h)
    print("Không phát hiện khuôn mặt")#nếu không phát hiện khuôn mặt thì in ra dòng này
    return None, None #trả về None nếu không phát hiện khuôn mặt
# Hàm trích xuất đặc trưng khuôn mặt từ ảnh
def get_face_embedding(face):
    """
    Lấy embedding vector từ DeepFace
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
        temp_path = tmpfile.name
        cv2.imwrite(temp_path, (face * 255).astype(np.uint8))
    
    try:
        embedding_result = DeepFace.represent(img_path=temp_path, model_name="Facenet", enforce_detection=False)
        os.remove(temp_path)
        return np.array(embedding_result[0]['embedding'], dtype=np.float32).flatten()
    except Exception as e:
        print(f"Lỗi khi trích xuất đặc trưng khuôn mặt: {e}")
        return None

def register_face(folder_path):
    """
    Đăng ký khuôn mặt từ thư mục chứa ảnh và thêm vào FAISS index
    """
    global index, face_database, face_ids
    face_vectors = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            processed_face, _ = preprocess_face(image_path, is_path=True)
            if processed_face is None:
                print(f"Không thể phát hiện khuôn mặt trong ảnh: {filename}")
                continue
            vector = get_face_embedding(processed_face)
            if vector is not None:
                uid = str(uuid.uuid4())
                face_ids.append(uid)
                face_database[uid] = vector
                face_vectors.append(vector)
                print(f"Đã đăng ký khuôn mặt từ ảnh: {filename}, FaceID: {uid}")
    if face_vectors:
        face_vectors = np.array(face_vectors, dtype=np.float32)
        index.add(face_vectors)
# Hàm nhận diện khuôn mặt từ camera
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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    if index.ntotal > 0:
        D, I = index.search(face_embedding.reshape(1, -1), 1)
        print(f"Kết quả tìm kiếm FAISS: Khoảng cách={D[0][0]}, Chỉ mục={I[0][0]}")  # Debug
        if D[0][0] < 20:
            uid = face_ids[I[0][0]]
            confidence = 1 - (D[0][0] / 20)
            cv2.putText(frame, f"FaceID: {uid} Conf: {confidence:.2f}", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return uid, frame
    return None, frame

# Đăng ký khuôn mặt từ thư mục có sẵn
register_face("./face_images")

# Mở camera và nhận diện khuôn mặt theo thời gian thực
cap = cv2.VideoCapture(0)
# Vòng lặp để nhận diện khuôn mặt từ camera
while True: # Vòng lặp vô hạn cho đến khi bị lỗi hoặc nhấn phím q
    # Đọc frame từ camera hàm cap lấy dữ liệu từ camera và đọc nó ra
    ret, frame = cap.read()
    if not ret: # Nếu không đọc được frame thì thoát khỏi vòng lặp
        break
    # Nhận diện khuôn mặt và hiển thị kết quả 
    # detection_face_id là id của khuôn mặt được nhận diện  
    # processed_frame là frame đã được xử lý 
    # hàm recognize_face để nhận diện khuân mặt từ frame
    detected_face_id, processed_frame = recognize_face(frame)
    cv2.imshow("Face Recognition", processed_frame)
    print(f"Face detected: {detected_face_id}")  # Debug nhận diện
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
