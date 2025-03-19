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
    # Tạo tệp tạm thời để lưu ảnh khuôn mặt với 
    # NamedTemporaryFile là hàm tạo tệp tạm thời
    # NamedTemporaryFile tạo tệp tạm thời với đuôi .jpg và không xóa tệp
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
        # Lưu đường dẫn tệp tạm thời
        temp_path = tmpfile.name
        #chuyển đổi file ảnh từ dạng float32 về dạng uint8
        # khi chuyển đổi về dạng uint8 thì giá trị của ảnh sẽ nằm trong khoảng 0-255
        #cần chuyển về dạng uint8 vì opencv chỉ nhận ảnh dạng uint8 sau đó lưu ảnh vào tệp tạm thời temp_path
        cv2.imwrite(temp_path, (face * 255).astype(np.uint8))
        #sử dụng hàm represent trong thư viện deepface để lấy vector đặc trưng từ ảnh
        #tham số img_path: đường dẫn ảnh
        #  model_name: tên mô hình, ở đây sử dụng mô hình Facenet 
        #  enforce_detection: kiểm tra khuôn mặt
        #  deepfacce sẽ trả về 1 vector đặc trưng khi không tìm thấy khuân mặt nếu enforce_detection=False
        embedding_result = DeepFace.represent(img_path=temp_path, model_name="Facenet", enforce_detection=False)
        #xóa tệp tạm thời khi đã lấy được đặc trưng khuôn mặt
        os.remove(temp_path)
        #trả về vector đặc trưng dưới dạng mảng numpy
        #embedding_result[0]['embedding'] là vector đặc trưng được trích xuất từ ảnh nhờ deepface
        #dtype=np.float32 chuyển về dạng float32 để tiết kiệm bộ nhớ
        #flatten() chuyển mảng nhiều chiều về mảng 1 chiều
        return np.array(embedding_result[0]['embedding'], dtype=np.float32).flatten()
    #nếu không lấy được vector đặc trưng thì trả về None
    return None

# Hàm đăng ký khuôn mặt từ thư mục có sẵn
def register_face(folder_path):
    """
    Đăng ký khuôn mặt từ thư mục chứa ảnh và thêm vào FAISS index
    """
    # Sử dụng biến toàn cục global để lưu trữ index, face_database, face_ids
    global index, face_database, face_ids
    # Duyệt qua các file trong thư mục chứa ảnh
    face_vectors = []
    # duyệt qua các file trong thư mục chứa ảnh
    # liệt kê các thư mục trong thư mục chứa ảnh
    for filename in os.listdir(folder_path):
        # lấy các file có đuôi là .jpg
        if filename.endswith(".jpg"):
            # lấy đường dẫn ảnh bằng cách ghép tên tệp với đường dẫn thư mục
            image_path = os.path.join(folder_path, filename)
            # processed_face đọc ảnh và xử lý ảnh
            # image_path là 1 chuỗi chứa đường dẫn ảnh
            # is_path=True để xác định đường dẫn ảnh chạy lại đoạn code trên với cv2.imread để đọc ảnh
            processed_face, _ = preprocess_face(image_path, is_path=True)
            # nếu không phát hiện khuôn mặt thì bỏ qua
            if processed_face is None:
                print(f"Không thể phát hiện khuôn mặt trong ảnh: {filename}")
                continue
            # lấy vector đặc trưng từ khuôn mặt
            vector = get_face_embedding(processed_face)
            # nếu vector là none thì có nghĩa là không thể lấy được ảnh nên bỏ qua 
            # nếu vector không none thì tiếp tục phần dưới
            if vector is not None:
                # tạo id cho face_id
                # uuid.uuid4() tạo 1 id ngẫu nhiên dạng chuỗi đêr làm key
                uid = str(uuid.uuid4())
                # thêm id vào face_ids giúp theo dõi thứ tự khuân mặt đã đang ký
                face_ids.append(uid)
                # lưu vector vào face_database với key là uid
                face_database[uid] = vector
                # thêm vector vào danh sách face_vectors
                face_vectors.append(vector)
                # in ra thông báo đã đăng ký khuôn mặt từ ảnh
                print(f"Đã đăng ký khuôn mặt từ ảnh: {filename}, FaceID: {uid}")
    # kiểm tra xem face_vectors có phần tử không
    # nếu không rỗng thì tiếp tục phần dưới
    if face_vectors:
        # chuyển face_vectors thành mảng numpy
        # với kiểu dữ liệu là float32 để tối ưu hiệu xuất tính toán
        face_vectors = np.array(face_vectors, dtype=np.float32)
        # thêm face_vectors vào index
        # index là 1 faiss index giúp tìm kiếm vector đặc trưng nhanh chóng
        index.add(face_vectors)
        
# Hàm nhận diện khuôn mặt từ camera
def recognize_face(frame):
    """
    Nhận diện khuôn mặt trong frame từ camera
    """
    # Tiền xử lý khuôn mặt từ frame
    # preprocess_face là ảnh đã được xử lý
    # face_coords là tọa độ khuôn mặt
    processed_face, face_coords = preprocess_face(frame, is_path=False)
    # Nếu không tìm thấy khuân mặt trong đầu vào thì trả về None
    if processed_face is None:
        return None, frame
    # Lấy vector đặc trưng từ khuôn mặt
    face_embedding = get_face_embedding(processed_face)
    # Nếu không lấy được vector đặc trưng thì trả về None
    if face_embedding is None:
        return None, frame
    # kiểm tra xem khuôn mặt được phát hiện hay không
    if face_coords is not None:
        # tọa độ khuôn mặt được truyền vào face_coords
        x, y, w, h = face_coords
        # vẽ hình chữ nhật xung quanh khuôn mặt 
        # x, y là tọa độ góc trên bên trái 
        # x+w y+h là tọa độ góc dưới bên phải
        # (0, 255, 0) là màu của hình chữ nhật
        # 2 là độ dày của hình chữ nhật
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Tìm kiếm khuôn mặt gần nhất trong index
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
