import cv2
import os
from deepface import DeepFace
import faiss
import numpy as np
import tempfile

# Hàm lấy tất cả các thư mục con bên trong một thư mục
def get_all_subfolders(folder):
    """
    Lấy danh sách tất cả các thư mục con bên trong một thư mục.
    """
    #os.listdir trả về danh sách các tệp và thư mục trong thư mục được chỉ định
    #kiểm tra nếu d là thư mục thì thêm vào danh sách subfolders
    #os.path.join nối đường dẫn thư mục với tên thư mục con
    subfolders = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    #trả về danh sách thư mục con
    return subfolders

# Kích thước vector đặc trưng
VECTOR_SIZE = 128

# Tạo FAISS index sử dụng metric cosine
index = faiss.IndexFlatIP(VECTOR_SIZE)

# Dictionary lưu trữ mapping từ face_id đến vector
face_database = {}
face_ids = []  # Danh sách các face_id theo thứ tự thêm vào index

# Tải bộ nhận diện khuôn mặt Haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Hàm kiểm tra thư mục chứa ảnh
def check_image_folders(folder_paths):
    """
    Kiểm tra danh sách thư mục có chứa ảnh không
    """
    #kiểm tra xem thư mục có tồn tại không và có phải là thư mục không
    valid_folders = []
    #duyệt qua các thư mục trong folder_paths
    for folder_path in folder_paths:
        #kiểm tra xem os.path.exists(folder_path) đường dẫn thư mục có tồn tại không
        # và os.path.isdir(folder_path) có phải là thư mục không
        # nếu không tồn tại hoặc không phải là thư mục thì bỏ qua
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print(f"Thư mục {folder_path} không tồn tại.")
            continue
        #os.listdir(folder_path) trả về danh sách các tệp và thư mục trong thư mục được chỉ định
        #kiểm tra f.lower().endswith(('.jpg', '.jpeg', '.png')) xem tệp có phải là ảnh không
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        #nếu không tìm thấy ảnh trong thư mục thì bỏ qua
        if not image_files:
            print(f"Không tìm thấy ảnh trong thư mục {folder_path}.")
            continue
        #nếu tìm thấy ảnh thì thêm thư mục vào valid_folders
        valid_folders.append(folder_path)
    #trả về danh sách thư mục chứa ảnh
    return valid_folders
# Hàm tiền xử lý khuôn mặt bằng OpenCV Haarcascade
def preprocess_face(img, is_path=False):
    """
    Hàm tiền xử lý khuôn mặt bằng OpenCV Haarcascade
    """
    if is_path:# Nếu đầu vào là đường dẫn ảnh
        img = cv2.imread(img)# Đọc ảnh từ đường dẫn
        if img is None:# Nếu không đọc được ảnh
            return None, None# Trả về None
    # Chuyển ảnh màu sang ảnh xám
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
        #chuyển đổi dữ liệu sang dạng float32 và 
        # /255.0 để chuẩn hóa dữ liệu về dạng 0-1 để giảm kích thước dữ liệu
        face = face.astype(np.float32) / 255.0
        #trả về khuôn mặt và tọa độ sau khi xử lý
        return face, (x, y, w, h)
    print("Không phát hiện khuôn mặt")
    return None, None
# Hàm trích xuất đặc trưng khuôn mặt từ ảnh
def get_face_embedding(face):
    """
    Lấy embedding vector từ DeepFace
    """
    # Tạo tệp tạm thời để lưu ảnh khuôn mặt với 
    # NamedTemporaryFile là hàm tạo tệp tạm thời
    # NamedTemporaryFile tạo tệp tạm thời với đuôi .jpg và không xóa tệp
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
        #lưu đường dẫn tệp tạm thời
        temp_path = tmpfile.name
        #chuyển đổi file ảnh từ dạng float32 về dạng uint8
        # khi chuyển đổi về dạng uint8 thì giá trị của ảnh sẽ nằm trong khoảng 0-255q
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
    #lấy vector đặc trưng từ embedding_result
    #embedding_result là một list chứa vector đặc trưng
    #dtype=np.float32 chuyển về dạng float32
    embedding = np.array(embedding_result[0]['embedding'], dtype=np.float32).flatten()
    #trả về vector đặc trưng đã được xử lý
    #np.linalg.norm(embedding) là chuẩn hóa vector
    return embedding / np.linalg.norm(embedding)  # Chuẩn hóa vector

# Hàm đăng ký khuôn mặt từ thư mục có sẵn
def register_faces(root_folder):
    """
    Đăng ký khuôn mặt từ một thư mục tổng chứa nhiều thư mục con
    """
    # Lấy tất cả các thư mục con trong thư mục gốc
    folder_paths = get_all_subfolders(root_folder)
    # Kiểm tra thư mục chứa ảnh và lấy thư mục chứa ảnh
    valid_folders = check_image_folders(folder_paths)
    # Sử dụng biến toàn cục global để lưu trữ index, face_database, face_ids
    global index, face_database, face_ids
    
    # Duyệt qua các thư mục chứa ảnh
    for folder_path in valid_folders:
        # face_vectors lưu trữ vector đặc trưng của khuôn mặt
        face_vectors = []
        # lấy tên thư mục chứa ảnh bằng cách lấy tên cuối cùng của đường dẫn
        # os.path.basename lấy tên cuối cùng của đường dẫn
        folder_name = os.path.basename(folder_path)
        # duyệt qua các file trong thư mục chứa ảnh
        # liệt kê các thư mục trong thư mục chứa ảnh
        for filename in os.listdir(folder_path):
            # lấy các file ảnh có đuôi .jpg
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
                    face_vectors.append(vector)
                    print(f"Đã thêm ảnh vào vector chung: {filename}")
        # kiểm tra xem face_vectors có phần tử không
        # nếu không rỗng thì tiếp tục phần dưới
        if face_vectors:
            # tính trung bình vector của các khuôn mặt theo từng chiều cột
            # np.mean(face_vectors, axis=0) tính trung bình theo cột
            aggregated_vector = np.mean(face_vectors, axis=0)
            # chuẩn hóa vector đưa về độ dài 1 để so sánh
            # np.linalg.norm tính độ dài vector
            # chia cho norm để chuẩn hóa vector
            aggregated_vector = aggregated_vector / np.linalg.norm(aggregated_vector)  # Chuẩn hóa
            #lưu tên thư mục vào face_ids
            face_ids.append(folder_name)
            #lưu vector vào face_database
            face_database[folder_name] = aggregated_vector
            # thêm vector vào chỉ mục tìm kiếm nhanh
            #chuyển hóa vector thành dạng ma trận 2D
            index.add(aggregated_vector.reshape(1, -1))
            print(f"Đã tổng hợp vector cho thư mục: {folder_name}")

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
    # index.ntotal là số lượng vector trong index 
    # nếu lớn hơn 0 thì tiếp tục chạy chương trình
    if index.ntotal > 0:
        face_embedding = face_embedding.reshape(1, -1)
        # tìm kiếm khoảng cách và chỉ mục từ vector đặc trưng
        # face_embedding là vector đặc trưng của khuôn mặt cần tìm
        D, I = index.search(face_embedding, 1) # K=1, tìm 1 vector gần nhất
        confidence = D[0][0]
        # Nếu khoảng cách nhỏ hơn 20 thì xác định là khuôn mặt đã đăng ký
        if confidence > 0.8:
            # Lấy face_id từ chỉ mục I
            uid = face_ids[I[0][0]]
            print(f"Đã phát hiện khuôn mặt [{uid}] với độ tin cậy: {confidence}")
            # Hiển thị thông tin khuôn mặt và độ tin cậy trên camera
            # cv2.FONT_HERSHEY_SIMPLEX là kiểu font chữ đơn giản của cv2
            cv2.putText(frame, uid, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Trả về face_id và frame đã xử lý
            return uid, frame
    return None, frame

# Đăng ký khuôn mặt từ thư mục tổng
# chưa chuyển qua mã unikey nên chỉ có thể lấy được thư mục không dấu
register_faces("./face_images")

# Mở camera và nhận diện khuôn mặt theo thời gian thực
cap = cv2.VideoCapture(0)
# Vòng lặp để nhận diện khuôn mặt từ camera
while True:# Vòng lặp vô hạn cho đến khi bị lỗi hoặc nhấn phím q
    # Đọc frame từ camera hàm cap lấy dữ liệu từ camera và đọc nó ra
    ret, frame = cap.read()
    if not ret:# Nếu không đọc được frame thì thoát khỏi vòng lặp
        break
    # Nhận diện khuôn mặt và hiển thị kết quả 
    # detection_face_id là id của khuôn mặt được nhận diện  
    # processed_frame là frame đã được xử lý 
    # hàm recognize_face để nhận diện khuân mặt từ frame
    detected_face_id, processed_frame = recognize_face(frame)
    # Hiển thị kết quả nhận diện khuôn mặt
    cv2.imshow("Face Recognition", processed_frame)
    if detected_face_id:
        print(f"Đã phát hiện khuôn mặt [{detected_face_id}]")  # Debug nhận diện
    # Nếu nhấn phím q thì thoát khỏi vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Giải phóng camera và đóng cửa sổ
cap.release()
# giải phóng bộ nhớ và đóng cửa sổ
cv2.destroyAllWindows()