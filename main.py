import cv2
import dlib

# Inisialisasi detector dan predictor untuk deteksi landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("path/to/shape_predictor_68_face_landmarks.dat")


# Inisialisasi webcam atau baca video dari file
cap = cv2.VideoCapture(0)  # Ganti angka 0 dengan alamat video jika Anda menggunakan file video

while True:
    # Baca frame dari webcam atau video
    ret, frame = cap.read()

    # Konversi frame ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah dalam frame
    faces = detector(gray)
    
    # Iterasi melalui wajah yang terdeteksi
    for face in faces:
        # Dapatkan landmark wajah
        landmarks = predictor(gray, face)
        
        # Gambar titik landmark di wajah
        for n in range(68):  # 68 merupakan jumlah landmark pada model ini
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        
        # Deteksi ekspresi bahagia
        left_eye = (landmarks.part(36).x, landmarks.part(36).y, landmarks.part(39).x, landmarks.part(39).y)
        right_eye = (landmarks.part(42).x, landmarks.part(42).y, landmarks.part(45).x, landmarks.part(45).y)
        mouth = (landmarks.part(48).x, landmarks.part(48).y, landmarks.part(54).x, landmarks.part(54).y)
        
        # Menghitung rasio lebar mata terhadap tinggi mata
        eye_aspect_ratio = ((right_eye[2] - right_eye[0] + left_eye[2] - left_eye[0]) /
                            (2 * (right_eye[3] - right_eye[1] + left_eye[3] - left_eye[1])))
        
        # Menghitung rasio lebar mulut terhadap tinggi mulut
        mouth_aspect_ratio = ((mouth[2] - mouth[0]) / (mouth[3] - mouth[1]))
        
        # Deteksi wajah bahagia jika rasio lebar mata tinggi mata cukup besar dan rasio lebar mulut tinggi mulut cukup besar
        if eye_aspect_ratio > 0.25 and mouth_aspect_ratio > 0.4:
            cv2.putText(frame, 'Bahagia!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Tampilkan hasil
    cv2.imshow('Face Detection with Expression', frame)

    # Hentikan loop jika pengguna menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup webcam atau video setelah selesai
cap.release()
cv2.destroyAllWindows()
