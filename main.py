import cv2
import dlib

# Inisialisasi detector dan predictor untuk deteksi landmark
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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
        
        # Tampilkan garis titik untuk deteksi wajah
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Menampilkan lekuk wajah
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            # Tampilkan garis menghubungkan landmark
            if i < 16:
                cv2.line(frame, (landmarks.part(i).x, landmarks.part(i).y),
                         (landmarks.part(i + 1).x, landmarks.part(i + 1).y), (255, 0, 0), 2)
            elif i == 16:
                cv2.line(frame, (landmarks.part(i).x, landmarks.part(i).y),
                         (landmarks.part(0).x, landmarks.part(0).y), (255, 0, 0), 2)
            elif i < 26 or (i > 29 and i < 36):
                cv2.line(frame, (landmarks.part(i).x, landmarks.part(i).y),
                         (landmarks.part(i + 1).x, landmarks.part(i + 1).y), (255, 0, 0), 2)
            elif i == 26 or i == 35:
                cv2.line(frame, (landmarks.part(i).x, landmarks.part(i).y),
                         (landmarks.part(29).x, landmarks.part(29).y), (255, 0, 0), 2)
            elif i < 48:
                cv2.line(frame, (landmarks.part(i).x, landmarks.part(i).y),
                         (landmarks.part(i + 1).x, landmarks.part(i + 1).y), (255, 0, 0), 2)
            elif i == 48:
                cv2.line(frame, (landmarks.part(i).x, landmarks.part(i).y),
                         (landmarks.part(60).x, landmarks.part(60).y), (255, 0, 0), 2)
            elif i < 60:
                cv2.line(frame, (landmarks.part(i).x, landmarks.part(i).y),
                         (landmarks.part(i + 1).x, landmarks.part(i + 1).y), (255, 0, 0), 2)
            elif i == 60:
                cv2.line(frame, (landmarks.part(i).x, landmarks.part(i).y),
                         (landmarks.part(48).x, landmarks.part(48).y), (255, 0, 0), 2)

        # Menghitung rasio tinggi alis terhadap tinggi mata
        eyebrow_height_ratio = ((landmarks.part(21).y - landmarks.part(17).y + landmarks.part(26).y - landmarks.part(22).y) /
                                (2 * (landmarks.part(24).x - landmarks.part(20).x)))
        
        # Menghitung rasio tinggi mulut terhadap tinggi wajah
        mouth_height_ratio = ((landmarks.part(54).y - landmarks.part(48).y) / (face.bottom() - face.top()))

        # Menghitung rasio tinggi hidung terhadap tinggi wajah
        nose_height_ratio = ((landmarks.part(27).y - landmarks.part(33).y) / (face.bottom() - face.top()))

        # Menghitung rasio lebar mulut terhadap lebar wajah
        mouth_width_ratio = ((landmarks.part(54).x - landmarks.part(48).x) / (landmarks.part(12).x - landmarks.part(4).x))

        # Deteksi ekspresi wajah
        expression = ""
        if eyebrow_height_ratio > 0.06:
            expression = 'Terkejut!'
        elif mouth_height_ratio > 0.15:
            expression = 'Senang!'
        elif eyebrow_height_ratio < 0.02 and mouth_height_ratio < 0.1:
            expression = 'Sedih!'
        elif mouth_width_ratio > 0.3:
            expression = 'Marah!'
        else:
            expression = 'Netral'

        # Tampilkan hasil
        cv2.putText(frame, expression, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Face Detection with Expression', frame)

    # Hentikan loop jika pengguna menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup webcam atau video setelah selesai
cap.release()
cv2.destroyAllWindows()
