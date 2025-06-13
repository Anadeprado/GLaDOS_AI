import cv2

# Configuración de la detección
FRAME_SKIP = 20 # Sólo detectar cada X frames para mejorar rendimiento
CONFIDENCE_THRESHOLD = 0.5

# Cargar clases desde coco.names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Cargar red YOLOv4-Tiny
net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # Usar CPU
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Captura de video
cap = cv2.VideoCapture(0)

frame_count = 0
boxes = []
person_box = None

# Función para dibujar un overlay de objetivo en la imagen
def draw_target_overlay(frame, center_x, center_y):
    # Parámetros de color y grosor
    color = (0, 0, 255)  # Rojo en BGR
    thickness = 2

    # Círculos concéntricos
    # for radius in [15, 30, 45]:
    for radius in [10, 20, 30]:
        cv2.circle(frame, (center_x, center_y), radius, color, thickness)

    # Línea vertical
    cv2.line(frame, (center_x, center_y - 50), (center_x, center_y + 50), color, thickness)
    # Línea horizontal
    cv2.line(frame, (center_x - 50, center_y), (center_x + 50, center_y), color, thickness)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Procesar cada FRAME_SKIP frames
    if frame_count % FRAME_SKIP == 0:
        height, width = frame.shape[:2]
        person_box = None
        center_x = width // 2
        center_y = height // 2

        # Preprocesamiento
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(scores.argmax())
                confidence = scores[class_id]

                if classes[class_id] == "person" and confidence > CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    person_box = (x, y, w, h)
                    break  # Detiene el bucle tras la primera persona detectada
            if person_box:
                break  # Sal de los outputs

    # Dibuja el cuadro si hay una persona detectada
    if person_box:
        x, y, w, h = person_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "STUPID HUMAN", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Dibuja el objetivo de la cámara (centro de la imagen o centro de la persona si se ha detectado). Pinta un punto rojo 
    # cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
    draw_target_overlay(frame, center_x, center_y)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()