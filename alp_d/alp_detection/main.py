from ultralytics import YOLO  # Importeer YOLO model voor objectdetectie
import cv2  # OpenCV voor videobewerking en beeldverwerking
import numpy as np  # Voor numerieke bewerkingen
from sort.sort import Sort  # Importeren van SORT tracker voor object tracking
from util import get_car, read_license_plate, write_csv  # Importeren van hulpmiddelen zoals functie om auto's te krijgen, OCR te doen, en CSV te schrijven

results = {}  # Dictionary om resultaten per frame op te slaan

mot_tracker = Sort()  # Initialiseren van de object tracker (SORT)

# Laad de modellen
coco_model = YOLO('yolov8n.pt')  # Laad het COCO model voor voertuigen detectie
license_plate_detector = YOLO('alp_detection\\license_plate_detector.pt')  # Laad het nummerplaat detectiemodel

# Laad de video
cap = cv2.VideoCapture('alp_detection\\sample.mp4')  # Open de video bestand voor verwerking

# Controleer of de video succesvol is geopend
if not cap.isOpened():
    print("Error: Could not open video stream.")  # Als de video niet geopend kan worden, geef foutmelding
    exit()

# Lijst met voertuigenklassen (uit COCO-model)
vehicles = [2, 3, 5, 7]  # Auto, motor, bus, vrachtwagen klassen uit COCO (2=car, 3=motorcycle, 5=bus, 7=truck)

# Lees de frames van de video
frame_nmr = -1  # Begin met frame nummer -1
ret = True  # Standaardwaarde voor video leesresultaten
while ret:
    frame_nmr += 1  # Verhoog het frame nummer
    ret, frame = cap.read()  # Lees het volgende frame van de video

    if ret:  # Als er een frame is gelezen
        results[frame_nmr] = {}  # Maak een lege dictionary voor de resultaten van dit frame

        # Detecteer voertuigen met het COCO-model
        detections = coco_model(frame)[0]  # Voer de detectie uit op het frame
        detections_ = []  # Lijst om gedetecteerde voertuigen op te slaan
        for detection in detections.boxes.data.tolist():  # Loop door alle gedetecteerde objecten
            x1, y1, x2, y2, score, class_id = detection  # Verkrijg de coördinaten van de bounding box en andere informatie
            if int(class_id) in vehicles:  # Als het object een voertuig is
                detections_.append([x1, y1, x2, y2, score])  # Voeg de detectie toe aan de lijst

        # Volg de voertuigen met behulp van de SORT tracker
        track_ids = mot_tracker.update(np.asarray(detections_))  # Verkrijg de object ID's van de getrackte voertuigen

        # Detecteer nummerplaten met het nummerplaat model
        license_plates = license_plate_detector(frame)[0]  # Detecteer nummerplaten in het frame
        for license_plate in license_plates.boxes.data.tolist():  # Loop door gedetecteerde nummerplaten
            x1, y1, x2, y2, score, class_id = license_plate  # Verkrijg de coördinaten van de nummerplaat

            # Verkrijg de bijpassende voertuig door de get_car functie te gebruiken
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:  # Als er een voertuig gevonden is
                # Snijd de nummerplaat uit het frame
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                # Verwerk de gesneden nummerplaat voor OCR (optische tekenherkenning)
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)  # Zet het beeld om naar grijswaarden
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)  # Threshold voor binarizeren van het beeld

                # Lees de nummerplaat tekst met OCR
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:  # Als er tekst is herkend
                    # Sla de resultaten voor dit frame op in de dictionary
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},  # Voeg voertuig bbox toe
                        'license_plate': {'bbox': [x1, y1, x2, y2],  # Voeg nummerplaat bbox toe
                                          'text': license_plate_text,  # Voeg nummerplaat tekst toe
                                          'bbox_score': score,  # Voeg nummerplaat detectie score toe
                                          'text_score': license_plate_text_score}  # Voeg OCR score toe
                    }

        # Stop de video weergave als de gebruiker 'q' indrukt
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Schrijf de resultaten naar een CSV bestand
write_csv(results, 'alp_detection\\test.csv')  # Sla de resultaten op in een CSV-bestand

import add_missing_data  # Importeer de module voor het toevoegen van ontbrekende data
import visualize  # Importeer de module voor visualisatie
