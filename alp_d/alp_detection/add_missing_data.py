import csv
import numpy as np
from scipy.interpolate import interp1d

# Functie om de ontbrekende frames tussen gedetecteerde frames in te vullen door middel van interpolatie
def interpolate_bounding_boxes(data):
    # Haal de benodigde gegevenskolommen uit de inputdata
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])  # Frame nummers
    car_ids = np.array([int(float(row['car_id'])) for row in data])  # Auto ID's
    
    # Zet de bounding box strings om naar lijsten van floats
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])  # Auto's bounding box
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])  # Nummerplaat bounding box

    interpolated_data = []  # Lijst om de geïnterpoleerde data op te slaan
    unique_car_ids = np.unique(car_ids)  # Verkrijg unieke auto ID's
    
    for car_id in unique_car_ids:  # Verwerk elke unieke auto ID afzonderlijk
        
        # Verkrijg de frame nummers voor de huidige auto ID
        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]
        print(frame_numbers_, car_id)  # Debugging print statement

        # Maak een masker om data voor deze specifieke auto ID te filteren
        car_mask = car_ids == car_id  # Boolean masker om auto's te filteren
        car_frame_numbers = frame_numbers[car_mask]  # Haal de frame nummers voor deze auto
        
        # Lijsten om geïnterpoleerde bounding boxes op te slaan
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]  # Eerste gedetecteerde frame
        last_frame_number = car_frame_numbers[-1]  # Laatste gedetecteerde frame

        # Itereer door de gedetecteerde frames en vul ontbrekende frames in met interpolatie
        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]  # Het vorige frame nummer
                prev_car_bbox = car_bboxes_interpolated[-1]  # Het vorige auto-bounding box
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]  # Het vorige nummerplaat-bounding box

                # Als er een gap is tussen de frames, interpolateer dan de ontbrekende bounding boxes
                if frame_number - prev_frame_number > 1:
                    frames_gap = frame_number - prev_frame_number  # Het aantal ontbrekende frames
                    x = np.array([prev_frame_number, frame_number])  # De bestaande frames
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)  # Nieuwe frames voor interpolatie
                    
                    # Interpoleer de auto bounding boxes
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    
                    # Interpoleer de nummerplaat bounding boxes
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])  # Voeg geïnterpoleerde auto bounding boxes toe
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])  # Voeg geïnterpoleerde nummerplaat bounding boxes toe

            # Voeg de originele bounding boxes toe voor het huidige frame
            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        # Sla de geïnterpoleerde resultaten op
        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i  # Huidig frame nummer
            row = {
                'frame_nmr': str(frame_number),
                'car_id': str(car_id),
                'car_bbox': ' '.join(map(str, car_bboxes_interpolated[i])),  # Zet de bounding box om naar een string
                'license_plate_bbox': ' '.join(map(str, license_plate_bboxes_interpolated[i]))  # Zet de nummerplaat-bounding box om naar een string
            }

            # Als het frame geïnterpoleerd is, zet dan de standaardwaarden voor ontbrekende velden
            if str(frame_number) not in frame_numbers_:
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                # Haal de originele data op voor bestaande frames
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
                row['license_plate_bbox_score'] = original_row.get('license_plate_bbox_score', '0')
                row['license_number'] = original_row.get('license_number', '0')
                row['license_number_score'] = original_row.get('license_number_score', '0')

            interpolated_data.append(row)  # Voeg de rij toe aan de geïnterpoleerde data

    return interpolated_data  # Retourneer de geïnterpoleerde data

# Laad het CSV-bestand
with open('alp_detection//test.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)  # Zet de CSV om naar een lijst van rijen

# Interpoleer de ontbrekende data
interpolated_data = interpolate_bounding_boxes(data)

# Schrijf de bijgewerkte data naar een nieuw CSV-bestand
header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
with open('alp_detection//test_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()  # Schrijf de header
    writer.writerows(interpolated_data)  # Schrijf de geïnterpoleerde gegevens
