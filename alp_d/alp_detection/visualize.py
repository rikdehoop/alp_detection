import ast
import cv2
import numpy as np
import pandas as pd

# Functie om een rand te tekenen rondom een object (bijv. auto of nummerplaat)
def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Teken lijnen voor de bovenkant en linkerzijde
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    # Teken lijnen voor de onderkant en linkerzijde
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    # Teken lijnen voor de bovenkant en rechterzijde
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    # Teken lijnen voor de onderkant en rechterzijde
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


# Lees de resultaten van de CSV in
results = pd.read_csv('alp_detection//test_interpolated.csv')

# Laad de video
video_path = 'alp_detection//sample.mp4'
cap = cv2.VideoCapture(video_path)

# Definieer het codec en video-instellingen
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec specificatie
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per seconde
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Breedte van de video
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Hoogte van de video
out = cv2.VideoWriter('alp_detection//out_video.mp4', fourcc, fps, (width, height))

# Dicteer waar de nummerplaten en crops voor elke auto worden opgeslagen
license_plate = {}

# Loop door de unieke auto ID's in de resultaten
for car_id in np.unique(results['car_id']):
    # Zoek de hoogste score van het nummerbord voor deze auto
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    license_plate[car_id] = {'license_crop': None,
                             'license_plate_number': results[(results['car_id'] == car_id) & 
                                                             (results['license_number_score'] == max_)]['license_number'].iloc[0]}

    # Zet de video naar het frame waar het hoogste score-nummerbord werd gedetecteerd
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) & 
                                             (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
    ret, frame = cap.read()

    # Haal de coördinaten van de nummerplaat uit de resultaten
    x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) & 
                                              (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

    # Snijd het nummerbord uit het frame
    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    # Bewaar het nummerplaat-crop voor later gebruik
    license_plate[car_id]['license_crop'] = license_crop


frame_nmr = -1

# Zet de video terug naar het begin
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Lees alle frames van de video
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            # Teken de auto-bounding box
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                        line_length_x=200, line_length_y=200)

            # Teken de nummerplaat-bounding box
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

            # Haal het gecropte nummerbord
            license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

            H, W, _ = license_crop.shape

            try:
                # Plaats het gecropte nummerbord op de frame
                frame[int(car_y1) - H - 100:int(car_y1) - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                # Zet een witte balk boven het nummerbord
                frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                      int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                # Haal de grootte van de tekst om de nummerplaat te tekenen
                (text_width, text_height), _ = cv2.getTextSize(
                    license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4.3,
                    17)

                # Zet de nummerplaattekst op de frame
                cv2.putText(frame,
                            license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                            (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4.3,
                            (0, 0, 0),
                            17)

            except:
                pass

        # Schrijf het frame naar de uitvoervideo
        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))

        # Laat het frame weergeven als je wilt
        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)

# Sluit de video-bestanden na afloop
out.release()
cap.release()
