
import cv2
import numpy as np
import serial
import time
from collections import deque
import torch

class AD5204Controller:
    def __init__(self, serial_port='COM3', baud_rate=9600):
        # Configuration série Arduino
        try:
            self.arduino = serial.Serial(serial_port, baud_rate, timeout=1)
            time.sleep(2)  # Attendre la connexion
            print(f"Connecté à Arduino sur {serial_port}")
        except:
            print("Arduino non connecté - mode simulation")
            self.arduino = None
        
        # Charger le modèle YOLOv5 pour la détection de personnes
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.classes = [0]  # Seulement la classe 'person'
        
        # Définir les zones (coordonnées normalisées 0-1)
        self.zones = {
            'Z1': (0.0, 0.0, 0.5, 0.5),    
            'Z2': (0.5, 0.0, 1.0, 0.5),    
            'Z3': (0.0, 0.5, 0.5, 1.0),    
            'Z4': (0.5, 0.5, 1.0, 1.0)     
        }
        
        # Historique pour lisser les valeurs
        self.history = {zone: deque(maxlen=10) for zone in self.zones}
        
    def is_in_zone(self, x_center, y_center, zone_coords):
        x1, y1, x2, y2 = zone_coords
        return x1 <= x_center <= x2 and y1 <= y_center <= y2
    
    def calculate_resistance(self, person_count, total_people):
        """
        Calcule la valeur de résistance pour l'AD5204
        Plus il y a de personnes, plus la résistance est faible
        """
        if total_people == 0:
            return 255  # Résistance maximale (10kΩ)
        
        # Résistance de base basée sur le nombre de personnes
        # Moins de personnes = plus de résistance
        base_resistance = max(0, 255 - (person_count * 85))  # 85 = 255/3
        
        # Ajuster selon la distribution
        # Plus la zone est concentrée, plus la résistance baisse
        if total_people > 0:
            concentration_factor = person_count / total_people
            adjusted_resistance = int(base_resistance * (1.0 - 0.3 * concentration_factor))
        else:
            adjusted_resistance = base_resistance
        
        return max(0, min(255, adjusted_resistance))
    
    def process_frame(self, frame):
        # Détection des personnes avec YOLOv5
        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()
        
        # Compter les personnes dans chaque zone
        zone_counts = {zone: 0 for zone in self.zones}
        total_people = len(detections)
        
        for detection in detections:
            if detection[4] > 0.5:  # Seuil de confiance
                x_center = (detection[0] + detection[2]) / 2 / frame.shape[1]
                y_center = (detection[1] + detection[3]) / 2 / frame.shape[0]
                
                for zone, coords in self.zones.items():
                    if self.is_in_zone(x_center, y_center, coords):
                        zone_counts[zone] += 1
                        break
        
        # Calculer la résistance pour chaque zone
        resistance_values = {}
        for zone, count in zone_counts.items():
            resistance = self.calculate_resistance(count, total_people)
            self.history[zone].append(resistance)
            
            # Moyenne mobile pour lisser les variations
            smoothed_resistance = int(np.mean(list(self.history[zone])))
            resistance_values[zone] = smoothed_resistance
        
        return resistance_values, zone_counts, detections
    
    def send_to_arduino(self, resistance_values):
        if self.arduino:
            command = f"Z1:{resistance_values['Z1']},Z2:{resistance_values['Z2']},Z3:{resistance_values['Z3']},Z4:{resistance_values['Z4']}\n"
            self.arduino.write(command.encode())
            print(f"Envoyé à Arduino: {command.strip()}")
    
    def draw_zones_and_info(self, frame, resistance_values, zone_counts, detections):
        # Dessiner les zones
        h, w = frame.shape[:2]
        for zone, coords in self.zones.items():
            x1, y1, x2, y2 = coords
            pt1 = (int(x1 * w), int(y1 * h))
            pt2 = (int(x2 * w), int(y2 * h))
            
            # Couleur basée sur la résistance (inversée pour l'affichage)
            resistance = resistance_values[zone]
            intensity = 255 - resistance  # Pour l'affichage visuel
            color = (0, intensity, 0)  # Vert plus intense = moins de résistance
            
            cv2.rectangle(frame, pt1, pt2, color, 2)
            cv2.putText(frame, f"{zone}: {zone_counts[zone]}p R:{resistance}", 
                       (pt1[0] + 10, pt1[1] + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Dessiner les détections
        for detection in detections:
            if detection[4] > 0.5:
                x1, y1, x2, y2 = map(int, detection[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{detection[4]:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Afficher le nombre total de personnes
        cv2.putText(frame, f"Total: {len(detections)} personnes", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(0)  # Caméra par défaut
        
        if not cap.isOpened():
            print("Erreur: Impossible d'ouvrir la caméra")
            return
        
        print("Démarrage de la détection... Appuyez sur 'q' pour quitter")
        print("Résistance: 255 = max (10kΩ), 0 = min (∼0Ω)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Traiter l'image
            resistance_values, zone_counts, detections = self.process_frame(frame)
            
            # Envoyer les données à Arduino
            self.send_to_arduino(resistance_values)
            
            # Dessiner les informations sur l'image
            frame = self.draw_zones_and_info(frame, resistance_values, zone_counts, detections)
            
            # Afficher l'image
            cv2.imshow('AD5204 Zone Controller', frame)
            
            # Quitter avec la touche 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Remettre les résistances au maximum à la fin
        if self.arduino:
            self.arduino.write(b"Z1:255,Z2:255,Z3:255,Z4:255\n")
        
        cap.release()
        cv2.destroyAllWindows()
        if self.arduino:
            self.arduino.close()

# Installation required packages:
# pip install opencv-python torch torchvision serial

if __name__ == "__main__":
    # Modifier le port série selon votre configuration
    controller = AD5204Controller(serial_port='COM3')  # Linux: '/dev/ttyUSB0', Mac: '/dev/tty.usbmodemXXX'
    controller.run()