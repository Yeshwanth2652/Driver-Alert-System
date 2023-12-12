# Driver-Alert-System
Driver Alert System using Python
Overview:
Developed a Driver Alert System using Python, integrating computer vision techniques and a Convolutional Neural Network (CNN) to detect driver fatigue. The system employed Haar Cascade classifiers to identify facial features, particularly the eyes, and relayed coordinates to a CNN model for real-time detection of closed eyes.

Functionalities:
Facial and Eye Detection: Implemented Haar Cascade classifiers to accurately detect and track the driver's face and eyes.
CNN Integration: Utilized a Convolutional Neural Network to analyze eye states, distinguishing between open and closed eyes.
Fatigue Detection and Alert Mechanism: Monitored eye closure instances and incremented a score; upon reaching a threshold (e.g., score above 15), triggered an audible alert (beep sound) to notify driver fatigue.
Key Contributions:
Computer Vision Implementation: Orchestrated the integration of Haar Cascade classifiers for facial and eye detection, enabling precise tracking.
CNN Model Development: Engineered and trained the CNN model to recognize eye states, enabling real-time fatigue detection.
Alert Mechanism Implementation: Implemented the scoring system and alert trigger for timely driver intervention.
Achievements:
Efficient Real-time Detection: Successfully achieved real-time detection of eye closure, enhancing driver safety.
Functional Alert System: Developed a reliable alert mechanism contributing to proactive fatigue detection and prevention.
Technologies Used:
Python, OpenCV, Haar Cascade Classifiers, Convolutional Neural Networks (CNN)
