import cv2
import mediapipe as mp
import numpy as np
from supabase import create_client
import base64
from datetime import datetime
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase = create_client(supabase_url, supabase_key)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

class FaceEnrollment:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_face_features(self, landmarks):
        # Convert landmarks to numpy array
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Normalize the points for consistent encoding
        points = points - np.mean(points, axis=0)
        points = points / np.std(points)
        
        return points.flatten()

    def is_face_in_circle(self, face_landmarks, frame_width, frame_height):
        # Circle parameters
        circle_center_x = frame_width / 2
        circle_center_y = frame_height / 2
        circle_radius = min(frame_width, frame_height) * 0.25

        # Get all facial landmark points
        points = [(lm.x * frame_width, lm.y * frame_height) for lm in face_landmarks.landmark]
        
        # Check if ALL facial landmarks are inside the circle
        for x, y in points:
            distance = ((x - circle_center_x) ** 2 + (y - circle_center_y) ** 2) ** 0.5
            if distance > circle_radius * 0.8:
                return False
            
        return True

    def get_landmarks(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None, "No face detected"

        if len(results.multi_face_landmarks) > 1:
            return None, "Multiple faces detected"

        # Get face landmarks
        face_landmarks = results.multi_face_landmarks[0]

        # Check face alignment
        is_aligned, message = self.check_face_alignment(face_landmarks)
        if not is_aligned:
            return None, message

        # Extract landmarks for drawing
        landmarks = [{'x': lm.x, 'y': lm.y} for lm in face_landmarks.landmark]

        return landmarks, "Success"

    def check_face_alignment(self, landmarks):
        # Get nose tip and eyes
        nose_tip = landmarks.landmark[4]
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]
        
        # Check if face is centered
        if not (0.4 < nose_tip.x < 0.6):
            return False, "Center your face horizontally"
        
        if not (0.4 < nose_tip.y < 0.6):
            return False, "Center your face vertically"
        
        # Check if face is too close or too far
        eye_distance = abs(left_eye.x - right_eye.x)
        if eye_distance < 0.2:
            return False, "Move closer to the camera"
        if eye_distance > 0.4:
            return False, "Move away from the camera"
        
        return True, "Face aligned properly"

    def save_face_data(self, user_id, face_encoding):
        try:
            # Convert numpy array to list for JSON storage
            face_embeddings = face_encoding.tolist()
            
            # Save to Supabase face_data table
            data = {
                'user_id': user_id,
                'face_embeddings': face_embeddings,
                'created_at': datetime.utcnow().isoformat()
            }
            
            result = supabase.table('face_data').insert(data).execute()
            
            return True, "Face data saved successfully"
        except Exception as e:
            return False, f"Error saving face data: {str(e)}"

    def verify_face(self, frame, user_id):
        try:
            # Get current face encoding
            current_encoding, message = self.process_enrollment_frame(frame)
            if current_encoding is None:
                return False, message
            
            # Get stored face embeddings from Supabase
            result = supabase.table('face_data')\
                .select('face_embeddings')\
                .eq('user_id', user_id)\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            
            if not result.data:
                return False, "No face data found for user"
            
            # Convert stored embeddings back to numpy array
            stored_encoding = np.array(result.data[0]['face_embeddings'])
            
            # Compare encodings
            distance = np.linalg.norm(current_encoding - stored_encoding)
            
            # Set threshold for verification
            threshold = 0.6
            
            if distance < threshold:
                return True, "Face verified successfully"
            else:
                return False, "Face verification failed"
                
        except Exception as e:
            return False, f"Error during face verification: {str(e)}"

def create_face_enrollment_api():
    app = Flask(__name__)
    CORS(app)
    face_enrollment = FaceEnrollment()

    @app.route('/enroll-face', methods=['POST'])
    def enroll_face():
        try:
            data = request.json
            frame_data = base64.b64decode(data['frame'])
            user_data = data['userData']
            
            # Convert frame data to numpy array
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            height, width = frame.shape[:2]

            # Process the captured frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_enrollment.face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                return jsonify({
                    'success': False, 
                    'message': 'No face detected in the captured image. Please try again.'
                })

            # Verify face is properly positioned
            face_in_circle = face_enrollment.is_face_in_circle(
                results.multi_face_landmarks[0], 
                width, 
                height
            )

            if not face_in_circle:
                return jsonify({
                    'success': False,
                    'message': 'Face was not properly centered. Please align your face and try again.'
                })
            
            try:
                # Extract face features and store
                face_encoding = face_enrollment.extract_face_features(results.multi_face_landmarks[0])
                face_embeddings = face_encoding.tolist()
                
                # Save to Supabase
                data = {
                    'uid': user_data['uid'],
                    'face_embeddings': face_embeddings,
                    'created_at': datetime.utcnow().isoformat()
                }
                
                # Insert into face_data table
                result = supabase.table('face_data').insert(data).execute()
                
                if not result.data:
                    raise Exception('Failed to save face data to database')
                    
                face_data_id = result.data[0]['id']
                
                return jsonify({
                    'success': True, 
                    'message': 'Face captured and enrolled successfully',
                    'face_data_id': face_data_id
                })
                
            except Exception as e:
                print(f"Database error: {str(e)}")  # Server-side logging
                return jsonify({
                    'success': False, 
                    'message': f'Failed to save face data: {str(e)}'
                })
            
        except Exception as e:
            print(f"Processing error: {str(e)}")  # Server-side logging
            return jsonify({
                'success': False, 
                'message': f'Failed to process image: {str(e)}'
            })

    @app.route('/verify-face', methods=['POST'])
    def verify_face():
        try:
            data = request.json
            frame_data = base64.b64decode(data['frame'])
            user_id = data['user_id']
            
            # Convert frame data to numpy array
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Verify face
            success, message = face_enrollment.verify_face(frame, user_id)
            
            return jsonify({'success': success, 'message': message})
            
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

    @app.route('/process-frame', methods=['POST'])
    def process_frame():
        try:
            data = request.json
            frame_data = base64.b64decode(data['frame'])
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            height, width = frame.shape[:2]

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = face_enrollment.face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                return jsonify({
                    'success': False, 
                    'message': 'No face detected',
                    'processed_frame': base64.b64encode(frame).decode('utf-8')
                })

            # Check if face is inside circle
            face_in_circle = face_enrollment.is_face_in_circle(
                results.multi_face_landmarks[0], 
                width, 
                height
            )

            # Only draw mesh and consider face detected if it's inside circle
            if face_in_circle:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )
                message = 'Face detected and centered'
                success = True
            else:
                message = 'Position your entire face inside the circle'
                success = False

            # Convert frame back to base64
            _, buffer = cv2.imencode('.jpg', frame)
            processed_frame = base64.b64encode(buffer).decode('utf-8')

            return jsonify({
                'success': success,
                'message': message,
                'processed_frame': processed_frame
            })

        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'ok'})

    return app

if __name__ == '__main__':
    app = create_face_enrollment_api()
    app.run(port=5000)
