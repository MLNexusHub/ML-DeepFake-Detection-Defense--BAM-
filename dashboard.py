from flask import Flask, render_template, jsonify, request, send_from_directory
import firebase_admin
from firebase_admin import credentials, firestore, storage
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
import uuid
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import watermarking
import deepfake_detector
import traceback

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['WATERMARKED_FOLDER'] = os.path.join('static', 'watermarked_images')
app.config['REAL_FOLDER'] = os.path.join('static', 'detected_real')
app.config['FAKE_FOLDER'] = os.path.join('static', 'detected_fake')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories if they don't exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['WATERMARKED_FOLDER'], 
               app.config['REAL_FOLDER'], app.config['FAKE_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Initialize Firebase
try:
    cred = credentials.Certificate('ml-deepfake-detection-defense-firebase-adminsdk-fbsvc-31c694b341.json')
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'ml-deepfake-detection-defense.appspot.com'
    })
    db = firestore.client()
    bucket = storage.bucket()
    firebase_enabled = True
except Exception as e:
    print(f"Firebase initialization error: {str(e)}")
    print("Running without Firebase integration")
    firebase_enabled = False

def allowed_file(filename):
    """Check if file type is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/api/assets')
def get_assets():
    """Get all media assets."""
    try:
        if not firebase_enabled:
            return jsonify({'status': 'success', 'data': []})
            
        assets_ref = db.collection('assets')
        assets = []
        for doc in assets_ref.stream():
            asset = doc.to_dict()
            asset['id'] = doc.id
            assets.append(asset)
        return jsonify({'status': 'success', 'data': assets})
    except Exception as e:
        print(f"Error in /api/assets: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/alerts')
def get_alerts():
    """Get all detection alerts."""
    try:
        if not firebase_enabled:
            return jsonify({'status': 'success', 'data': []})
            
        alerts_ref = db.collection('alerts').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10)
        alerts = []
        for doc in alerts_ref.stream():
            alert = doc.to_dict()
            alert['id'] = doc.id
            alerts.append(alert)
        return jsonify({'status': 'success', 'data': alerts})
    except Exception as e:
        print(f"Error in /api/alerts: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/logs')
def get_logs():
    """Get all incident logs."""
    try:
        if not firebase_enabled:
            return jsonify({'status': 'success', 'data': []})
            
        logs_ref = db.collection('logs').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(20)
        logs = []
        for doc in logs_ref.stream():
            log = doc.to_dict()
            log['id'] = doc.id
            logs.append(log)
        return jsonify({'status': 'success', 'data': logs})
    except Exception as e:
        print(f"Error in /api/logs: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/test', methods=['POST'])
def test_image():
    """Test an image for watermark and deepfake detection."""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'status': 'error', 'message': 'File type not allowed'}), 400

        # Save the test image
        filename = secure_filename(file.filename)
        unique_filename = f"test_{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        try:
            # Initialize the watermark protector
            watermark_protector = watermarking.WatermarkProtector()
            
            # Verify the image (checks both watermark and deepfake)
            verification_result = watermark_protector.verify_image(filepath)
            
            # Ensure all values are JSON serializable
            def convert_to_native_types(obj):
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return [convert_to_native_types(x) for x in obj.tolist()]
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_native_types(x) for x in obj]
                elif isinstance(obj, dict):
                    return {k: convert_to_native_types(v) for k, v in obj.items()}
                elif hasattr(obj, 'tolist'):
                    return convert_to_native_types(obj.tolist())
                elif hasattr(obj, 'item'):
                    return obj.item()
                else:
                    return obj
            
            # Format the results for the frontend with safe defaults
            analysis_result = {
                "watermark_analysis": {
                    "has_watermark": bool(verification_result.get('has_watermark', False)),
                    "is_authentic": bool(verification_result.get('is_authentic', False)),
                    "watermark_data": verification_result.get('watermark_data', {}),
                    "watermark_strength": float(verification_result.get('watermark_strength', 0.0)),
                    "in_registry": bool(verification_result.get('in_registry', False)),
                    "original_path": f"/uploads/{unique_filename}"
                },
                "deepfake_detection": {
                    "is_deepfake": bool(verification_result.get('is_deepfake', False)),
                    "confidence": float(verification_result.get('deepfake_score', 0.0)),
                    "whitelisted": bool(verification_result.get('whitelisted', False)),
                    "model_predictions": verification_result.get('model_predictions', {})
                }
            }
            
            # Convert all values to native Python types
            analysis_result = convert_to_native_types(analysis_result)

            return jsonify({
                'status': 'success',
                'data': {
                    'filename': filename,
                    'analysis': analysis_result
                }
            })

        except Exception as e:
            print(f"Test verification error: {str(e)}")
            print(traceback.format_exc())
            # Return a structured error response even when there's an exception
            return jsonify({
                'status': 'error',
                'message': f'Error verifying image: {str(e)}',
                'error_details': traceback.format_exc(),
                'data': {
                    'filename': filename,
                    'analysis': {
                        'watermark_analysis': {'has_watermark': False, 'is_authentic': False},
                        'deepfake_detection': {'is_deepfake': False, 'confidence': 0.0}
                    }
                }
            }), 500

    except Exception as e:
        print(f"Test API error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error', 
            'message': str(e),
            'error_details': traceback.format_exc()
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and watermark embedding."""
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file provided'
            }), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
            
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': 'File type not allowed. Please upload a PNG, JPG, JPEG, or GIF file.'
            }), 400

        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        try:
            # Initialize the watermark protector
            watermark_protector = watermarking.WatermarkProtector()
            
            # Add watermark to the image
            result = watermark_protector.add_watermark(filepath)
            
            if not result.get('success', False):
                # If watermarking fails, return a helpful error but don't crash
                error_msg = result.get('error', 'Unknown error during watermarking')
                print(f"Watermarking error: {error_msg}")
                return jsonify({
                    'status': 'error',
                    'message': f'Failed to add watermark: {error_msg}',
                    'data': {
                        'filename': filename,
                        'original_path': f"/uploads/{unique_filename}"
                    }
                }), 400
                
            watermarked_path = result.get('output_path', '')
            if not watermarked_path or not os.path.exists(watermarked_path):
                # If path is missing or file doesn't exist, use a fallback approach
                print("Watermarked path missing or file doesn't exist, using original")
                watermarked_path = filepath
                watermarked_filename = os.path.basename(filepath)
            else:
                watermarked_filename = os.path.basename(watermarked_path)
            
            # Verify the watermarked image
            verification_result = watermark_protector.verify_image(watermarked_path)
            
            # Ensure all values are JSON serializable
            def convert_to_native_types(obj):
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return [convert_to_native_types(x) for x in obj.tolist()]
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_native_types(x) for x in obj]
                elif isinstance(obj, dict):
                    return {k: convert_to_native_types(v) for k, v in obj.items()}
                elif hasattr(obj, 'tolist'):
                    return convert_to_native_types(obj.tolist())
                elif hasattr(obj, 'item'):
                    return obj.item()
                else:
                    return obj
            
            # Format the response with consistent keys and safe defaults
            analysis_result = {
                "watermark_verification": {
                    "is_authentic": bool(verification_result.get('is_authentic', False)),
                    "has_watermark": bool(verification_result.get('has_watermark', False)),
                    "watermark_strength": float(verification_result.get('watermark_strength', 0.0)),
                    "in_registry": bool(verification_result.get('in_registry', False)),
                    "original_path": f"/uploads/{unique_filename}",
                    "watermarked_path": f"/static/watermarked_images/{watermarked_filename}",
                    "watermark_data": verification_result.get('watermark_data', {}),
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "process": "Initial Watermarking",
                        "strength": float(verification_result.get('watermark_strength', 0.0))
                    }
                },
                "deepfake_detection": {
                    "is_deepfake": bool(verification_result.get('is_deepfake', False)),
                    "confidence": float(verification_result.get('deepfake_score', 0.0)),
                    "whitelisted": bool(verification_result.get('whitelisted', False)),
                    "model_predictions": verification_result.get('model_predictions', {})
                }
            }
            
            # Convert all values to native Python types
            analysis_result = convert_to_native_types(analysis_result)

            # Store analysis result in Firebase
            if firebase_enabled:
                try:
                    doc_ref = db.collection('assets').document()
                    doc_ref.set({
                        'filename': filename,
                        'upload_time': datetime.now().isoformat(),
                        'analysis_result': analysis_result,
                        'status': 'watermarked',
                        'original_path': f"/uploads/{unique_filename}",
                        'watermarked_path': f"/static/watermarked_images/{watermarked_filename}"
                    })
                except Exception as e:
                    print(f"Firebase storage error: {str(e)}")
                    # Continue even if Firebase storage fails

            return jsonify({
                'status': 'success',
                'data': {
                    'filename': filename,
                    'analysis': analysis_result
                }
            })

        except Exception as e:
            print(f"Watermarking error: {str(e)}")
            print(traceback.format_exc())
            # Clean up the uploaded file if watermarking fails
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
                
            # Return a structured error response
            return jsonify({
                'status': 'error',
                'message': f'Failed to add watermark to image: {str(e)}',
                'error_details': traceback.format_exc(),
                'data': {
                    'filename': filename
                }
            }), 500

    except Exception as e:
        print(f"Upload error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Error processing upload: {str(e)}',
            'error_details': traceback.format_exc()
        }), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving uploaded file: {str(e)}")
        return jsonify({'status': 'error', 'message': 'File not found'}), 404

@app.route('/static/watermarked_images/<filename>')
def watermarked_file(filename):
    """Serve watermarked files."""
    try:
        return send_from_directory(app.config['WATERMARKED_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving watermarked file: {str(e)}")
        return jsonify({'status': 'error', 'message': 'File not found'}), 404

@app.route('/static/detected_real/<filename>')
def real_file(filename):
    """Serve detected real files."""
    try:
        return send_from_directory(app.config['REAL_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving real file: {str(e)}")
        return jsonify({'status': 'error', 'message': 'File not found'}), 404

@app.route('/static/detected_fake/<filename>')
def fake_file(filename):
    """Serve detected fake files."""
    try:
        return send_from_directory(app.config['FAKE_FOLDER'], filename)
    except Exception as e:
        print(f"Error serving fake file: {str(e)}")
        return jsonify({'status': 'error', 'message': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)