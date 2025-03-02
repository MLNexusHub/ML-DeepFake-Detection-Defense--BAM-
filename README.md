# ML Deepfake Detection & Defense System

A comprehensive solution for detecting and defending against deepfakes using a multi-layered approach combining watermarking, traditional image analysis, and machine learning techniques.

## Overview

This system provides a robust framework for:
1. Watermarking authentic images to verify their integrity
2. Detecting potential deepfakes using multiple analysis methods
3. Managing and tracking verified vs. manipulated media
4. Providing a user-friendly dashboard for monitoring and analysis

## Key Features

- **Watermarking System**: Embeds secure digital signatures into images
- **Deepfake Detection Engine**: Utilizes multiple analysis techniques to identify manipulated images
- **White-listing Mechanism**: Maintains a registry of known authentic images
- **User Dashboard**: Web interface for uploading, testing, and monitoring images
- **Robust API**: RESTful endpoints for integration with other systems

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JahagirdarPrajwal/ML-DeepFake-Detection-Defense--BAM-.git
cd ML-DeepFake-Detection-Defense--BAM-
```

2. Create and activate a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Starting the Server

```bash
python dashboard.py
```

The dashboard will be available at: http://localhost:5000

### API Endpoints

- `/api/upload` - Upload and watermark an image
- `/api/test` - Test an image for watermarks and deepfake detection
- `/api/assets` - Get all media assets
- `/api/alerts` - Get detection alerts
- `/api/logs` - Get system logs

## Components

### Watermarking Module

The `watermarking.py` file implements:
- Simple Invertible Neural Network (INN) for encoding/decoding
- Error Correction Code (ECC) for robust watermark extraction
- Methods for embedding and extracting watermarks

### Deepfake Detection Module

The `deepfake_detector.py` file implements:
- Traditional image analysis methods (noise, compression artifacts, face consistency)
- Ensemble approach for more reliable detection
- White-listing system for known authentic images

### Dashboard

The `dashboard.py` file provides:
- Flask web server
- API endpoints for image processing
- Integration with Firebase (optional)
- File serving and management

## License

MIT

## Contributors

- Prajwal Jahagirdar
