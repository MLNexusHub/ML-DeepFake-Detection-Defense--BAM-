import os
from datetime import datetime
from pathlib import Path
import mimetypes
from typing import Dict, List, Optional

import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
def initialize_firebase(credentials_path: str) -> None:
    """
    Initialize Firebase Admin SDK with the provided service account credentials.
    
    Args:
        credentials_path (str): Path to the Firebase service account credentials JSON file
    """
    try:
        cred = credentials.Certificate(credentials_path)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        raise Exception(f"Failed to initialize Firebase: {str(e)}")

def get_file_metadata(file_path: Path) -> Dict:
    """
    Extract metadata from a file.
    
    Args:
        file_path (Path): Path to the file
        
    Returns:
        Dict: Dictionary containing file metadata
    """
    stats = file_path.stat()
    mime_type, _ = mimetypes.guess_type(str(file_path))
    
    return {
        "filename": file_path.name,
        "file_type": mime_type or "application/octet-stream",
        "path": str(file_path),
        "created_at": datetime.fromtimestamp(stats.st_ctime),
        "updated_at": datetime.fromtimestamp(stats.st_mtime)
    }

def scan_directory(directory_path: str, file_types: Optional[List[str]] = None) -> List[Dict]:
    """
    Scan a directory for media files and extract their metadata.
    
    Args:
        directory_path (str): Path to the directory to scan
        file_types (List[str], optional): List of file extensions to include (e.g., ['.jpg', '.mp4'])
        
    Returns:
        List[Dict]: List of dictionaries containing file metadata
    """
    if file_types is None:
        file_types = ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mov', '.avi']
    
    media_files = []
    directory = Path(directory_path)
    
    try:
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in file_types:
                metadata = get_file_metadata(file_path)
                media_files.append(metadata)
    except Exception as e:
        raise Exception(f"Error scanning directory: {str(e)}")
    
    return media_files

def update_firestore(media_files: List[Dict]) -> None:
    """
    Update Firestore with media file metadata.
    
    Args:
        media_files (List[Dict]): List of dictionaries containing file metadata
    """
    try:
        db = firestore.client()
        collection = db.collection('media_assets')
        
        # Use batch writes for better performance
        batch = db.batch()
        batch_size = 0
        max_batch_size = 500  # Firestore batch limit
        
        for media_file in media_files:
            # Use file path as document ID for uniqueness
            doc_ref = collection.document(media_file['path'])
            batch.set(doc_ref, media_file, merge=True)
            batch_size += 1
            
            # Commit batch when size limit is reached
            if batch_size >= max_batch_size:
                batch.commit()
                batch = db.batch()
                batch_size = 0
        
        # Commit any remaining documents
        if batch_size > 0:
            batch.commit()
            
    except Exception as e:
        raise Exception(f"Error updating Firestore: {str(e)}")

def main(directory_path: str, credentials_path: str, file_types: Optional[List[str]] = None) -> None:
    """
    Main function to scan directory and update Firestore.
    
    Args:
        directory_path (str): Path to the directory to scan
        credentials_path (str): Path to the Firebase service account credentials JSON file
        file_types (List[str], optional): List of file extensions to include
    """
    try:
        # Initialize Firebase
        initialize_firebase(credentials_path)
        
        # Scan directory for media files
        media_files = scan_directory(directory_path, file_types)
        
        # Update Firestore
        update_firestore(media_files)
        
        print(f"Successfully processed {len(media_files)} media files")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python asset_discovery.py <media_directory_path> <firebase_credentials_path>")
        sys.exit(1)
        
    directory_path = sys.argv[1]
    credentials_path = sys.argv[2]
    main(directory_path, credentials_path)