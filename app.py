import os
import sqlite3  # Still imported, can be removed if SQLite is no longer a fallback
from flask import Flask, render_template, request, redirect, url_for, session, flash, g, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from fpdf import FPDF
from datetime import datetime
import cv2  # For image processing
import numpy as np  # For numerical operations with images
import json  # For parsing Firebase config and storing JSON data
import traceback # Import for printing full tracebacks
import uuid # For generating unique filenames

# --- NEW IMPORTS FOR POSTGRESQL & FLASK-SQLALCHEMY ---
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
# --- END NEW IMPORTS ---

# --- NEW IMPORTS FOR AI (Machine Learning) INTEGRATION ---
from sklearn.neighbors import KNeighborsClassifier
import joblib  # For saving/loading machine learning models
import pandas as pd  # For CSV handling
# ---------------------------------------------------------

# --- IMAGE PROCESSING FUNCTIONS (Self-contained for simplicity) ---
def gray_world_white_balance(img):
    """
    Applies Gray World Algorithm for white balancing an image.
    Args:
        img (numpy.ndarray): The input image in BGR format.
    Returns:
        numpy.ndarray: The white-balanced image in BGR format.
    """
    result = img.copy().astype(np.float32)  # Convert to float32 for calculations

    # Calculate average intensity for each channel
    avgB = np.mean(result[:, :, 0])
    avgG = np.mean(result[:, :, 1])
    avgR = np.mean(result[:, :, 2])

    # Calculate overall average gray value
    avgGray = (avgB + avgG + avgR) / 3

    # Apply scaling factor to each channel
    result[:, :, 0] = np.minimum(result[:, :, 0] * (avgGray / avgB), 255)
    result[:, :, 1] = np.minimum(result[:, :, 1] * (avgGray / avgG), 255)
    result[:, :, 2] = np.minimum(result[:, :, 2] * (avgGray / avgR), 255)
    return result.astype(np.uint8)  # Convert back to uint8 for image display/saving


def correct_lighting(img_np_array):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting correction.
    Args:
        img_np_array (numpy.ndarray): The input image (NumPy array, BGR format).
    Returns:
        numpy.ndarray: The corrected image (NumPy array, BGR format).
    """
    # Convert to LAB color space
    img_lab = cv2.cvtColor(img_np_array, cv2.COLOR_BGR2LAB)

    # Split channels
    l, a, b = cv2.split(img_lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    # Merge channels back
    img_lab_eq = cv2.merge((l_eq, a, b))

    # Convert back to BGR color space
    img_corrected = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2BGR)
    return img_corrected


# --- END IMAGE PROCESSING FUNCTIONS ---


# ===============================================
# 1. FLASK APPLICATION SETUP & CONFIGURATION
# ===============================================

app = Flask(__name__)
# Secret key from environment variable for production readiness
app.secret_key = os.environ.get("SECRET_KEY", "your_strong_dev_secret_key_12345")

# Define upload and report folders
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'

# Ensure these directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Configure Flask app with folder paths
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER

# --- NEW DATABASE CONFIGURATION FOR POSTGRESQL (using Flask-SQLAlchemy) ---
# Get the PostgreSQL database URL from the environment variable 'DATABASE_URL'.
# For local development (when DATABASE_URL is not set), it will fall back to SQLite.
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL',
    'sqlite:///database.db' # This is your local SQLite file for development
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Recommended to suppress warnings

# Initialize your database instance for Flask-SQLAlchemy
db = SQLAlchemy(app)

# Initialize Flask-Migrate (crucial for creating and updating tables in Postgres)
migrate = Migrate(app, db)
# --- END NEW DATABASE CONFIGURATION ---


# --- Firestore (Simulated for Canvas) - Retained from your original code ---
# These are conceptual imports for a standard Python environment.
# In the Canvas environment, Firebase interaction happens via fetch to API.
# However, we keep the imports for conceptual structure for a real Python Flask app.
# For this Canvas, we will simulate the Firestore calls.
# from firebase_admin import credentials, firestore, initialize_app
# from google.cloud.firestore import Client as FirestoreClient # For type hinting if using client library

# __app_id and __firebase_config are provided by the Canvas environment.
app_id = os.environ.get('__app_id', 'default-app-id')  # Fallback for local testing
firebase_config_str = os.environ.get('__firebase_config', '{}')
firebase_config = json.loads(firebase_config_str)

# Global for simulated Firestore database object
# This will be initialized once as a simple dict to hold all simulated data
db_data = {
    'artifacts': {
        app_id: {
            'users': {},
            'public': {'data': {}}
        }
    }
}
# IMPORTANT: 'db' in your original code pointed to 'db_data'. Now 'db' points to SQLAlchemy.
# Ensure your Firestore simulation functions use 'db_data' directly, not 'db'.
# This avoids conflict with SQLAlchemy's 'db' object.


# --- Firestore Initialization Logic (moved to a function not directly called by app_context) ---
def setup_initial_firebase_globals():
    """
    Sets up conceptual global data for Firestore simulation if needed.
    This runs once at app startup.
    """
    print(f"DEBUG: App ID: {app_id}")
    print(f"DEBUG: Firebase Config (partial): {list(firebase_config.keys())[:3]}...")

# Call this once at the module level or on app init if needed
setup_initial_firebase_globals()


# ===============================================
# 2. DATABASE MODELS (for PostgreSQL with Flask-SQLAlchemy)
# ===============================================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Define relationship to PatientRecord
    # 'backref' allows you to access associated PatientRecords from a User object: user.patient_records
    patient_records = db.relationship('PatientRecord', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class PatientRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    patient_name = db.Column(db.String(100), nullable=False)
    record_date = db.Column(db.DateTime, default=datetime.utcnow)
    uploaded_image_path = db.Column(db.String(255))
    analysis_results_json = db.Column(db.Text) # To store JSON string of the analysis results

    # Add specific fields for shades if you want them as separate columns in DB
    incisal_shade = db.Column(db.String(20))
    middle_shade = db.Column(db.String(20))
    cervical_shade = db.Column(db.String(20))
    overall_ml_shade = db.Column(db.String(20))
    # You can add more fields based on the 'detected_shades' dictionary
    # For example:
    # simulated_overall_shade = db.Column(db.String(50))
    # tooth_condition = db.Column(db.String(100))
    # stain_presence = db.Column(db.String(100))
    # decay_presence = db.Column(db.String(100))
    # suggested_aesthetic_shade = db.Column(db.String(50))
    # aesthetic_confidence = db.Column(db.String(20))
    # recommendation_notes = db.Column(db.Text)

    def __repr__(self):
        return f'<PatientRecord {self.patient_name} by User {self.user_id}>'

# ===============================================
# END DATABASE MODELS
# ===============================================


# ===============================================
# 3. DATABASE INITIALIZATION & HELPERS (Firestore Simulation - Unchanged)
# ===============================================

# These functions still work with the simulated Firestore 'db_data' global

def get_firestore_collection(path_segments):
    """Navigates the simulated Firestore structure to get a collection."""
    current_level = db_data # Use the global db_data directly
    for segment in path_segments:
        if segment not in current_level:
            current_level[segment] = {}
        current_level = current_level[segment]
    return current_level


def get_firestore_document(path_segments):
    """Navigates the simulated Firestore structure to get a document."""
    collection = get_firestore_collection(path_segments[:-1])
    doc_id = path_segments[-1]
    return collection.get(doc_id)


def set_firestore_document(path_segments, data):
    """Sets a document in the simulated Firestore."""
    collection = get_firestore_collection(path_segments[:-1])
    doc_id = path_segments[-1]
    collection[doc_id] = data
    print(f"DEBUG: Simulated Firestore set: {os.path.join(*path_segments)}")


def add_firestore_document(path_segments, data):
    """Adds a document with auto-generated ID in the simulated Firestore."""
    collection = get_firestore_collection(path_segments)
    doc_id = str(np.random.randint(100000, 999999))  # Simulate auto-ID
    collection[doc_id] = data
    print(f"DEBUG: Simulated Firestore added: {os.path.join(*path_segments)}/{doc_id}")
    return doc_id  # Return the simulated ID


def get_firestore_documents_in_collection(path_segments, query_filters=None):
    """Gets documents from a simulated Firestore collection, with basic filtering."""
    collection = get_firestore_collection(path_segments)
    results = []
    for doc_id, doc_data in collection.items():
        if query_filters:
            match = True
            for field, value in query_filters.items():
                if doc_data.get(field) != value:
                    match = False
                    break
            if match:
                results.append(doc_data)
        else:
            results.append(doc_data)

    # Simulate ordering by record_date if present
    if results and 'record_date' in results[0]:
        results.sort(key=lambda x: x.get('record_date', ''), reverse=True)
    return results


# ===============================================
# 4. AUTHENTICATION HELPERS (UPDATED for Flask-SQLAlchemy)
# ===============================================

@app.before_request
def load_logged_in_user():
    """
    Loads the logged-in user into Flask's `g` object for the current request.
    Uses session for persistence across requests.
    Attempts to load user from PostgreSQL.
    """
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
        g.user_id = None
        g.firestore_user_id = None # Keep this for your simulated Firestore logic
    else:
        try:
            # Query user from PostgreSQL by ID using SQLAlchemy
            user = User.query.get(user_id) # Use .get() for primary key
            if user:
                g.user = {'id': user.id, 'username': user.username}
                g.user_id = user.id
                g.firestore_user_id = g.user_id # Maintain consistency for Firestore sim
            else:
                # User not found in DB (e.g., deleted or session stale)
                session.clear()
                g.user = None
                g.user_id = None
                g.firestore_user_id = None
                print(f"DEBUG: User with ID {user_id} not found in DB. Clearing session.")
        except Exception as e:
            # Handle potential database connection errors during before_request
            print(f"ERROR: Database error in load_logged_in_user: {e}")
            traceback.print_exc() # Print full traceback for debugging
            session.clear() # Clear session to prevent infinite loop on error
            g.user = None
            g.user_id = None
            g.firestore_user_id = None

    # Handle initial_auth_token for Canvas environment if no DB user is set
    # Or fallback to an anonymous ID if no token and no real user
    if g.user_id is None:
        initial_auth_token = os.environ.get('__initial_auth_token')
        if initial_auth_token:
            # This token might be a generic canvas user ID, not a DB user ID
            simulated_id = initial_auth_token.split(':')[-1]
            g.user_id = simulated_id
            g.user = {'id': simulated_id, 'username': f"CanvasUser_{simulated_id[:8]}"}
            g.firestore_user_id = g.user_id # Use this for Firestore pathing
            print(f"DEBUG: Setting g.user from initial_auth_token: {g.user['username']}")
        else:
            # Fallback to anonymous if no token and no real user
            g.user_id = 'anonymous-' + str(np.random.randint(100000, 999999))
            g.user = {'id': g.user_id, 'username': f"AnonUser_{g.user_id[-6:]}"}
            g.firestore_user_id = g.user_id
            print(f"DEBUG: Setting g.user to anonymous: {g.user['username']}")

    # print(f"DEBUG: Request user loaded: {g.user['username']} (FS ID: {g.firestore_user_id})")


def login_required(view):
    """Decorator to protect routes that require a logged-in user (not anonymous)."""
    import functools

    @functools.wraps(view)
    def wrapped_view(**kwargs):
        # Check if the user is truly "logged in" (not the default anonymous user)
        if g.user is None or 'anonymous' in str(g.user_id): # Check if user is anonymous or g.user_id is not set
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return view(**kwargs)

    return wrapped_view


# ===============================================
# 5. CORE HELPER FUNCTIONS (Image Correction, Shade Detection, PDF Generation, Enhanced Simulated AI)
# ===============================================

# Function to map L-value to a VITA shade based on adjusted rules
# This function will now expect L_value to be in 0-100 range
def map_l_to_shade_rule_based(l_value_100_scale): # Parameter name changed for clarity
    """
    Maps L-value (0-100 scale) to a VITA shade.
    Adjusted thresholds for a more granular mapping.
    """
    if l_value_100_scale > 85: # Very bright, whiter than B1
        return "B1+" # Custom for very bright
    elif l_value_100_scale > 80:
        return "B1"
    elif l_value_100_scale > 75:
        return "A1"
    elif l_value_100_scale > 70:
        return "B2"
    elif l_value_100_scale > 65:
        return "A2"
    elif l_value_100_scale > 60:
        return "D2" # Introduce D shades
    elif l_value_100_scale > 55:
        return "C1"
    elif l_value_100_scale > 50:
        return "A3"
    elif l_value_100_scale > 45:
        return "D3"
    elif l_value_100_scale > 40:
        return "B3"
    elif l_value_100_scale > 35:
        return "C2"
    elif l_value_100_scale > 30:
        return "A3.5"
    elif l_value_100_scale > 25:
        return "D4"
    elif l_value_100_scale > 20:
        return "C3"
    else:
        return "C4" # Darkest shades


# --- AI Model Setup (Loading Data from CSV & Training/Loading) ---
MODEL_FILENAME = "shade_classifier_model.pkl"
DATASET_FILENAME = "tooth_shades_simulated.csv"


def train_model():
    """Train a new KNN model using the CSV file and save it."""
    if not os.path.exists(DATASET_FILENAME):
        print(f"ERROR: Dataset '{DATASET_FILENAME}' is missing. Cannot train model.")
        return None

    try:
        df = pd.read_csv(DATASET_FILENAME)
        if df.empty:
            print(f"ERROR: Dataset '{DATASET_FILENAME}' is empty. Cannot train model.")
            return None

        # Ensure that the training data itself is expected to be in 0-100 scale for L values.
        # If your CSV's incisal_l, middle_l, cervical_l are already 0-255, this might need adjustment.
        # Based on previous context, these CSV values are assumed to be 0-100 L scale.
        X = df[['incisal_l', 'middle_l', 'cervical_l']].values
        y = df['overall_shade'].values
        print(f"DEBUG: Training data shape={X.shape}, classes={np.unique(y)}")

        model_to_train = KNeighborsClassifier(n_neighbors=3)
        model_to_train.fit(X, y)
        joblib.dump(model_to_train, MODEL_FILENAME)
        print(f"DEBUG: Model trained and saved to {MODEL_FILENAME}")
        return model_to_train
    except Exception as e:
        print(f"ERROR: Failed to train model: {e}")
        return None


def load_or_train_model():
    """Load existing model or train a new one."""
    if os.path.exists(MODEL_FILENAME):
        try:
            loaded_model = joblib.load(MODEL_FILENAME)
            print(f"DEBUG: Loaded pre-trained shade model from {MODEL_FILENAME}")
            return loaded_model
        except Exception as e:
            print(f"WARNING: Could not load model from {MODEL_FILENAME}: {e}. Attempting to retrain.")
            return train_model()
    else:
        print(f"DEBUG: No existing model found at {MODEL_FILENAME}. Attempting to train new model.")
        return train_model()


# Global variable to store the loaded/trained model
shade_classifier_model = load_or_train_model()

# --- End AI Model Setup ---


# =========================================================
# ENHANCED: Placeholder AI Modules for Advanced Analysis
# These functions now return more "intelligent" simulated data
# based on image properties, mimicking commercial insights.
# =========================================================

def detect_face_features(image_np_array):
    """
    ENHANCED PLACEHOLDER: Simulates detailed face feature extraction.
    Now attempts to derive more nuanced skin tone (including undertones),
    detailed lip color, and eye contrast based on average color properties
    and simple statistical analysis of the input image.

    Args:
        image_np_array (numpy.ndarray): The input image (BGR format).
    Returns:
        dict: Simulated facial features including skin and lip tone.
    """
    print("DEBUG: Simulating detailed Face Detection and Feature Extraction with color analysis...")

    img_lab = cv2.cvtColor(image_np_array, cv2.COLOR_BGR2LAB)
    avg_l = np.mean(img_lab[:, :, 0])
    avg_a = np.mean(img_lab[:, :, 1])
    avg_b = np.mean(img_lab[:, :, 2])

    # Simulate Skin Tone and Undertone based on 'a' (green-red) and 'b' (blue-yellow) channels
    # More detailed heuristics for simulation
    skin_tone_category = "Medium"
    skin_undertone = "Neutral"

    if avg_l > 75:
        skin_tone_category = "Light"
    elif avg_l > 60:
        skin_tone_category = "Medium"
    elif avg_l > 45:
        skin_tone_category = "Dark"
    else:
        skin_tone_category = "Very Dark"

    if avg_b > 15 and avg_a > 8:
        skin_undertone = "Warm (Golden/Peach)"
    elif avg_b < 0 and avg_a < 5:
        skin_undertone = "Cool (Pink/Blue)"
    elif avg_b >= 0 and avg_a >= 5 and avg_a <= 8 and avg_b <= 15:
        skin_undertone = "Neutral"
    elif avg_b > 5 and avg_a < 5:
        skin_undertone = "Olive (Greenish)"

    simulated_skin_tone = f"{skin_tone_category} with {skin_undertone} undertones"

    # Simulate Lip Color based on 'a' (redness) and 'b' (yellowness) and L (lightness)
    simulated_lip_color = "Natural Pink"
    if avg_a > 20 and avg_l < 60:
        simulated_lip_color = "Deep Rosy Red"
    elif avg_a > 15 and avg_l >= 60:
        simulated_lip_color = "Bright Coral"
    elif avg_b < 5 and avg_l < 50:
        simulated_lip_color = "Subtle Mauve/Berry"
    elif avg_l < 50 and avg_a < 10 and avg_b < 10:
        simulated_lip_color = "Muted Nude/Brown"
    elif avg_l > 70 and avg_a < 10:
        simulated_lip_color = "Pale Nude"

    # Simulate Eye Contrast (using percentile spread in L-channel)
    l_channel = img_lab[:, :, 0]
    p10, p90 = np.percentile(l_channel, [10, 90])
    contrast_spread = p90 - p10
    eye_contrast_sim = "Medium"
    if contrast_spread > 40:
        eye_contrast_sim = "High (Distinct Features)"
    elif contrast_spread < 20:
        eye_contrast_sim = "Low (Soft Features)"

    return {
        "skin_tone": simulated_skin_tone,
        "lip_color": simulated_lip_color,
        "eye_contrast": eye_contrast_sim,
        "facial_harmony_score": round(np.random.uniform(0.7, 0.95), 2),
        # "notes": "Simulated analysis. Actual facial feature detection requires specialized deep learning models." # REMOVED
    }


def segment_and_analyze_teeth(image_np_array):
    """
    ENHANCED PLACEHOLDER: Simulates advanced tooth segmentation and shade analysis.
    Provides more detailed simulated insights on tooth condition and stain presence.

    Args:
        image_np_array (numpy.ndarray): The pre-processed image (BGR format).
    Returns:
        dict: Simulated overall tooth shade characteristics and condition.
    """
    print("DEBUG: Simulating detailed Tooth Segmentation and Analysis...")

    img_lab = cv2.cvtColor(image_np_array, cv2.COLOR_BGR2LAB)

    avg_l = np.mean(img_lab[:, :, 0])
    avg_a = np.mean(img_lab[:, :, 1])
    avg_b = np.mean(img_lab[:, :, 2])

    # Simulate Overall Shade (expanded VITA-like mapping)
    if avg_l > 78:
        simulated_overall_shade = "B1 (High Brightness)"
    elif avg_l > 73:
        simulated_overall_shade = "A1 (Natural Brightness)"
    elif avg_l > 68:
        simulated_overall_shade = "A2 (Medium Brightness)"
    elif avg_l > 63:
        simulated_overall_shade = "B2 (Slightly Darker)"
    elif avg_l > 58:
        simulated_overall_shade = "C1 (Moderate Darkness)"
    elif avg_l > 53:
        simulated_overall_shade = "C2 (Noticeable Darkness)"
    elif avg_l > 48:
        simulated_overall_shade = "A3 (Darker, Reddish Tint)"
    else:
        simulated_overall_shade = "C3 (Very Dark)"

    # Simulate Tooth Condition based on 'L' and 'b' (yellowish tint)
    tooth_condition_sim = "Normal & Healthy Appearance"
    if avg_b > 20 and avg_l < 70:
        tooth_condition_sim = "Mild Discoloration (Yellowish)"
    elif avg_b > 25 and avg_l < 60:
        tooth_condition_sim = "Moderate Discoloration (Strong Yellow)"
    elif avg_l < 55 and avg_a > 10:
        tooth_condition_sim = "Pronounced Discoloration (Brown/Red)"
    elif avg_l < 60 and avg_b < 0:
        tooth_condition_sim = "Greyish Appearance"

    # Simulate Stain Presence based on contrast within tooth area (requires segmentation in real model)
    # For simulation, use a general image contrast heuristic
    l_std_dev = np.std(img_lab[:, :, 0])
    stain_presence_sim = "None detected"
    if l_std_dev > 25 and avg_l > 60:  # High contrast in brighter teeth could suggest stains
        stain_presence_sim = "Possible light surface stains"
    elif l_std_dev > 35 and avg_l < 60:  # High contrast in darker teeth
        stain_presence_sim = "Moderate localized stains"

    decay_presence_sim = "No visible signs of decay"  # Dummy
    if np.random.rand() < 0.05:  # 5% chance of simulating decay (random)
        decay_presence_sim = "Potential small carious lesion (simulated - consult professional)"

    return {
        "overall_lab": {"L": float(avg_l), "a": float(avg_a), "b": float(avg_b)},  # Ensure float for JSON
        "simulated_overall_shade": simulated_overall_shade,
        "tooth_condition": tooth_condition_sim,
        "stain_presence": stain_presence_sim,
        "decay_presence": decay_presence_sim,
        # "notes": "Simulated tooth analysis. Real analysis requires intra-oral images and advanced dental AI." # REMOVED
    }


def aesthetic_shade_suggestion(facial_features, tooth_analysis):
    """
    ENHANCED PLACEHOLDER: Simulates an aesthetic mapping model with more context.
    Suggestions are now more specific, considering simulated skin/lip tones.
    Confidence is now more dynamic based on harmony score and conditions.

    Args:
        facial_features (dict): Output from detect_face_features.
        tooth_analysis (dict): Output from segment_and_analyze_teeth.
    Returns:
        dict: Simulated aesthetic recommendations.
    """
    print("DEBUG: Simulating detailed Aesthetic Mapping and Shade Suggestion...")

    suggested_shade = "No specific aesthetic suggestion (Simulated)"
    aesthetic_confidence = "Low" # Changed default to 'Low'
    recommendation_notes = "This is a simulated aesthetic suggestion. Consult a dental specialist for personalized cosmetic planning based on your unique facial features and desired outcome. Advanced AI for aesthetics is complex and evolving."

    current_shade = tooth_analysis.get('simulated_overall_shade', '')
    skin_tone = facial_features.get('skin_tone', '').lower()
    lip_color = facial_features.get('lip_color', '').lower()
    facial_harmony_score = facial_features.get('facial_harmony_score', 0.5) # Get score, default to 0.5

    # Example sophisticated simulated logic with more dynamic confidence
    if "warm" in skin_tone:
        if "b1" in current_shade or "a1" in current_shade:
            suggested_shade = "Optimal Match (Simulated - Warm Undertone)"
            aesthetic_confidence = "Very High"
            recommendation_notes = "Your simulated warm skin undertone harmonizes exceptionally well with this bright shade, suggesting an optimal match. Consider maintaining this shade."
        elif "c3" in current_shade or "c2" in current_shade or "a3" in current_shade:
            suggested_shade = "B1 or A2 (Simulated - Brightening for Warm Undertone)"
            aesthetic_confidence = "High"
            recommendation_notes = "Your simulated warm skin undertone would be beautifully complemented by a brighter, slightly warmer tooth shade like B1 or A2. Consider professional whitening for a more radiant smile."
        else:
            aesthetic_confidence = "Medium" # Default for warm skin if no specific shade match

    elif "cool" in skin_tone:
        if "a1" in current_shade or "b1" in current_shade:
            suggested_shade = "Optimal Match (Simulated - Cool Undertone)"
            aesthetic_confidence = "Very High"
            recommendation_notes = "This shade provides excellent contrast and harmony with your simulated cool skin undertone, suggesting an optimal match. A very crisp and bright appearance."
        elif "a3" in current_shade or "b2" in current_shade or "d" in current_shade:
            suggested_shade = "A1 or B1 (Simulated - Brightening for Cool Undertone)"
            aesthetic_confidence = "High"
            recommendation_notes = "With your simulated cool skin undertone, a crisp, bright shade like A1 or B1 could enhance your overall facial harmony. Avoid overly yellow shades for best results."
        else:
            aesthetic_confidence = "Medium" # Default for cool skin if no specific shade match

    elif "neutral" in skin_tone:
        if "b1" in current_shade or "a1" in current_shade or "a2" in current_shade:
            suggested_shade = "Balanced Brightness (Simulated)"
            aesthetic_confidence = "High"
            recommendation_notes = "Your neutral skin tone offers great versatility. This shade provides a balanced and natural bright smile. Options for further brightening or warmth can be explored."
        else:
            aesthetic_confidence = "Medium" # Default for neutral skin

    elif "olive" in skin_tone:
        if "a2" in current_shade or "b2" in current_shade:
            suggested_shade = "Enhanced Natural (Simulated - Olive Tone)"
            aesthetic_confidence = "Medium"
            recommendation_notes = "For a simulated olive skin tone, a balanced brightening to A2 can provide a natural yet enhanced smile. Be mindful of shades that pull too much yellow or grey."
        elif "a1" in current_shade or "b1" in current_shade:
            suggested_shade = "Significant Brightening (Simulated - Olive Tone)"
            aesthetic_confidence = "High"
            recommendation_notes = "While your current shade provides a natural look, shades like A1 or B1 could offer a more noticeable brightening effect while maintaining harmony with your olive tone."
        else:
            aesthetic_confidence = "Low" # Default for olive skin if no specific shade match

    # General confidence boost based on Facial Harmony Score
    if facial_harmony_score >= 0.90:
        if aesthetic_confidence == "Low": # Upgrade 'Low' to 'Medium'
            aesthetic_confidence = "Medium"
        elif aesthetic_confidence == "Medium": # Upgrade 'Medium' to 'High'
            aesthetic_confidence = "High"
    elif facial_harmony_score >= 0.80 and aesthetic_confidence == "Low":
        aesthetic_confidence = "Medium" # Slightly boost low confidence if harmony is decent

    # Ensure notes are concise if confidence is high, or revert to general if still low
    if aesthetic_confidence == "Very High":
        # Recommendation notes are already specific for 'Optimal Match'
        pass
    elif aesthetic_confidence == "High":
        # Recommendation notes are already specific for 'Brightening'
        pass
    elif aesthetic_confidence == "Medium":
        if "Balanced Brightness" not in suggested_shade and "Enhanced Natural" not in suggested_shade:
            recommendation_notes = "This shade offers a natural and pleasing appearance. For more significant changes, a dental consultation is recommended."
    else: # Low or Very Low confidence
        suggested_shade = "Consult Dental Specialist (Simulated)"
        recommendation_notes = "Based on the simulated analysis, a personalized consultation with a dental specialist is highly recommended for tailored cosmetic planning due to the complexity of aesthetic matching."


    return {
        "suggested_aesthetic_shade": suggested_shade,
        "aesthetic_confidence": aesthetic_confidence,
        "recommendation_notes": recommendation_notes
    }


# =========================================================
# END ENHANCED: Placeholder AI Modules
# =========================================================


def detect_shades_from_image(image_path):
    """
    Performs lighting correction, white balance, extracts features,
    and then uses the pre-trained ML model for overall tooth shade detection.
    Also, provides rule-based shades for individual zones for UI consistency.
    This now also calls new placeholder AI modules.
    """
    print(f"DEBUG: Starting image processing for {image_path}")
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: cv2.imread returned None for image at {image_path}. File might be missing, corrupted, or not an image (e.g., non-image file type, invalid path).")
            return {
                "incisal": "N/A", "middle": "N/A", "cervical": "N/A",
                "overall_ml_shade": "N/A",
                "face_features": {}, "tooth_analysis": {}, "aesthetic_suggestion": {}
            }
        print(f"DEBUG: Image loaded successfully. Shape: {img.shape}, Type: {img.dtype}")

        # Ensure the image is not empty after loading
        if img.size == 0:
            print(f"ERROR: Image loaded but is empty (0 size) for {image_path}.")
            return {
                "incisal": "N/A", "middle": "N/A", "cervical": "N/A",
                "overall_ml_shade": "N/A",
                "face_features": {}, "tooth_analysis": {}, "aesthetic_suggestion": {}
            }


        # --- Apply Image Pre-processing ---
        img_wb = gray_world_white_balance(img)
        print("DEBUG: Gray world white balance applied.")
        img_corrected = correct_lighting(img_wb)
        print("DEBUG: Lighting correction applied.")

        # --- Call Enhanced Placeholder AI modules ---
        # Note: These functions also have print statements for their simulation steps
        face_features = detect_face_features(img_corrected)
        tooth_analysis = segment_and_analyze_teeth(img_corrected)
        aesthetic_suggestion = aesthetic_shade_suggestion(face_features, tooth_analysis)
        print("DEBUG: Simulated AI modules executed.")
        # ------------------------------------

        height, width, _ = img_corrected.shape # Get width as well for sanity check
        print(f"DEBUG: Corrected image dimensions: Height={height}, Width={width}")

        # Ensure image has enough height for slicing
        min_height_for_slicing = 30 # Adjust based on expected smallest tooth image size
        if height < min_height_for_slicing:
            print(f"ERROR: Image height ({height} pixels) is too small for zonal slicing. Minimum required: {min_height_for_slicing} pixels. Cannot perform detailed shade detection.")
            return {
                "incisal": "Error", "middle": "Error", "cervical": "Error",
                "overall_ml_shade": "Error - Image Too Small",
                "face_features": {}, "tooth_analysis": {}, "aesthetic_suggestion": {}
            }

        incisal_zone = img_corrected[0:int(height*0.3), :, :]
        middle_zone = img_corrected[int(height*0.3):int(height*0.7), :, :]
        cervical_zone = img_corrected[int(height*0.7):height, :, :]
        print("DEBUG: Image zones sliced.")

        # Basic check to ensure zones are not empty before converting color space
        if incisal_zone.size == 0 or middle_zone.size == 0 or cervical_zone.size == 0:
            print(f"ERROR: One or more sliced image zones are empty. Check image dimensions and slicing logic. Incisal size: {incisal_zone.size}, Middle size: {middle_zone.size}, Cervical size: {cervical_zone.size}")
            return {
                "incisal": "Error", "middle": "Error", "cervical": "Error",
                "overall_ml_shade": "Error - Zone Empty",
                "face_features": {}, "tooth_analysis": {}, "aesthetic_suggestion": {}
            }

        incisal_lab = cv2.cvtColor(incisal_zone, cv2.COLOR_BGR2LAB)
        middle_lab = cv2.cvtColor(middle_zone, cv2.COLOR_BGR2LAB)
        cervical_lab = cv2.cvtColor(cervical_zone, cv2.COLOR_BGR2LAB)
        print("DEBUG: Zones converted to LAB color space (L range 0-255).")

        avg_incisal_l_255 = np.mean(incisal_lab[:,:,0])
        avg_middle_l_255 = np.mean(middle_lab[:,:,0])
        avg_cervical_l_255 = np.mean(cervical_lab[:,:,0])
        print(f"DEBUG: Average L values (0-255 scale): Incisal={avg_incisal_l_255:.2f}, Middle={avg_middle_l_255:.2f}, Cervical={avg_cervical_l_255:.2f}")

        # Normalize L values to 0-100 scale for ML prediction and rule-based mapping consistency
        avg_incisal_l_100 = avg_incisal_l_255 / 2.55
        avg_middle_l_100 = avg_middle_l_255 / 2.55
        avg_cervical_l_100 = avg_cervical_l_255 / 2.55
        print(f"DEBUG: Average L values (0-100 scale): Incisal={avg_incisal_l_100:.2f}, Middle={avg_middle_l_100:.2f}, Cervical={avg_cervical_l_100:.2f}")

        # Use rule-based mapping for individual zones for UI consistency
        incisal_shade_rule = map_l_to_shade_rule_based(avg_incisal_l_100)
        middle_shade_rule = map_l_to_shade_rule_based(avg_middle_l_100)
        cervical_shade_rule = map_l_to_shade_rule_based(avg_cervical_l_100)
        print("DEBUG: Rule-based shades determined for zones.")

        overall_ml_shade = "Model Not Loaded or Error"
        if shade_classifier_model:
            try:
                # Predict overall shade using the loaded ML model
                # The model expects L values in the 0-100 scale, matching our normalization
                input_features = np.array([[avg_incisal_l_100, avg_middle_l_100, avg_cervical_l_100]])
                overall_ml_shade = shade_classifier_model.predict(input_features)[0]
                print(f"DEBUG: ML model predicted overall shade: {overall_ml_shade}")
            except Exception as e:
                print(f"ERROR: ML model prediction failed: {e}")
                traceback.print_exc()
                overall_ml_shade = "ML Prediction Error"
        else:
            print("WARNING: Shade classifier model is not available. ML prediction skipped.")

        return {
            "incisal": incisal_shade_rule,
            "middle": middle_shade_rule,
            "cervical": cervical_shade_rule,
            "overall_ml_shade": overall_ml_shade,
            "face_features": face_features,
            "tooth_analysis": tooth_analysis,
            "aesthetic_suggestion": aesthetic_suggestion
        }

    except Exception as e:
        print(f"ERROR: An unexpected error occurred during shade detection: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return {
            "incisal": "Error", "middle": "Error", "cervical": "Error",
            "overall_ml_shade": "Error - See Logs",
            "face_features": {}, "tooth_analysis": {}, "aesthetic_suggestion": {}
        }


def generate_pdf_report(patient_name, shades, image_filename, report_filename, analysis_date):
    """
    Generates a detailed PDF report for the shade analysis.
    Now includes enhanced AI insights.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="ShadeView Dental Analysis Report", ln=True, align="C")
    pdf.ln(5) # Add some space

    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 10, f"Analysis Date: {analysis_date}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Tooth Shade Analysis", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 7, f"Incisal Zone: {shades.get('incisal', 'N/A')}", ln=True)
    pdf.cell(0, 7, f"Middle Zone: {shades.get('middle', 'N/A')}", ln=True)
    pdf.cell(0, 7, f"Cervical Zone: {shades.get('cervical', 'N/A')}", ln=True)
    pdf.cell(0, 7, f"Overall ML Shade Recommendation: {shades.get('overall_ml_shade', 'N/A')}", ln=True)
    pdf.ln(5)

    # Add image to PDF
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    if os.path.exists(image_path):
        try:
            pdf.image(image_path, x=10, y=None, w=pdf.w - 20) # Use full width minus margins
            pdf.ln(5)
            pdf.cell(0, 5, "Analyzed Image:", ln=True)
            pdf.ln(5)
        except Exception as e:
            print(f"WARNING: Could not embed image {image_filename} in PDF: {e}")
            pdf.cell(0, 10, f"Image could not be embedded: {e}", ln=True)
    else:
        pdf.cell(0, 10, "Original image not found for embedding in report.", ln=True)
    pdf.ln(5)

    # --- Add Enhanced AI Insights ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Facial Harmony Analysis (Simulated)", ln=True)
    pdf.set_font("Arial", size=10)
    face_features = shades.get('face_features', {})
    pdf.cell(0, 7, f"Skin Tone: {face_features.get('skin_tone', 'N/A')}", ln=True)
    pdf.cell(0, 7, f"Lip Color: {face_features.get('lip_color', 'N/A')}", ln=True)
    pdf.cell(0, 7, f"Eye Contrast: {face_features.get('eye_contrast', 'N/A')}", ln=True)
    pdf.cell(0, 7, f"Facial Harmony Score: {face_features.get('facial_harmony_score', 'N/A')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Detailed Tooth Analysis (Simulated)", ln=True)
    pdf.set_font("Arial", size=10)
    tooth_analysis = shades.get('tooth_analysis', {})
    pdf.cell(0, 7, f"Overall LAB (L, a, b): {tooth_analysis.get('overall_lab', 'N/A')}", ln=True)
    pdf.cell(0, 7, f"Simulated Overall Shade: {tooth_analysis.get('simulated_overall_shade', 'N/A')}", ln=True)
    pdf.cell(0, 7, f"Tooth Condition: {tooth_analysis.get('tooth_condition', 'N/A')}", ln=True)
    pdf.cell(0, 7, f"Stain Presence: {tooth_analysis.get('stain_presence', 'N/A')}", ln=True)
    pdf.cell(0, 7, f"Decay Presence: {tooth_analysis.get('decay_presence', 'N/A')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Aesthetic Shade Suggestion (Simulated)", ln=True)
    pdf.set_font("Arial", size=10)
    aesthetic_suggestion = shades.get('aesthetic_suggestion', {})
    pdf.cell(0, 7, f"Suggested Shade: {aesthetic_suggestion.get('suggested_aesthetic_shade', 'N/A')}", ln=True)
    pdf.cell(0, 7, f"Confidence: {aesthetic_suggestion.get('aesthetic_confidence', 'N/A')}", ln=True)
    # Use multi_cell for longer notes
    pdf.multi_cell(0, 7, f"Recommendation Notes: {aesthetic_suggestion.get('recommendation_notes', 'N/A')}")
    pdf.ln(5)
    # --- End Enhanced AI Insights ---


    output_path = os.path.join(app.config['REPORT_FOLDER'], report_filename)
    pdf.output(output_path)
    return output_path


# ===============================================
# 6. FLASK ROUTES (UPDATED for Flask-SQLAlchemy)
# ===============================================

@app.route('/')
def index():
    if g.user and 'anonymous' not in str(g.user_id):
        return render_template('dashboard.html', user=g.user)
    return redirect(url_for('login'))


@app.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'
        elif not email:
            error = 'Email is required.'

        if error is None:
            # Check if username or email already exists in PostgreSQL
            existing_user = User.query.filter_by(username=username).first()
            existing_email = User.query.filter_by(email=email).first()

            if existing_user:
                error = f"User {username} is already registered."
            elif existing_email:
                error = f"Email {email} is already registered."
            else:
                # Create new user and add to PostgreSQL
                new_user = User(username=username, email=email)
                new_user.set_password(password) # This hashes the password
                db.session.add(new_user)
                db.session.commit() # Save the new user to the database
                flash('Registration successful!', 'success')
                return redirect(url_for('login'))
        flash(error, 'danger')
    return render_template('register.html')


@app.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None

        # Query user from PostgreSQL
        user = User.query.filter_by(username=username).first()

        if user is None:
            error = 'Incorrect username.'
        elif not user.check_password(password): # Checks hashed password
            error = 'Incorrect password.'

        if error is None:
            session.clear()
            session['user_id'] = user.id # Store user ID in session
            session['user'] = {'id': user.id, 'username': user.username} # Store user dict for g.user
            flash('You were successfully logged in!', 'success')
            return redirect(url_for('index')) # Or your main dashboard route
        flash(error, 'danger')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear() # Clear the entire session
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=g.user)


@app.route('/upload_page/<op_number>')
@login_required
def upload_page(op_number):
    # This route is intended to prepare the context for upload, e.g., for a specific patient/operation number
    return render_template('upload_file.html', op_number=op_number)


@app.route('/upload_file', methods=['POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        patient_name = request.form.get('patient_name')
        op_number = request.form.get('op_number', 'N/A') # Get op_number from form if available

        if not patient_name:
            flash('Patient Name is required.', 'danger')
            return redirect(request.url)

        if 'file' not in request.files:
            flash('No file part in the request.', 'danger')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file.', 'danger')
            return redirect(request.url)

        if file:
            unique_filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            print(f"DEBUG: File saved to {filepath}")

            # Perform shade detection and AI analysis
            detected_shades = detect_shades_from_image(filepath)
            formatted_analysis_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            report_filename = f"report_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
            pdf_path = generate_pdf_report(patient_name, detected_shades, unique_filename, report_filename, formatted_analysis_date)
            print(f"DEBUG: PDF report generated at {pdf_path}")

            # --- NEW: Save Patient Record to PostgreSQL ---
            if g.user_id: # Ensure a user is logged in
                try:
                    new_record = PatientRecord(
                        user_id=g.user_id,
                        patient_name=patient_name,
                        record_date=datetime.utcnow(), # Use UTC for consistency
                        uploaded_image_path=unique_filename,
                        analysis_results_json=json.dumps(detected_shades), # Store full analysis as JSON
                        # Saving individual shades for easier querying/display if needed
                        incisal_shade=detected_shades.get('incisal', 'N/A'),
                        middle_shade=detected_shades.get('middle', 'N/A'),
                        cervical_shade=detected_shades.get('cervical', 'N/A'),
                        overall_ml_shade=detected_shades.get('overall_ml_shade', 'N/A')
                        # Add other fields here if you extend PatientRecord model above
                    )
                    db.session.add(new_record)
                    db.session.commit()
                    flash('Image processed and patient record saved successfully!', 'success')
                except Exception as e:
                    db.session.rollback() # Rollback in case of database error
                    flash(f'Error saving patient record to database: {e}', 'danger')
                    print(f"ERROR: Failed to save patient record: {e}")
                    traceback.print_exc()
            else:
                flash('User not logged in, patient record could not be saved to database.', 'warning')

            return render_template('results.html',
                                   patient_name=patient_name,
                                   shades=detected_shades,
                                   image_filename=unique_filename, # Pass unique filename to template
                                   report_filename=report_filename,
                                   analysis_date=formatted_analysis_date) # Pass formatted date

    # Handle GET request for upload_file route (if a user directly navigates to /upload_file)
    return render_template('upload_file.html', op_number='N/A')


@app.route('/reports/<filename>')
@login_required # Protect report access
def download_report(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)


@app.route('/uploads/<filename>')
@login_required # Protect uploaded image access
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/patient_history')
@login_required
def patient_history():
    if g.user_id:
        try:
            # Retrieve records for the logged-in user from PostgreSQL, ordered by date
            user_records = PatientRecord.query.filter_by(user_id=g.user_id).order_by(PatientRecord.record_date.desc()).all()

            # Parse JSON analysis results for display in the template
            for record in user_records:
                if record.analysis_results_json:
                    record.parsed_analysis = json.loads(record.analysis_results_json)
                else:
                    record.parsed_analysis = {}
            return render_template('patient_history.html', records=user_records)
        except Exception as e:
            flash(f'Error retrieving patient history: {e}', 'danger')
            print(f"ERROR: Failed to retrieve patient history: {e}")
            traceback.print_exc()
            return render_template('patient_history.html', records=[]) # Render with empty list on error
    else:
        flash('Please log in to view your history.', 'warning')
        return redirect(url_for('login'))


if __name__ == '__main__':
    # This block is typically for local development only.
    # On Render, Gunicorn will manage running your app.
    # You generally don't need db.create_all() here if using Flask-Migrate in Render's build command.
    # with app.app_context():
    #     db.create_all() # Only for initial creation if NOT using Flask-Migrate
    app.run(debug=True) # Run in debug mode for local development