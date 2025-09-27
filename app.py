from flask import Flask, render_template, request, redirect, url_for, session, send_file
import joblib
import numpy as np
import sqlite3
import bcrypt
from datetime import datetime
from io import BytesIO
import pdfkit
import os
import requests
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key'

model = joblib.load('crop_recommendation_model.pkl')

# PDFKit setup
if os.name == 'nt':
    path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
else:
    path_wkhtmltopdf = '/usr/bin/wkhtmltopdf'
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

# Initialize DB
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            timestamp TEXT,
            predicted_crop TEXT,
            location TEXT,
            N REAL, P REAL, K REAL,
            temperature REAL, humidity REAL, ph REAL, rainfall REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Weather API
def get_weather(lat, lon):
    try:
        api_key = 'a901a1f72153e2ae8bbb3f978b3d3176'
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        return data['main']['temp'], data['main']['humidity']
    except:
        return 25, 50

# Geolocation
def get_location_name(lat, lon):
    try:
        geolocator = Nominatim(user_agent="crop_app")
        location = geolocator.reverse((lat, lon), language='en')
        if location:
            address = location.raw.get('address', {})
            city = address.get('city', address.get('town', address.get('village', '')))
            state = address.get('state', '')
            country = address.get('country', '')
            return f"{city}, {state}, {country}".strip(', ')
        return "Unknown"
    except:
        return "Unknown"

# Generate soil chart
def generate_soil_chart(N, P, K):
    labels = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
    values = [N, P, K]
    colors = ['#4CAF50', '#2196F3', '#FF9800']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)

    plt.ylabel('Normalized %')
    plt.title('Soil Nutrient Levels')

    # Correct Legend
    plt.legend(['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)'], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return image_base64

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = bcrypt.hashpw(request.form['password'].encode('utf-8'), bcrypt.gensalt())
        try:
            conn = sqlite3.connect('users.db')
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (uname, pwd))
            conn.commit()
            return redirect(url_for('login'))
        except:
            return "Username already exists!"
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password'].encode('utf-8')
        conn = sqlite3.connect('users.db')
        user = conn.execute("SELECT * FROM users WHERE username=?", (uname,)).fetchone()
        conn.close()
        if user and bcrypt.checkpw(pwd, user[2]):
            session['user'] = uname
            return redirect(url_for('recommend'))
        return "Invalid credentials"
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/recommend')
def recommend():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('recommend.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        lat = float(request.form['latitude'])
        lon = float(request.form['longitude'])
        temp, hum = get_weather(lat, lon)
        loc = get_location_name(lat, lon)

        form_data = {
            'N': float(request.form['N']),
            'P': float(request.form['P']),
            'K': float(request.form['K']),
            'ph': float(request.form['ph']),
            'rainfall': float(request.form['rainfall']),
            'temperature': temp,
            'humidity': hum
        }

        features = [form_data[key] for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        pred_crop = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        classes = model.classes_

        top_indices = np.argsort(probabilities)[::-1][:3]
        top_crops = [{'crop': classes[i], 'prob': round(probabilities[i] * 100, 2)} for i in top_indices]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = sqlite3.connect('users.db')
        conn.execute('''INSERT INTO history (username, timestamp, predicted_crop, location, N, P, K, temperature, humidity, ph, rainfall)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (session['user'], timestamp, pred_crop, loc,
                      form_data['N'], form_data['P'], form_data['K'], temp, hum, form_data['ph'], form_data['rainfall']))
        conn.commit()
        conn.close()

        soil_chart = generate_soil_chart(form_data['N'], form_data['P'], form_data['K'])

        return render_template("report.html", user=session['user'], input_values=form_data,
                               prediction=pred_crop, top_crops=top_crops, location=loc,
                               timestamp=timestamp, soil_chart=soil_chart)

    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = session['user']
    query = request.args.get('query', '').lower()

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM history WHERE username = ? ORDER BY timestamp DESC", (user,))
    all_records = c.fetchall()
    conn.close()

    if query:
        records = [row for row in all_records if query in row[3].lower()]
    else:
        records = all_records

    return render_template('history.html', records=records)

@app.route('/download_report', methods=['POST'])
def download_report():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        form = request.form
        form_values = {
            'N': float(form['N']),
            'P': float(form['P']),
            'K': float(form['K']),
            'temperature': float(form['temperature']),
            'humidity': float(form['humidity']),
            'ph': float(form['ph']),
            'rainfall': float(form['rainfall'])
        }

        location = form['location']
        prediction = form['predicted_crop']
        timestamp = form['timestamp']

        probabilities = model.predict_proba([[form_values[key] for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]])[0]
        classes = model.classes_
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_crops = [{'crop': classes[i], 'prob': round(probabilities[i] * 100, 2)} for i in top_indices]

        soil_chart = generate_soil_chart(form_values['N'], form_values['P'], form_values['K'])

        rendered_html = render_template("report.html", user=session['user'],
                                        input_values=form_values, prediction=prediction,
                                        top_crops=top_crops, location=location,
                                        timestamp=timestamp, soil_chart=soil_chart)

        options = {'enable-local-file-access': ''}
        pdf = pdfkit.from_string(rendered_html, False, configuration=config, options=options)

        return send_file(BytesIO(pdf), as_attachment=True, download_name="Crop_Recommendation_Report.pdf", mimetype='application/pdf')

    except Exception as e:
        return f"Error in generating PDF report: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
import matplotlib
matplotlib.use('Agg')  # ✅ Use non-GUI backend to avoid Tkinter errors

from flask import Flask, render_template, request, redirect, url_for, session, send_file
import joblib
import numpy as np
import sqlite3
import bcrypt
from datetime import datetime
from io import BytesIO
import pdfkit
import os
import requests
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import pandas as pd
import base64

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load trained model
model = joblib.load('crop_recommendation_model.pkl')

# PDFKit setup
if os.name == 'nt':
    path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
else:
    path_wkhtmltopdf = '/usr/bin/wkhtmltopdf'
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

# Initialize DB
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            timestamp TEXT,
            predicted_crop TEXT,
            location TEXT,
            N REAL, P REAL, K REAL,
            temperature REAL, humidity REAL, ph REAL, rainfall REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Weather API
def get_weather(lat, lon):
    try:
        api_key = 'a901a1f72153e2ae8bbb3f978b3d3176'
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        return data['main']['temp'], data['main']['humidity']
    except:
        return 25, 50

# Geolocation
def get_location_name(lat, lon):
    try:
        geolocator = Nominatim(user_agent="crop_app")
        location = geolocator.reverse((lat, lon), language='en')
        if location:
            address = location.raw.get('address', {})
            city = address.get('city', address.get('town', address.get('village', '')))
            state = address.get('state', '')
            country = address.get('country', '')
            return f"{city}, {state}, {country}".strip(', ')
        return "Unknown"
    except:
        return "Unknown"

# Generate soil chart
def generate_soil_chart(N, P, K):
    labels = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
    values = [N, P, K]
    colors = ['#888888', '#AAAAAA', '#CCCCCC']  # ✅ Softer grayscale colors

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors, edgecolor='black')

    plt.ylabel('Normalized %', color='black')
    plt.title('Soil Nutrient Levels', color='black')

    # ✅ Legend in black for better visibility in grayscale
    plt.legend(['N', 'P', 'K'], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, facecolor='white', edgecolor='black')

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return image_base64

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = bcrypt.hashpw(request.form['password'].encode('utf-8'), bcrypt.gensalt())
        try:
            conn = sqlite3.connect('users.db')
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (uname, pwd))
            conn.commit()
            return redirect(url_for('login'))
        except:
            return "Username already exists!"
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password'].encode('utf-8')
        conn = sqlite3.connect('users.db')
        user = conn.execute("SELECT * FROM users WHERE username=?", (uname,)).fetchone()
        conn.close()
        if user and bcrypt.checkpw(pwd, user[2]):
            session['user'] = uname
            return redirect(url_for('recommend'))
        return "Invalid credentials"
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

@app.route('/recommend')
def recommend():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('recommend.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        lat = float(request.form['latitude'])
        lon = float(request.form['longitude'])
        temp, hum = get_weather(lat, lon)
        loc = get_location_name(lat, lon)

        form_data = {
            'N': float(request.form['N']),
            'P': float(request.form['P']),
            'K': float(request.form['K']),
            'ph': float(request.form['ph']),
            'rainfall': float(request.form['rainfall']),
            'temperature': temp,
            'humidity': hum
        }

        # ✅ FIX: Wrap features in DataFrame with correct column names
        input_df = pd.DataFrame([[form_data['N'], form_data['P'], form_data['K'],
                                  form_data['temperature'], form_data['humidity'],
                                  form_data['ph'], form_data['rainfall']]],
                                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        pred_crop = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_

        top_indices = np.argsort(probabilities)[::-1][:3]
        top_crops = [{'crop': classes[i], 'prob': round(probabilities[i] * 100, 2)} for i in top_indices]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = sqlite3.connect('users.db')
        conn.execute('''INSERT INTO history (username, timestamp, predicted_crop, location, N, P, K, temperature, humidity, ph, rainfall)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (session['user'], timestamp, pred_crop, loc,
                      form_data['N'], form_data['P'], form_data['K'], temp, hum, form_data['ph'], form_data['rainfall']))
        conn.commit()
        conn.close()

        soil_chart = generate_soil_chart(form_data['N'], form_data['P'], form_data['K'])

        return render_template("report.html", user=session['user'], input_values=form_data,
                               prediction=pred_crop, top_crops=top_crops, location=loc,
                               timestamp=timestamp, soil_chart=soil_chart)

    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = session['user']
    query = request.args.get('query', '').lower()

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM history WHERE username = ? ORDER BY timestamp DESC", (user,))
    all_records = c.fetchall()
    conn.close()

    if query:
        records = [row for row in all_records if query in row[3].lower()]
    else:
        records = all_records

    return render_template('history.html', records=records)

@app.route('/download_report', methods=['POST'])
def download_report():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        form = request.form
        form_values = {
            'N': float(form['N']),
            'P': float(form['P']),
            'K': float(form['K']),
            'temperature': float(form['temperature']),
            'humidity': float(form['humidity']),
            'ph': float(form['ph']),
            'rainfall': float(form['rainfall'])
        }

        location = form['location']
        prediction = form['predicted_crop']
        timestamp = form['timestamp']

        input_df = pd.DataFrame([[form_values['N'], form_values['P'], form_values['K'],
                                  form_values['temperature'], form_values['humidity'],
                                  form_values['ph'], form_values['rainfall']]],
                                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_crops = [{'crop': classes[i], 'prob': round(probabilities[i] * 100, 2)} for i in top_indices]

        soil_chart = generate_soil_chart(form_values['N'], form_values['P'], form_values['K'])

        rendered_html = render_template("report.html", user=session['user'],
                                        input_values=form_values, prediction=prediction,
                                        top_crops=top_crops, location=location,
                                        timestamp=timestamp, soil_chart=soil_chart)

        options = {'enable-local-file-access': ''}
        pdf = pdfkit.from_string(rendered_html, False, configuration=config, options=options)

        return send_file(BytesIO(pdf), as_attachment=True, download_name="Crop_Recommendation_Report.pdf", mimetype='application/pdf')

    except Exception as e:
        return f"Error in generating PDF report: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
