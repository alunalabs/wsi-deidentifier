#!/usr/bin/env python3

import csv
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import base64

app = Flask(__name__)

def load_mappings():
    """Load mappings from CSV file"""
    mappings = []
    csv_file = 'filename_to_id_mapping.csv'
    
    if not os.path.exists(csv_file):
        return mappings
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mappings.append({
                'filename': row['filename'],
                'extracted_id': row['extracted_id']
            })
    
    return mappings

def save_mappings(mappings):
    """Save mappings back to CSV file"""
    csv_file = 'filename_to_id_mapping.csv'
    
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['filename', 'extracted_id']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mapping in mappings:
            writer.writerow(mapping)

@app.route('/')
def index():
    """Main page showing all mappings"""
    mappings = load_mappings()
    return render_template('index.html', mappings=mappings)

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve PNG images"""
    return send_from_directory('.', filename)

@app.route('/update_id', methods=['POST'])
def update_id():
    """Update an ID mapping"""
    data = request.get_json()
    filename = data.get('filename')
    new_id = data.get('new_id')
    
    mappings = load_mappings()
    
    # Update the mapping
    for mapping in mappings:
        if mapping['filename'] == filename:
            mapping['extracted_id'] = new_id
            break
    
    # Save back to file
    save_mappings(mappings)
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)