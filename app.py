import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Import configuration
from config import Config

# Import database modules
from database.db import init_db, db
from database.models import User, Scan, RecyclingLocation, Product

from models.classifier import WasteClassifier

# Remove the global model loading
# model_path = 'models/saved_model/waste_classifier.h5'
# waste_classifier = WasteClassifier(model_path=model_path)

app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
init_db(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create a function to get the classifier
def get_classifier():
    model_path = app.config['MODEL_PATH']
    return WasteClassifier(model_path=model_path)

# Login required decorator
def login_required(f):
    def wrap(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

# =========================================================
# Web Routes (HTML Pages)
# =========================================================

@app.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """Render the dashboard page"""
    user = User.query.get(session['user_id'])
    return render_template('dashboard.html', user=user)

@app.route('/marketplace')
def marketplace():
    """Render the marketplace page"""
    # Get products from the database
    db_products = Product.query.order_by(Product.created_at.desc()).all()
    
    # Convert database products to dictionary format
    products_from_db = []
    for product in db_products:
        products_from_db.append({
            'id': product.id,
            'title': product.title,
            'price': product.price,
            'image': product.image_path,
            'description': product.description,
            'waste_type': product.waste_type,
            'seller': User.query.get(product.user_id).username,
            'created_at': product.created_at
        })
    
    # Static products for demonstration
    static_products = [
        {
            'id': 1001,
            'title': 'Recycled Plastic Bottle',
            'price': 0.50,
            'image': 'marketplace/product1.jpg',
            'description': 'A high-quality recycled plastic bottle.',
            'waste_type': 'recyclable',
            'seller': 'EcoLoop',
            'created_at': datetime(2025, 4, 1)
        },
        {
            'id': 1002,
            'title': 'Compostable Bag',
            'price': 1.20,
            'image': 'marketplace/product2.jpg',
            'description': 'Eco-friendly compostable bag made from sustainable materials.',
            'waste_type': 'compostable',
            'seller': 'EcoLoop',
            'created_at': datetime(2025, 4, 1)
        }
    ]
    
    # Combine both product lists
    all_products = products_from_db + static_products
    
    # Sort by most recent
    all_products.sort(key=lambda x: x['created_at'], reverse=True)
    
    return render_template('marketplace.html', products=all_products)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Validate form data
        if not all([username, password]):
            flash('Please fill in all fields.', 'error')
            return render_template('login.html')

        # Check if user exists and password matches
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            # Update last login
            user.last_login = datetime.utcnow()
            db.session.commit()
            # Set user session
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')
            return render_template('login.html')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration"""
    if request.method == 'POST':
        # Get form data
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        # Validate form data
        if not all([username, email, password, confirm_password]):
            flash('Please fill in all fields.', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
            
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            return render_template('register.html')
            
        if User.query.filter_by(email=email).first():
            flash('Email already exists.', 'error')
            return render_template('register.html')
            
        # Create new user
        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        # Add user to database
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')



@app.route('/logout')
def logout():
    """Handle user logout"""
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    """Render the contact page and handle form submission"""
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        # Process contact form submission
        flash('Your message has been sent!', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/product/<int:product_id>')
def product_detail(product_id):
    """Render the product detail page"""
    # Check if it's a database product
    product = Product.query.get(product_id)
    
    if product:
        product_data = {
            'id': product.id,
            'title': product.title,
            'price': product.price,
            'image': product.image_path,
            'description': product.description,
            'waste_type': product.waste_type,
            'seller': User.query.get(product.user_id).username,
            'created_at': product.created_at
        }
    else:
        # Check if it's a static product
        static_products = {
            1001: {
                'id': 1001,
                'title': 'Recycled Plastic Bottle',
                'price': 0.50,
                'image': 'marketplace/product1.jpg',
                'description': 'A high-quality recycled plastic bottle.',
                'waste_type': 'recyclable',
                'seller': 'EcoSort',
                'created_at': datetime(2025, 4, 1)
            },
            1002: {
                'id': 1002,
                'title': 'Compostable Bag',
                'price': 1.20,
                'image': 'marketplace/product2.jpg',
                'description': 'Eco-friendly compostable bag made from sustainable materials.',
                'waste_type': 'compostable',
                'seller': 'EcoSort',
                'created_at': datetime(2025, 4, 1)
            },
            1003: {
                'id': 1003,
                'title': 'Upcycled Furniture',
                'price': 75.00,
                'image': 'marketplace/product3.jpg',
                'description': 'Beautifully upcycled furniture piece with a modern design.',
                'waste_type': 'recyclable',
                'seller': 'EcoSort',
                'created_at': datetime(2025, 4, 1)
            }
        }
        
        product_data = static_products.get(product_id)
        
    if not product_data:
        flash('Product not found.', 'error')
        return redirect(url_for('marketplace'))
        
    return render_template('product_detail.html', product=product_data)

@app.route('/buyer_dashboard')
@login_required
def buyer_dashboard():
    """Render the buyer dashboard page"""
    user = User.query.get(session['user_id'])
    return render_template('buyer_dashboard.html', user=user)

@app.route('/seller_dashboard')
@login_required
def seller_dashboard():
    """Render the seller dashboard page"""
    user = User.query.get(session['user_id'])
    return render_template('seller_dashboard.html', user=user)

@app.route('/classification_dashboard', methods=['GET', 'POST'])
def classification_dashboard():
    """Render the classification dashboard page"""
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image file provided', 'error')
            return redirect(request.url)
            
        file = request.files['image']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
            
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get classifier instance
            classifier = get_classifier()
            
            # Process image for classification
            class_label, class_index, confidence = classifier.predict(filepath)
            
            # Save scan to database if user is logged in
            if 'user_id' in session:
                scan = Scan(
                    user_id=session['user_id'],
                    image_path=filepath,
                    classification=class_label,
                    confidence=confidence
                )
                
                db.session.add(scan)
                db.session.commit()
                
                return render_template('classification_result.html', scan=scan)
            else:
                # For non-logged in users, just show the result without saving
                scan = Scan(
                    image_path=filepath,
                    classification=class_label,
                    confidence=confidence,
                    created_at=datetime.utcnow()
                )
                return render_template('classification_result.html', scan=scan)
    
    return render_template('classification_dashboard.html')

# =========================================================
# API Routes (JSON Endpoints)
# =========================================================

@app.route('/api/classify', methods=['POST'])
def api_classify():
    """API endpoint for classifying waste images"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get classifier instance
            classifier = get_classifier()
            
            # Get prediction
            class_label, class_index, confidence = classifier.predict(filepath)
            
            # Save scan to database if user is logged in
            if 'user_id' in session:
                scan = Scan(
                    user_id=session['user_id'],
                    image_path=filepath,
                    classification=class_label,
                    confidence=confidence
                )
                
                db.session.add(scan)
                db.session.commit()
                
            return jsonify({
                'success': True,
                'classification': class_label,
                'confidence': confidence,
                'waste_type': class_label  # Add this for frontend compatibility
            })
            
    except Exception as e:
        print(f"Classification error: {str(e)}")
        return jsonify({'error': 'An error occurred while classifying the image'}), 500



@app.route('/api/recycling-centers', methods=['GET'])
def api_recycling_centers():
    """API endpoint for finding nearby recycling centers"""
    try:
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        waste_type = request.args.get('type', 'recyclable')
        
        if not lat or not lng:
            return jsonify({'error': 'Latitude and longitude are required'}), 400
        
        # For demonstration, return dummy data
        # In a real app, you would query a database or external API
        centers = [
            {
                'name': 'City Recycling Center',
                'address': 'Tamil Nadu',
                'latitude': lat + 0.01,
                'longitude': lng + 0.01,
                'distance': 1.2,
                'accepts': ['recyclable', 'compostable']  # Changed to array for frontend
            },
            {
                'name': 'Community Waste Management',
                'address': 'Chengalpattu',
                'latitude': lat - 0.01,
                'longitude': lng - 0.01,
                'distance': 2.5,
                'accepts': ['recyclable', 'general_waste']  # Changed to array for frontend
            }
        ]
        
        return jsonify(centers)
    except Exception as e:
        print(f"Error finding recycling centers: {str(e)}")
        return jsonify({'error': 'An error occurred while finding recycling centers'}), 500


@app.route('/api/create-product', methods=['POST'])
@login_required
def api_create_product():
    """API endpoint for creating a product from a waste scan"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['scan_id', 'title', 'description', 'price']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Get the scan
        scan = Scan.query.get(data['scan_id'])
        if not scan:
            return jsonify({'error': 'Scan not found'}), 404
            
        # Verify the scan belongs to the current user
        if scan.user_id != session['user_id']:
            return jsonify({'error': 'Unauthorized access to scan'}), 403
            
        # Create a new product
        product = Product(
            title=data['title'],
            description=data['description'],
            price=float(data['price']),
            image_path=scan.image_path,
            waste_type=scan.classification,
            user_id=session['user_id'],
            scan_id=scan.id
        )
        
        db.session.add(product)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'product_id': product.id,
            'message': 'Product created successfully'
        })
        
    except Exception as e:
        print(f"Error creating product: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'An error occurred while creating the product'}), 500

# =========================================================
# Error Handlers
# =========================================================

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000, debug=True)
