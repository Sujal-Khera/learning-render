from datetime import datetime
from .db import db

class User(db.Model):
    """User model for storing user account information"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)


    # Relationships
    products = db.relationship('Product', backref='seller', lazy=True, cascade="all, delete-orphan")
    scans = db.relationship('Scan', backref='user', lazy=True, cascade="all, delete-orphan")

    
    def __repr__(self):
        return f'<User {self.username}>'

class Scan(db.Model):
    """Model for storing waste image scans and classifications"""
    __tablename__ = 'scans'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    image_path = db.Column(db.String(256), nullable=False)
    classification = db.Column(db.String(50), nullable=False)  # recyclable, compostable, general_waste
    confidence = db.Column(db.Float, nullable=False)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    #Relationship
    products = db.relationship('Product', backref='scan', lazy=True)
    
    def __repr__(self):
        return f'<Scan {self.id}: {self.classification}>'


class Product(db.Model):
    """Model for storing marketplace products created from waste items"""
    __tablename__ = 'products'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text, nullable=False)
    price = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(256), nullable=False)
    waste_type = db.Column(db.String(50), nullable=False)  # recyclable, compostable, general_waste
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    scan_id = db.Column(db.Integer, db.ForeignKey('scans.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Product {self.id}: {self.title}>'


class RecyclingLocation(db.Model):
    """Model for storing recycling center locations"""
    __tablename__ = 'recycling_locations'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    address = db.Column(db.String(256), nullable=False)
    latitude = db.Column(db.Float, nullable=False, index=True)
    longitude = db.Column(db.Float, nullable=False, index=True)
    accepts = db.Column(db.String(256), nullable=False)  # Comma-separated list of accepted waste types
    rating = db.Column(db.Float, nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    website = db.Column(db.String(256), nullable=True)
    hours = db.Column(db.String(512), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<RecyclingLocation {self.name}>'
