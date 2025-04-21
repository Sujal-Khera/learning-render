# services/location_service.py
import random

def find_recycling_centers(lat, lng, waste_type='recyclable', radius=5000):
    """
    Find recycling centers near a location
    Parameters:
        lat (float): Latitude
        lng (float): Longitude
        waste_type (str): Type of waste to recycle
        radius (int): Search radius in meters
    Returns:
        list: List of recycling centers
    """
    # In a real app, you would use a geolocation API or database query
    # For demo purposes, return dummy data
    centers = [
        {
            'name': 'EcoRecycle Center',
            'address': '123 Green St, Eco City',
            'latitude': lat + random.uniform(-0.01, 0.01),
            'longitude': lng + random.uniform(-0.01, 0.01),
            'distance': random.uniform(0.5, 4.5),
            'accepts': 'recyclable,compostable',
            'rating': 4.5
        },
        {
            'name': 'City Recycling Facility',
            'address': '456 Earth Ave, Eco City',
            'latitude': lat + random.uniform(-0.01, 0.01),
            'longitude': lng + random.uniform(-0.01, 0.01),
            'distance': random.uniform(0.5, 4.5),
            'accepts': 'recyclable,general_waste',
            'rating': 4.0
        },
        {
            'name': 'Green Future Recycling',
            'address': '789 Sustainability Blvd, Eco City',
            'latitude': lat + random.uniform(-0.01, 0.01),
            'longitude': lng + random.uniform(-0.01, 0.01),
            'distance': random.uniform(0.5, 4.5),
            'accepts': 'recyclable,compostable,general_waste',
            'rating': 4.8
        }
    ]
    
    # Filter centers by waste type
    if waste_type:
        centers = [center for center in centers if waste_type in center['accepts']]
    
    return centers
