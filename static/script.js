document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const imageUpload = document.getElementById('imageUpload');
    const uploadBtn = document.getElementById('uploadBtn');
    const preview = document.getElementById('preview');
    const resultContainer = document.getElementById('resultContainer');
    const classificationResult = document.getElementById('classificationResult');
    const wasteType = document.getElementById('wasteType');
    const confidence = document.getElementById('confidence');
    const instructions = document.getElementById('instructions');
    const locationStatus = document.getElementById('locationStatus');
    const recyclingLocations = document.getElementById('recyclingLocations');
    const loadingSpinner = document.getElementById('loadingSpinner');

    // Preview image when selected
    if (imageUpload) {
        imageUpload.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                }
                reader.readAsDataURL(file);
            }
        });
    }

    // Handle image upload and classification
    if (uploadBtn) {
        uploadBtn.addEventListener('click', async function() {
            const file = imageUpload.files[0];
            if (!file) {
                alert('Please select an image first.');
                return;
            }

            // Show loading spinner
            loadingSpinner.style.display = 'inline-block';
            uploadBtn.disabled = true;
            resultContainer.style.display = 'none';

            try {
                // Upload and classify image
                const formData = new FormData();
                formData.append('image', file);
                
                const response = await fetch('/api/classify', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`Server responded with status: ${response.status}`);
                }

                const result = await response.json();
                console.log('Classification result:', result);

                if (result.error) {
                    throw new Error(result.error);
                }

                // Display classification results
                displayClassificationResult(result);

                // Get user location and find recycling centers
                if (result.classification) {
                    getLocationAndFindCenters(result.classification);
                }
            } catch (error) {
                console.error('Error:', error);
                classificationResult.className = 'alert alert-danger';
                classificationResult.textContent = 'An error occurred during classification. Please try again.';
                resultContainer.style.display = 'block';
            } finally {
                loadingSpinner.style.display = 'none';
                uploadBtn.disabled = false;
            }
        });
    }

    // Display classification results
    function displayClassificationResult(result) {
        
        if (!resultContainer) return;
        
        resultContainer.style.display = 'block';
        
        const wasteTypeInfo = {
            'recyclable': {
                color: 'success',
                instructions: 'This item can be recycled. Please make sure it is clean and dry before recycling.'
            },
            'compostable': {
                color: 'warning',
                instructions: 'This item can be composted. Place it in your compost bin or take it to a composting facility.'
            },
            'general_waste': {
                color: 'danger',
                instructions: 'This item should go in the general waste bin. It cannot be recycled or composted.'
            }
        };
        
        // Use classification from the response
        const type = result.waste_type || result.classification;
        const info = wasteTypeInfo[type] || {
            color: 'info',
            instructions: 'Please check local guidelines for disposal.'
        };
        
        classificationResult.className = `alert alert-${info.color}`;
        classificationResult.textContent = `This item is classified as ${type.replace('_', ' ')}.`;
        
        if (wasteType) wasteType.textContent = type.replace('_', ' ');
        if (confidence) confidence.textContent = `${(result.confidence * 100).toFixed(2)}%`;
        if (instructions) instructions.innerHTML = info.instructions;

        // In displayClassificationResult function
    }

    // Get user location and find nearby recycling centers
    async function getLocationAndFindCenters(wasteType) {
        if (!locationStatus) return;
        
        locationStatus.innerHTML = 'Searching for nearby recycling centers...';
        
        try {
            // Get user's geolocation
            const position = await new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(resolve, reject, {
                    enableHighAccuracy: true,
                    timeout: 5000,
                    maximumAge: 0
                });
            });
            
            const { latitude, longitude } = position.coords;
            
            // Call API to find nearby recycling centers
            const response = await fetch(`/api/recycling-centers?lat=${latitude}&lng=${longitude}&type=${wasteType}`);
            
            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            
            const centers = await response.json();
            displayRecyclingCenters(centers);
        } catch (error) {
            console.error('Location error:', error);
            locationStatus.innerHTML = 'Could not access your location or find recycling centers.';
        }
    }

    // Display nearby recycling centers
    function displayRecyclingCenters(centers) {
        if (!locationStatus || !recyclingLocations) return;
        
        if (!centers || centers.length === 0) {
            locationStatus.innerHTML = 'No recycling centers found nearby.';
            return;
        }
        
        locationStatus.innerHTML = `Found ${centers.length} recycling centers near you:`;
        
        let locationsHTML = '';
        centers.forEach(center => {
            const acceptsText = Array.isArray(center.accepts) 
                ? center.accepts.join(', ') 
                : center.accepts;
                
            locationsHTML += `
                <div class="card mb-3">
                    <div class="card-body">
                        <h5 class="card-title">${center.name || 'Recycling Center'}</h5>
                        <p class="card-text">${center.address}</p>
                        <p class="card-text"><strong>Distance:</strong> ${center.distance.toFixed(1)} km</p>
                        <a href="https://maps.google.com/?q=${center.latitude},${center.longitude}" 
                           class="btn btn-sm btn-primary" target="_blank">Get Directions</a>
                        <p class="card-text mt-2"><strong>Accepts:</strong> ${acceptsText}</p>
                    </div>
                </div>
            `;
        });
        
        recyclingLocations.innerHTML = locationsHTML;
    }
});
