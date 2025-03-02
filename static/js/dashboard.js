// Refresh intervals (in milliseconds)
const REFRESH_INTERVALS = {
    assets: 10000,  // 10 seconds
    alerts: 5000,   // 5 seconds
    logs: 15000     // 15 seconds
};

// Store the interval IDs for cleanup
let intervalIds = {};

// Format timestamp to readable format
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString();
}

// Create status badge HTML
function createStatusBadge(status) {
    return `<span class="status-badge status-${status.toLowerCase()}">${status}</span>`;
}

// Format confidence value with color coding
function formatConfidence(value) {
    const percentage = (value * 100).toFixed(1);
    let colorClass = 'confidence-medium';
    if (value >= 0.8) colorClass = 'confidence-high';
    if (value < 0.5) colorClass = 'confidence-low';
    return `<span class="${colorClass}">${percentage}%</span>`;
}

// Update table data
function updateTable(target, data) {
    const tableBody = $(`#${target}-table tbody`);
    tableBody.empty();

    data.forEach(item => {
        let row = '';
        switch(target) {
            case 'assets':
                row = `
                    <tr>
                        <td>${item.id}</td>
                        <td>${createStatusBadge(item.status)}</td>
                        <td>${(item.confidence * 100).toFixed(1)}%</td>
                    </tr>
                `;
                break;
            case 'alerts':
                row = `
                    <tr>
                        <td>${formatTimestamp(item.timestamp)}</td>
                        <td>${item.type}</td>
                        <td>${createStatusBadge(item.status)}</td>
                    </tr>
                `;
                break;
            case 'logs':
                row = `
                    <tr>
                        <td>${formatTimestamp(item.timestamp)}</td>
                        <td>${item.event}</td>
                        <td>${item.details}</td>
                    </tr>
                `;
                break;
        }
        tableBody.append(row);
    });
}

// Fetch data from API
function fetchData(target) {
    $.ajax({
        url: `/api/${target}`,
        method: 'GET',
        success: function(response) {
            if (response.status === 'success') {
                updateTable(target, response.data);
            } else {
                console.error(`Error fetching ${target}:`, response.message);
            }
        },
        error: function(xhr, status, error) {
            console.error(`Error fetching ${target}:`, error);
        }
    });
}

// Setup auto-refresh for each section
function setupAutoRefresh(target) {
    // Clear existing interval if any
    if (intervalIds[target]) {
        clearInterval(intervalIds[target]);
    }
    
    // Initial fetch
    fetchData(target);
    
    // Setup new interval
    intervalIds[target] = setInterval(() => {
        fetchData(target);
    }, REFRESH_INTERVALS[target]);
}

// Display analysis results
function displayAnalysisResults(results) {
    // Watermark Verification Results
    const watermarkHtml = `
        <div class="metric-item">
            <span class="metric-label">Watermark Status:</span>
            <span class="metric-value">${results.watermark_verification.has_watermark ? 'Detected ✅' : 'Not Detected ❌'}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Authenticity:</span>
            <span class="metric-value">${results.watermark_verification.is_authentic ? 'Authentic ✅' : 'Not Verified ❓'}</span>
        </div>
        ${results.watermark_verification.watermark_strength !== undefined ? `
        <div class="metric-item">
            <span class="metric-label">Watermark Strength:</span>
            <span class="metric-value">${(results.watermark_verification.watermark_strength * 100).toFixed(1)}%</span>
        </div>
        ` : ''}
        ${results.watermark_verification.in_registry ? `
        <div class="metric-item">
            <span class="metric-label">Registry Status:</span>
            <span class="metric-value">Image in Registry ✅</span>
        </div>
        ` : ''}
        <div class="metric-item">
            <span class="metric-label">Original Image:</span>
            <div class="metric-value">
                <img src="${results.watermark_verification.original_path}" class="img-fluid rounded" style="max-height: 150px;" onerror="this.src='https://via.placeholder.com/150?text=Image+Not+Found'">
            </div>
        </div>
        <div class="metric-item">
            <span class="metric-label">Watermarked Image:</span>
            <div class="metric-value">
                <img src="${results.watermark_verification.watermarked_path}" class="img-fluid rounded" style="max-height: 150px;" onerror="this.src='https://via.placeholder.com/150?text=Image+Not+Found'">
            </div>
        </div>
        ${results.watermark_verification.metadata ? `
        <div class="metric-item">
            <span class="metric-label">Process:</span>
            <span class="metric-value">${results.watermark_verification.metadata.process || 'N/A'}</span>
        </div>
        ` : ''}
    `;
    $('#watermark-results').html(watermarkHtml);

    // Deepfake Detection Results (if available)
    if (results.deepfake_detection) {
        const predictions = results.deepfake_detection.model_predictions || {};
        const predictionHtml = Object.entries(predictions)
            .map(([model, score]) => `
                <div class="metric-item">
                    <span class="metric-label">${model}:</span>
                    <span class="metric-value">${(score * 100).toFixed(1)}%</span>
                </div>
            `).join('');

        const deepfakeHtml = `
            <div class="metric-item">
                <span class="metric-label">Detection Status:</span>
                <span class="metric-value ${results.deepfake_detection.is_deepfake ? 'text-danger' : 'text-success'}">
                    ${results.deepfake_detection.is_deepfake ? 'Deepfake Detected ⚠️' : 'Authentic ✅'}
                </span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Confidence:</span>
                <span class="metric-value">${(results.deepfake_detection.confidence * 100).toFixed(1)}%</span>
            </div>
            ${results.deepfake_detection.whitelisted ? `
            <div class="metric-item">
                <span class="metric-label">Whitelist Status:</span>
                <span class="metric-value text-success">Image Whitelisted ✅</span>
            </div>
            ` : ''}
            ${predictionHtml}
        `;
        $('#deepfake-results').html(deepfakeHtml);
    }

    // Remove the threat intelligence section as it's not part of the workflow
    $('#threat-results').closest('.analysis-section').remove();
    
    // Show the analysis results
    $('#analysis-results').removeClass('d-none');
}

// Handle file input change
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            $('#image-preview').attr('src', e.target.result);
            $('#preview-container').removeClass('d-none');
        };
        reader.readAsDataURL(file);
    }
}

// Handle form submission
function handleFormSubmit(event) {
    event.preventDefault();
    
    const fileInput = $('#file-input')[0];
    if (!fileInput.files.length) {
        alert('Please select an image to analyze');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    // Show loading indicator
    $('#analysis-results').addClass('d-none');
    $('#loading-indicator').removeClass('d-none');

    $.ajax({
        url: '/api/upload',
        method: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            if (response.status === 'success') {
                displayAnalysisResults(response.data.analysis);
                $('#analysis-results').removeClass('d-none');
                // Refresh assets table
                fetchData('assets');
            } else {
                alert('Error analyzing image: ' + response.message);
            }
        },
        error: function(xhr, status, error) {
            alert('Error uploading image: ' + error);
        },
        complete: function() {
            $('#loading-indicator').addClass('d-none');
        }
    });
}

// Initialize dashboard
function initDashboard() {
    // Setup auto-refresh for all sections
    Object.keys(REFRESH_INTERVALS).forEach(target => {
        setupAutoRefresh(target);
    });

    // Handle manual refresh button clicks
    $('.refresh-btn').click(function() {
        const target = $(this).data('target');
        fetchData(target);
        
        // Add rotation animation
        $(this).find('i').addClass('fa-spin');
        setTimeout(() => {
            $(this).find('i').removeClass('fa-spin');
        }, 1000);
    });

    // Setup file input handler
    $('#file-input').change(handleFileSelect);

    // Setup form submit handler
    $('#upload-form').submit(handleFormSubmit);
}

// Start the dashboard when the document is ready
$(document).ready(function() {
    initDashboard();
});

// Image protection functionality
document.getElementById('protect-file-input').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const preview = document.getElementById('protect-image-preview');
        preview.src = URL.createObjectURL(file);
        document.getElementById('protect-preview-container').classList.remove('d-none');
    }
});

// Display watermark results
function displayWatermarkResults(results, containerId) {
    const watermarkHtml = `
        <div class="metric-item">
            <span class="metric-label">Watermark Status:</span>
            <span class="metric-value ${results.has_watermark ? 'text-success' : 'text-danger'}">
                ${results.has_watermark ? 'Detected ✅' : 'Not Detected ❌'}
            </span>
        </div>
        ${results.watermark_strength !== undefined ? `
        <div class="metric-item">
            <span class="metric-label">Watermark Strength:</span>
            <span class="metric-value">${(results.watermark_strength * 100).toFixed(1)}%</span>
        </div>
        ` : ''}
        ${results.in_registry ? `
        <div class="metric-item">
            <span class="metric-label">Registry Status:</span>
            <span class="metric-value text-success">Image in Registry ✅</span>
        </div>
        ` : ''}
        ${results.watermark_data && Object.keys(results.watermark_data).length > 0 ? `
        <div class="metric-item">
            <span class="metric-label">Timestamp:</span>
            <span class="metric-value">${results.watermark_data.timestamp ? new Date(results.watermark_data.timestamp).toLocaleString() : 'N/A'}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Method:</span>
            <span class="metric-value">${results.watermark_data.method || 'N/A'}</span>
        </div>
        ` : ''}
    `;
    $(`#${containerId}`).html(watermarkHtml);
}

// Display deepfake results
function displayDeepfakeResults(results, containerId) {
    const predictions = results.model_predictions || {};
    const predictionHtml = Object.entries(predictions)
        .map(([model, score]) => `
            <div class="metric-item">
                <span class="metric-label">${model}:</span>
                <span class="metric-value">${(score * 100).toFixed(1)}%</span>
            </div>
        `).join('');

    const deepfakeHtml = `
        <div class="metric-item">
            <span class="metric-label">Detection Status:</span>
            <span class="metric-value ${results.is_deepfake ? 'text-danger' : 'text-success'}">
                ${results.is_deepfake ? 'Deepfake Detected ⚠️' : 'Authentic ✅'}
            </span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Confidence:</span>
            <span class="metric-value">${(results.confidence * 100).toFixed(1)}%</span>
        </div>
        ${results.whitelisted ? `
        <div class="metric-item">
            <span class="metric-label">Whitelist Status:</span>
            <span class="metric-value text-success">Image Whitelisted ✅</span>
        </div>
        ` : ''}
        ${predictionHtml}
    `;
    $(`#${containerId}`).html(deepfakeHtml);
}

// Handle protect form submission
$('#protect-form').submit(function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    
    // Show loading
    $('#protect-loading').removeClass('d-none');
    $('#protect-results').addClass('d-none');
    
    $.ajax({
        url: '/api/upload',
        method: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            if (response.status === 'success') {
                const results = response.data.analysis;
                $('#protect-results').removeClass('d-none');
                displayWatermarkResults(results.watermark_verification, 'watermark-results');
                
                // If deepfake detection results are available, display them
                if (results.deepfake_detection) {
                    displayDeepfakeResults(results.deepfake_detection, 'deepfake-results');
                }
            } else {
                alert('Error: ' + response.message);
            }
        },
        error: function(xhr, status, error) {
            alert('Error: ' + error);
        },
        complete: function() {
            $('#protect-loading').addClass('d-none');
        }
    });
});

// Handle test form submission
$('#test-form').submit(function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    
    // Show loading
    $('#test-loading').removeClass('d-none');
    $('#test-results').addClass('d-none');
    
    $.ajax({
        url: '/api/test',
        method: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            if (response.status === 'success') {
                const results = response.data.analysis;
                $('#test-results').removeClass('d-none');
                
                // Handle different response formats
                if (results.watermark_analysis) {
                    displayWatermarkResults(results.watermark_analysis, 'watermark-test-results');
                } else if (results.watermark_verification) {
                    displayWatermarkResults(results.watermark_verification, 'watermark-test-results');
                }
                
                if (results.deepfake_detection) {
                    displayDeepfakeResults(results.deepfake_detection, 'deepfake-results');
                }
            } else {
                alert('Error: ' + response.message);
            }
        },
        error: function(xhr, status, error) {
            alert('Error: ' + error);
        },
        complete: function() {
            $('#test-loading').addClass('d-none');
        }
    });
}); 