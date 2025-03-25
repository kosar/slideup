document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const uploadForm = document.getElementById('uploadForm');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const spinner = document.getElementById('spinner');
    
    console.log('Upload handler initialized');
    
    // Handle drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropZone.classList.add('dragover');
    }
    
    function unhighlight() {
        dropZone.classList.remove('dragover');
    }
    
    // Handle file drop
    dropZone.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        console.log('File dropped');
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            console.log('Dropped file:', files[0].name, 'Size:', (files[0].size / (1024 * 1024)).toFixed(2), 'MB');
            fileInput.files = files;
            updateFileInfo();
        }
    }
    
    // Handle file selection via input
    fileInput.addEventListener('change', function(e) {
        console.log('File input change event triggered');
        updateFileInfo();
    });
    
    function updateFileInfo() {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            const fileSizeInMB = (file.size / 1024 / 1024).toFixed(2);
            
            console.log('Selected file:', file.name, 'Size:', fileSizeInMB, 'MB');
            fileName.textContent = `Selected: ${file.name} (${fileSizeInMB} MB)`;
            fileInfo.classList.remove('d-none');
            fileInfo.classList.remove('alert-success');
            fileInfo.classList.add('alert-info');
        } else {
            console.log('No file selected');
            fileInfo.classList.add('d-none');
        }
    }
    
    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault(); // Prevent default form submission
        
        // Validate file selection
        if (fileInput.files.length === 0) {
            console.log('No file selected for upload');
            alert('Please select a file to upload');
            return;
        }
        
        console.log('Starting file upload');
        
        // Show loading state
        btnText.textContent = 'Processing...';
        spinner.classList.remove('d-none');
        submitBtn.disabled = true;
        
        // Create FormData and append file
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        // Add any other form data
        if (document.getElementById('limit_to_one_minute').checked) {
            formData.append('limit_to_one_minute', 'true');
        }
        
        // Send the file using fetch
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log('Upload response status:', response.status);
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.text();
        })
        .then(data => {
            console.log('Upload successful:', data);
            window.location.href = data; // Redirect to the response URL
        })
        .catch(error => {
            console.error('Upload error:', error);
            alert('Error uploading file: ' + error.message);
            // Reset button state
            btnText.textContent = 'Convert to Video';
            spinner.classList.add('d-none');
            submitBtn.disabled = false;
        });
    });
    
    // Make the drop zone clickable
    dropZone.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        fileInput.click();
    });
    
    // Prevent the click event from bubbling up from the file input
    fileInput.addEventListener('click', function(e) {
        e.stopPropagation();
    });
}); 