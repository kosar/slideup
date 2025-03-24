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
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            updateFileInfo();
        }
    }
    
    // Handle file selection via input
    fileInput.addEventListener('change', updateFileInfo);
    
    function updateFileInfo() {
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            const fileSizeInMB = (file.size / 1024 / 1024).toFixed(2);
            
            fileName.textContent = `${file.name} (${fileSizeInMB} MB)`;
            fileInfo.classList.remove('d-none');
        } else {
            fileInfo.classList.add('d-none');
        }
    }
    
    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        // Validate file selection
        if (fileInput.files.length === 0) {
            e.preventDefault();
            alert('Please select a file to upload');
            return;
        }
        
        // Show loading state
        btnText.textContent = 'Processing...';
        spinner.classList.remove('d-none');
        submitBtn.disabled = true;
    });
    
    // Make the entire drop zone clickable
    dropZone.addEventListener('click', function() {
        fileInput.click();
    });
}); 