
// Function to check server environment variables
async function checkServerKeys() {
    try {
        const response = await fetch('/env_keys_check');
        const data = await response.json();
        
        document.getElementById('openaiEnvKeyStatus').innerHTML = 
            data.openai_key_set ? '<small>(Server key available)</small>' : '';
        document.getElementById('stabilityEnvKeyStatus').innerHTML = 
            data.stability_key_set ? '<small>(Server key available)</small>' : '';
        document.getElementById('deepseekEnvKeyStatus').innerHTML = 
            data.deepseek_key_set ? '<small>(Server key available)</small>' : '';
    } catch (e) {
        console.error("Failed to check server API keys:", e);
    }
}

// Function to save API keys to localStorage
function setupApiKeyPersistence() {
    // Restore API keys from localStorage
    const openaiField = document.getElementById('openaiApiKey');
    const stabilityField = document.getElementById('stabilityApiKey');
    const deepseekField = document.getElementById('deepseekApiKey');
    
    if (localStorage.getItem('openaiApiKey')) {
        openaiField.value = localStorage.getItem('openaiApiKey');
    }
    if (localStorage.getItem('stabilityApiKey')) {
        stabilityField.value = localStorage.getItem('stabilityApiKey');
    }
    if (localStorage.getItem('deepseekApiKey')) {
        deepseekField.value = localStorage.getItem('deepseekApiKey');
    }
    
    // Save API keys to localStorage when they change
    openaiField.addEventListener('input', function() {
        localStorage.setItem('openaiApiKey', this.value);
    });
    stabilityField.addEventListener('input', function() {
        localStorage.setItem('stabilityApiKey', this.value);
    });
    deepseekField.addEventListener('input', function() {
        localStorage.setItem('deepseekApiKey', this.value);
    });
    
    // Check server environment variables
    checkServerKeys();
}

// Call this function on page load
document.addEventListener('DOMContentLoaded', setupApiKeyPersistence);
