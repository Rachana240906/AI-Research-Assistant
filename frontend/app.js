// Expose query filler utility globally
window.populateQuery = function(promptText) {
    const queryInput = document.getElementById('queryInput');
    if(queryInput) {
        queryInput.value = promptText;
        queryInput.focus();
    }
};

// Expose instant text copying module globally
window.copyWorkspaceOutput = function() {
    const markdownZone = document.getElementById('resultText');
    if (!markdownZone) return;
    
    navigator.clipboard.writeText(markdownZone.innerText)
        .then(() => alert('Research brief successfully copied to clipboard!'))
        .catch(() => alert('Failed to access clipboard operations.'));
};

// Lightweight, bulletproof Markdown formatting utility
function parseMarkdownToHTML(text) {
    if (!text) return "";
    let html = text;
    
    // Convert bold weights (**text**)
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Convert tertiary level header markers (### Title)
    html = html.replace(/^### (.*?)$/gm, '<h3>$1</h3>');
    
    // Convert item bullets (- Bullet)
    html = html.replace(/^- (.*?)$/gm, '<li>$1</li>');
    
    // Wrap scattered bullet lists inside proper outer UL wrappers
    html = html.replace(/(<li>.*?<\/li>)+/g, '<ul>$&</ul>');
    
    // Process remaining spacing breaks safely
    html = html.split('\n\n').map(p => {
        if (p.trim().startsWith('<h') || p.trim().startsWith('<ul')) return p;
        return `<p>${p.replace(/\n/g, '<br>')}</p>`;
    }).join('');
    
    return html;
}

document.getElementById('searchBtn').addEventListener('click', async () => {
    const queryInput = document.getElementById('queryInput');
    const query = queryInput.value.trim();

    if (!query) {
        alert('Please enter a query or research topic first!');
        return;
    }

    const emptyStateWrapper = document.getElementById('emptyStateWrapper');
    const suggestionsWrapper = document.getElementById('suggestionsWrapper');
    const loadingState = document.getElementById('loadingState');
    const errorState = document.getElementById('errorState');
    const resultCard = document.getElementById('resultCard');
    const resultText = document.getElementById('resultText');
    const workspaceTitle = document.getElementById('workspaceTitle');
    const executiveInsightText = document.getElementById('executiveInsightText');

    // Minimize landing page elements
    emptyStateWrapper.classList.add('hidden');
    suggestionsWrapper.classList.add('hidden');
    
    // Reset view states
    loadingState.classList.remove('hidden');
    errorState.classList.add('hidden');
    resultCard.classList.add('hidden');

    try {
        const response = await fetch('http://127.0.0.1:8000/api/research', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Failed to complete research execution pass.');
        }

        // Configure workspace dynamic heading details based on current request
        if(workspaceTitle) workspaceTitle.textContent = `Research Brief: "${query}"`;
        
        // Formulate a short single paragraph insight highlight
        const rawOutput = data.output;
        if(executiveInsightText) {
            const firstSentenceEnd = rawOutput.indexOf('.') !== -1 ? rawOutput.indexOf('.') + 1 : 120;
            executiveInsightText.textContent = rawOutput.substring(0, firstSentenceEnd) || rawOutput;
        }

        // Run string transformation module and inject into render layer
        if(resultText) {
            resultText.innerHTML = parseMarkdownToHTML(rawOutput);
        }
        
        resultCard.classList.remove('hidden');
    } catch (error) {
        document.getElementById('errorMessage').textContent = error.message;
        errorState.classList.remove('hidden');
        emptyStateWrapper.classList.remove('hidden');
        suggestionsWrapper.classList.remove('hidden');
    } finally {
        loadingState.classList.add('hidden');
    }
});
// Expose instant text copying module globally with secure fallback
window.copyWorkspaceOutput = function() {
    const markdownZone = document.getElementById('resultText');
    if (!markdownZone) return;
    
    const textToCopy = markdownZone.innerText;

    // Mode A: Try the modern Async Clipboard API first
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(textToCopy)
            .then(() => alert('Research brief successfully copied to clipboard!'))
            .catch(() => runLegacyCopyFallback(textToCopy));
    } else {
        // Mode B: Instantly engage the fail-safe fallback for non-HTTPS or strict local environments
        runLegacyCopyFallback(textToCopy);
    }
};

// Bulletproof hidden textarea fallback handler
function runLegacyCopyFallback(text) {
    try {
        const textArea = document.createElement("textarea");
        textArea.value = text;
        
        // Lock layout settings so it doesn't cause any visual layout jump or scrolling artifact
        textArea.style.top = "0";
        textArea.style.left = "0";
        textArea.style.position = "fixed";
        textArea.style.opacity = "0";
        
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        const successful = document.execCommand('copy');
        document.body.removeChild(textArea);
        
        if (successful) {
            alert('Research brief successfully copied to clipboard! (via legacy mode)');
        } else {
            alert('Unable to copy. Please manually highlight and copy the selection.');
        }
    } catch (err) {
        alert('An unexpected error stopped the clipboard process.');
    }
}