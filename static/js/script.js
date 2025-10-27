document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const submitBtn = document.getElementById('submitBtn');
    const spinner = submitBtn.querySelector('.btn-spinner');
    const predictionResult = document.getElementById('predictionResult');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');

    // Show loading state
    spinner.style.display = 'inline-block';
    submitBtn.disabled = true;
    predictionResult.style.display = 'none';
    errorMessage.style.display = 'none';

    try {
        const formData = new FormData(e.target);
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: JSON.stringify(Object.fromEntries(formData)),
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response.ok) throw new Error('Prediction failed');

        const data = await response.json();
        document.getElementById('usdPrice').textContent = `$${data.usdPrice.toFixed(2)}`;
        document.getElementById('rwfPrice').textContent = `${data.rwfPrice.toFixed(0)} RWF`;
        predictionResult.style.display = 'block';

        // Update current selection
        document.getElementById('currentCommodity').textContent = formData.get('commodity');
        document.getElementById('currentMarket').textContent = formData.get('market');
        document.getElementById('currentPriceType').textContent = formData.get('pricetype');
        document.getElementById('currentUnit').textContent = formData.get('unit');
    } catch (error) {
        errorText.textContent = error.message || 'An error occurred';
        errorMessage.style.display = 'block';
    } finally {
        spinner.style.display = 'none';
        submitBtn.disabled = false;
    }
});