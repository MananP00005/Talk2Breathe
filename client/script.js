const sendButton = document.getElementById('send');
const questionInput = document.getElementById('question');
const responseDiv = document.getElementById('response');

sendButton.addEventListener('click', async () => {
    const question = questionInput.value.trim();
    if (!question) return;

    responseDiv.innerHTML = '<p>Breathe is thinking...</p>';

    try {
        const response = await fetch(`http://localhost:8000/chat?question=${encodeURIComponent(question)}`, {
            method: 'POST',
        });

        const data = await response.json();
        responseDiv.innerHTML = `<p><strong>Breathe:</strong> ${data.response}</p>`;

    } catch (error) {
        responseDiv.innerHTML = '<p>Sorry, I could not connect. Please try again.</p>';
    }
});