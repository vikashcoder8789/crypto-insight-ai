function sendQuery() {
  const coin = document.getElementById('coin').value;
  const query = document.getElementById('query').value;
  const resultDiv = document.getElementById('result');

  const backendURL = `${window.location.origin}/query`;

  fetch(backendURL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ coin, query })
  })
      .then(res => {
        if (!res.ok) {
          return res.json().then(err => {
            throw new Error(err.error || 'Something went wrong');
          });
        }
        return res.json();
      })
      .then(data => {
        const { coin, query, sentiment, top_answers } = data;
    
        const formattedAnswers = top_answers
          .map(ans => `<li>${ans.replace(/^[-\d\.\s]+/, '').replace(/\n/g, ' ').trim()}</li>`)
          .join('');
    
        resultDiv.innerHTML = `
          <p><strong>Coin:</strong> ${capitalize(coin)}</p>
          <p><strong>Question:</strong> ${query}</p>
          <p><strong>Sentiment:</strong> ${sentiment}</p>
          <p><strong>Top Answers:</strong></p>
          <ol>${formattedAnswers}</ol>
        `;
      })
      .catch(err => {
        resultDiv.textContent = err.message;
        console.log(err)
      });
}

const capitalize = str => str.charAt(0).toUpperCase() + str.slice(1);
