
document.getElementById('send').addEventListener('click', async ()=>{
  const text = document.getElementById('msg').value;
  if(!text) return;
  const messages = document.getElementById('messages');
  const userDiv = document.createElement('div'); userDiv.textContent = 'You: ' + text; messages.appendChild(userDiv);
  const res = await fetch('/api/message', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text})});
  const data = await res.json();
  const botDiv = document.createElement('div'); botDiv.textContent = 'Bot: ' + data.reply; messages.appendChild(botDiv);
  document.getElementById('msg').value='';
});
