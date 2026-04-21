import requests
import json
import os

r = requests.post('http://localhost:5050/search', json={
    'query': 'Who invented the telephone',
    'model': 'llama-3.3-70b-versatile',
    'api_key': os.environ.get('GROQ_API_KEY', 'your_groq_api_key_here')
})
d = r.json()
print('Status:', r.status_code)
print('Num results:', len(d.get('results', [])))
print('Avg hallucination:', d.get('avg_hallucination'))
for x in d.get('results', []):
    print(f"  [{x['verdict']}] {x['score']}% - {x['title'][:60]}")
