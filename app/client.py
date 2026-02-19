import requests

response = requests.post(
    'http://localhost:5000/predict',
    json={'text': 'This movie was amazing!'}
)

print("Status Code:", response.status_code)
print("Response:", response.json())
