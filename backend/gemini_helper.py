import requests
import json
import base64

# Hardcoded for your major project verification
API_KEY = ""

def get_available_model():
    """Finds the correct model name authorized for this API key."""
    list_url = f"https://generativelanguage.googleapis.com/v1/models?key={API_KEY}"
    try:
        response = requests.get(list_url)
        models = response.json().get('models', [])
        # Look for any gemini-flash or gemini-pro model
        for m in models:
            if "gemini" in m['name'] and "generateContent" in m['supportedGenerationMethods']:
                # Returns something like 'models/gemini-1.5-flash-latest'
                return m['name']
    except:
        pass
    return "models/gemini-pro-vision" # Fallback

def gemini_image_analysis(image_path):
    print(f"🚀 Discovering available models...")
    model_name = get_available_model()
    print(f"📡 Using discovered model: {model_name}")
    
    # Construct URL using the discovered model name
    url = f"https://generativelanguage.googleapis.com/v1/{model_name}:generateContent?key={API_KEY}"
    
    try:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        payload = {
            "contents": [{
                "parts": [
                    {"text": "Is this image forged or AI generated? Return JSON: {\"is_fake\": true/false, \"reason\": \"explanation\"}"},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_data
                        }
                    }
                ]
            }]
        }

        response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        res_json = response.json()

        if response.status_code != 200:
            raise Exception(f"Error {response.status_code}: {res_json}")

        raw_text = res_json['candidates'][0]['content']['parts'][0]['text']
        clean_text = raw_text.replace('```json', '').replace('```', '').strip()
        data = json.loads(clean_text)
        
        label = "Fake" if data.get("is_fake") else "Authentic"
        print(f"✅ SUCCESS: Analysis complete using {model_name}")
        return label, data.get("reason", "Analysis successful.")

    except Exception as e:
        print(f"❌ FINAL ERROR: {str(e)}")
        return "Unavailable", f"REST Error: {str(e)}"
    
 
