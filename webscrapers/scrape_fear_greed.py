# import json 
# from datetime import datetime
# import requests
# import os

# def scrape_fear_greed_index(api_url):
#     json_path = os.path.join("json_folder", "feargreed.json")
#     existing_values = []
    
#     # Load existing data if file exists
#     if os.path.exists(json_path):
#         with open(json_path, "r", encoding="utf-8") as file:
#             print("Loaded .json file...")
#             existing_values = json.load(file)
            
#     print("Fetching Fear & Greed index from API...")
#     try:
#         response = requests.get(api_url, timeout=10)
#         response.raise_for_status()  # Raises HTTPError for bad responses

#         data = response.json()["data"][0]
#         value = data["value"]
#         date = datetime.now().strftime("%Y-%m-%d")  # Proper date format

#         # Don't duplicate today's entry
#         if existing_values and existing_values[0]["date"] == date:
#             print("Entry for today already exists.")
#             return value

#         # Save to JSON
#         new_entry = {
#             "date": date,
#             "fear_greed_index": value
#         }

#         with open(json_path, "w", encoding="utf-8") as json_file:
#             json.dump([new_entry] + existing_values, json_file, indent=4)

#         print("New entry saved.")
#         return value

#     except Exception as e:
#         print(f"Error fetching Fear & Greed index: {e}")
#         return None

# # Run the function
# scrape_fear_greed_index("https://api.alternative.me/fng/")


