import requests
import json
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

api_key = os.getenv("API_KEY")

def get_category_id():
    url = "https://api.coingecko.com/api/v3/coins/categories/list"
    response = requests.get(url)

    print(f"Category API call status code: {response.status_code}")

    if response.status_code == 200:
        categories = response.json()

        for category in categories:
            if category['name'] == 'Meme':
                print(f"Found category: {category['name']} with ID: {category['category_id']}")
                return category['category_id']
    else:
        print(f"Failed to retrieve category_id {response.text}")
        return None
    

def get_tokens(api_key):
    category_id = get_category_id()

    if not category_id:
        print("Could not find Category ID")
        return None
    print(f"Using Category ID: {category_id}")

    url = "https://api.coingecko.com/api/v3/coins/markets"
    all_tokens = []
    page = 1
    per_page = 250

    while True:
        params = {
            "vs_currency": "usd",
            "category": category_id,
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": page,
            "sparkline": False,
        }
        print(f"Fetching page {page}...")
        response = requests.get(url, params=params)
        print(f"API Response Status Code: {response.status_code}")

        if response.status_code == 200:
            page_data = response.json()

            if not page_data:
                print(f"No more data available. Total records found: {len(all_tokens)}")
                break

            all_tokens.extend(page_data)
            print(f"Retrieved {len(page_data)} tokens from page {page}. Total so far: {len(all_tokens)}")

            if len(page_data) < per_page:
                print(f"Reached end of available data. Total records: {len(all_tokens)}")
                break

            page += 1

            import time
            time.sleep(2)
        elif response.status_code == 429:
            print("Rate limit exceeded waiting 60 seconds before retrying...")
            import time
            time.sleep(60)
            continue

        else:
            print(f"API Request failed on page {page}. Response: {response.text}")
            print("Saving partial data collected so far...")
            break


    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_directory = os.path.join(project_root, "data", "raw")
    file_name = "raw_token_data.json"
    full_path = os.path.join(output_directory, file_name)

    os.makedirs(output_directory, exist_ok=True)


    try:
        with open(full_path, "w") as f:
            json.dump(all_tokens, f, indent=4)  # indent for pretty-printing
        print(f"JSON data successfully saved to: {full_path}")
        print(f"Total records saved: {len(all_tokens)}")
        return all_tokens
    except IOError as e:
        print(f"Error saving file: {e}")
        return None


def to_csv():
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_file = os.path.join(project_root, "data", "raw", "raw_token_data.json")
        output_file = os.path.join(project_root, "data", "raw", "meme_token_data.csv")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        df = pd.read_json(input_file)
        df.to_csv(output_file, index=False)
        print(f"JSON data succesfully converted to CSV: {output_file}")

    except FileNotFoundError:
        print("Error: 'input.json' not found. Please ensure the file exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    tokens = get_tokens(api_key)
    if tokens:
        print(f"Successfully retrieved {len(tokens)} tokens")

        to_csv()

        # Print first token as example
        for i, tokens in enumerate(tokens[:3]):
            print(f"\nMeme Token {i+1}:")
            print(f"  Name: {tokens['name']}")
            print(f"  Symbol: {tokens['symbol']}")
            print(f"  Market Cap Rank: {tokens.get('market_cap_rank', 'N/A')}")
            print(f"  Market Cap: ${tokens.get('market_cap', 'N/A'):,}" if tokens.get('market_cap') else "  Market Cap: N/A")
            print(f"  All Time High: ${tokens.get('ath', 'N/A')}")
            print(f"  Current Price: ${tokens.get('current_price', 'N/A')}")
    else:
        print("Failed to retrieve token data")