import base64
import datetime
import io
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import webdataset as wds
from openai import OpenAI
from tqdm import tqdm

from loader.wave import preprocess_sample
from utils.dataset_utils import get_tar_file_list
from utils.utils import get_args, set_seed


class RateLimiter:
    def __init__(self, max_requests_per_minute):
        self.max_requests = max_requests_per_minute
        self.requests = queue.Queue()
        self.lock = threading.Lock()
        
    def acquire(self):
        with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            while not self.requests.empty():
                if now - self.requests.queue[0] > 60:
                    self.requests.get()
                else:
                    break
            
            if self.requests.qsize() >= self.max_requests:
                # Wait until the oldest request is 1 minute old
                wait_time = 60 - (now - self.requests.queue[0])
                if wait_time > 0:
                    time.sleep(wait_time)
            
            self.requests.put(now)

def check_balance(client):
    try:
        # Get organization details which includes usage
        org_id = client.organization
        headers = {
            "Authorization": f"Bearer {client.api_key}",
            "OpenAI-Organization": org_id
        }
        
        # Get current date
        current_date = datetime.datetime.now()
        start_of_month = current_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Format dates for API
        start_date = start_of_month.strftime("%Y-%m-%d")
        end_date = current_date.strftime("%Y-%m-%d")
        
        # Get usage data
        usage_url = f"https://api.openai.com/v1/usage?date={end_date}"
        response = requests.get(usage_url, headers=headers)
        usage_data = response.json()
        
        # Get billing data
        billing_url = "https://api.openai.com/v1/dashboard/billing/usage"
        params = {
            "start_date": start_date,
            "end_date": end_date
        }
        billing_response = requests.get(billing_url, headers=headers, params=params)
        billing_data = billing_response.json()
        
        print("\nCurrent API Usage:")
        print(f"Total tokens used today: {usage_data.get('total_tokens', 'N/A')}")
        print(f"Total cost today: ${usage_data.get('total_cost', 0):.4f}")
        
        print(f"\nMonthly Usage (since {start_date}):")
        print(f"Monthly tokens used: {billing_data.get('total_usage', 'N/A')}")
        print(f"Monthly cost: ${billing_data.get('total_cost', 0):.4f}")
        
        return usage_data.get('total_cost', 0), billing_data.get('total_cost', 0)
        
    except Exception as e:
        print(f"Error checking balance: {e}")
        return None, None

def calculate_cost(prompt_tokens, completion_tokens, model="gpt-4-vision-preview"):
    # Current pricing as of March 2024
    if model == "gpt-4o-mini":
        input_cost_per_1k = 0.00015 # $0.00015 per 1K tokens for input
        output_cost_per_1k = 0.0006 # $0.0006 per 1K tokens for output
    elif model == "gpt-4o":
        input_cost_per_1k = 0.0025 # $0.0025 per 1K tokens for input
        output_cost_per_1k = 0.01  # $0.01 per 1K tokens for output
    else:
        raise ValueError(f"Pricing not configured for model: {model}")
    
    input_cost = (prompt_tokens / 1000) * input_cost_per_1k
    output_cost = (completion_tokens / 1000) * output_cost_per_1k
    total_cost = input_cost + output_cost
    return total_cost

def process_image(args):
    client, rate_limiter, sample, prompt_template, i = args
    try:
        image = sample['pil_img']
        label = sample['img_label']
        img_name = sample['img_name']
        prompt = prompt_template + label
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Rate limit the request
        rate_limiter.acquire()
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates scene graphs based on image content."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
        )
        
        # Get token usage from response
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        
        # Calculate cost for this request
        cost = calculate_cost(prompt_tokens, completion_tokens, model="gpt-4o")
        
        scene_graph = response.choices[0].message.content
        print(scene_graph)
        return {
            'index': i,
            'img_name': img_name,
            'img_label': label,
            'scene_graph': scene_graph,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'cost': cost
        }
        
    except Exception as e:
        print(f"Error processing image {i}: {e}")
        return None

def main(args):
    api_key = "YOUR_API_KEY"
    client = OpenAI(api_key=api_key)

    # Check balance before starting
    total_cost, monthly_cost = check_balance(client)
    if total_cost is None:
        print("Failed to check balance. Proceeding with caution.")
    else:
        print("\nProceeding with processing...")
    
    prompt_path = 'data/bold5000_scene_graph_prompts.txt'
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()
    
    tar_file_list = get_tar_file_list(
        tar_dir=args.wave_dir,
        subject_list=['CSI1'],
        split_list=['train', 'test']
    )
    wave_dataset = wds.WebDataset(tar_file_list).decode("pil").map(preprocess_sample)
    
    # Initialize rate limiter (adjust max_requests_per_minute based on your API limits)
    rate_limiter = RateLimiter(max_requests_per_minute=60)  # Adjust this value based on your API limits
    # Convert dataset to list for easier processing
    samples = list(wave_dataset)
    # Initialize results dictionary
    results = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0
    
    # Process images with multiple threads
    num_threads = 5  # Adjust based on your needs and API limits
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Prepare arguments for each image
        args_list = [(client, rate_limiter, sample, prompt_template, i) 
                    for i, sample in enumerate(samples)]
        # Process images with progress bar
        futures = list(tqdm(executor.map(process_image, args_list), 
                          total=len(samples),
                          desc="Processing images"))
        
        # Collect results
        for result in futures:
            if result is not None:
                i = result['index']
                results[i] = {
                    'img_name': result['img_name'],
                    'img_label': result['img_label'],
                    'scene_graph': result['scene_graph']
                }
                total_prompt_tokens += result['prompt_tokens']
                total_completion_tokens += result['completion_tokens']
                total_cost += result['cost']
                
                # Print progress for this request
                print(f"\nRequest {i+1} Token Usage:")
                print(f"Prompt tokens: {result['prompt_tokens']}")
                print(f"Completion tokens: {result['completion_tokens']}")
                print(f"Cost for this request: ${result['cost']:.4f}")
    
    # Print total usage
    print("\nTotal Usage Summary:")
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total completion tokens: {total_completion_tokens}")
    print(f"Total cost: ${total_cost:.4f}")
    
    # Save results to CSV
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('data/bold5000_scene_graph.csv', index=False)

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    main(args)