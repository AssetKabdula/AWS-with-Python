# %%
# Importing required libraries and set up our environment
import os
import random
import pprint
import string
import time
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import boto3

# Initialize pretty printer for better output formatting
pp = pprint.PrettyPrinter(indent=2)


print("📚 Setting up the environment...")

# Creating S3 client using default credentials from AWS CLI
# boto3 will automatically use credentials from ~/.aws/credentials
s3 = boto3.client(
    "s3",
    region_name="eu-west-1",  # Ireland region
)
# %%
# Let's first check what buckets already exist in your AWS account
# This helps us understand what resources we're starting with
print("📋 Listing all S3 buckets in your account...")
response = s3.list_buckets()

print("\n📦 Raw response from AWS:")
pp.pprint(response)

print("\n📦 Your current S3 buckets:")
if response["Buckets"]:
    for bucket in response["Buckets"]:
        print(f"- {bucket['Name']}")
else:
    print("No buckets found in your account")

print(f"\n✅ Successfully retrieved {len(response['Buckets'])} buckets")
# %%
# Uploading our aufiofile to S3
# The name of our bucket
bucket_name = "ceu-asset-de2"

print(f"⬆️  Uploading file to bucket: {bucket_name}")

try:
    s3.upload_file("/workspaces/Data-Engineering-2-Cloud-Computing/Elon_1_year_ago.mp3", bucket_name, "Elon_1_year_ago.mp3")
    print("✅ Upload successful!")

    # Verify the upload by listing objects in the bucket
    objects = s3.list_objects_v2(Bucket=bucket_name)
    print("\n📦 Current bucket contents:")
    for obj in objects.get("Contents", []):
        print(f"- {obj['Key']} ({obj['Size']} bytes)")
except Exception as e:
    print(f"❌ Error uploading file: {str(e)}")
# %%
try:
    s3.upload_file("/workspaces/Data-Engineering-2-Cloud-Computing/Elon_2_years_ago.mp3", bucket_name, "Elon_2_years_ago.mp3")
    print("✅ Upload successful!")
    print(f"📍 File location: s3://{bucket_name}/Elon_2_years_ago.mp3")
    print(f"ℹ️ Note: The https URL https://{bucket_name}.s3.eu-west-1.amazonaws.com/Elon_2_years_ago.mp3")
    print("   won't work directly because S3 objects are private by default!")
    print("   We'll generate a pre-signed URL later to access this file via HTTPS.")

    # Verify the upload by listing objects in the bucket
    objects = s3.list_objects_v2(Bucket=bucket_name)
    print("\n📦 Current bucket contents:")
    for obj in objects.get("Contents", []):
        print(f"- {obj['Key']} ({obj['Size']} bytes)")
except Exception as e:
    print(f"❌ Error uploading file: {str(e)}")
# %%
try:
    s3.upload_file("/workspaces/Data-Engineering-2-Cloud-Computing/Elon_5_years_ago_1.mp3", bucket_name, "Elon_5_years_ago_1.mp3")
    print("✅ Upload successful!")
    print(f"📍 File location: s3://{bucket_name}/Elon_5_years_ago_1.mp3")
    print(f"ℹ️ Note: The https URL https://{bucket_name}.s3.eu-west-1.amazonaws.com/Elon_5_years_ago_1.mp3")
    print("   won't work directly because S3 objects are private by default!")
    print("   We'll generate a pre-signed URL later to access this file via HTTPS.")

    # Verify the upload by listing objects in the bucket
    objects = s3.list_objects_v2(Bucket=bucket_name)
    print("\n📦 Current bucket contents:")
    for obj in objects.get("Contents", []):
        print(f"- {obj['Key']} ({obj['Size']} bytes)")
except Exception as e:
    print(f"❌ Error uploading file: {str(e)}")
# %%
try:
    s3.upload_file("/workspaces/Data-Engineering-2-Cloud-Computing/Elon_5_years_ago_2.mp3", bucket_name, "Elon_5_years_ago_2.mp3")
    print("✅ Upload successful!")
    print(f"📍 File location: s3://{bucket_name}/Elon_5_years_ago_2.mp3")
    print(f"ℹ️ Note: The https URL https://{bucket_name}.s3.eu-west-1.amazonaws.com/Elon_5_years_ago_2.mp3")
    print("   won't work directly because S3 objects are private by default!")
    print("   We'll generate a pre-signed URL later to access this file via HTTPS.")

    # Verify the upload by listing objects in the bucket
    objects = s3.list_objects_v2(Bucket=bucket_name)
    print("\n📦 Current bucket contents:")
    for obj in objects.get("Contents", []):
        print(f"- {obj['Key']} ({obj['Size']} bytes)")
except Exception as e:
    print(f"❌ Error uploading file: {str(e)}")
# %%
# Transcribe audio file with speaker diarization
print("🔍 Transcribing audio file with speaker diarization...")

S3_BUCKET = "ceu-asset-de2"
S3_KEY = "Elon_1_year_ago.mp3"

print("\n📂 Audio file location:")
print(f"- Bucket: {S3_BUCKET}")
print(f"- Key: {S3_KEY}")

# Initialize AWS Transcribe client
transcribe = boto3.client('transcribe')
# %%
try:
    # Define transcription job parameters
    job_name = "ElonLexInterview_1YearAgo"
    media_uri = f"s3://{S3_BUCKET}/{S3_KEY}"
    language_code = "en-US"  # Specify the language of the audio file

    # Start transcription job with speaker diarization enabled
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": media_uri},
        MediaFormat="mp3",  # Specify audio format
        LanguageCode=language_code,
        Settings={
            "ShowSpeakerLabels": True,  # Enable speaker diarization
            "MaxSpeakerLabels": 2       # Maximum number of speakers to detect
        }
    )

    print("\n🚀 Transcription job started:")
    print(f"- Job Name: {job_name}")
    print(f"- Media URI: {media_uri}")

    # Wait for the transcription job to complete
    print("\n⏳ Waiting for transcription to complete...")
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        job_status = status['TranscriptionJob']['TranscriptionJobStatus']

        if job_status in ["COMPLETED", "FAILED"]:
            break

        print(f"🕒 Current status: {job_status} (checking again in 10 seconds...)")
        time.sleep(10)

except Exception as e:
    print(f"❌ Error transcribing audio file: {str(e)}")
# %%
# URL from the Transcribe response
transcription_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']

# Fetch transcription JSON
try:
    print("\n📥 Downloading transcription...")
    response = requests.get(transcription_url)
    response.raise_for_status()  # Ensure no HTTP errors
    transcription_json = response.json()
    print("\n✅ Transcription downloaded successfully!")
except Exception as e:
    print(f"❌ Error fetching transcription: {str(e)}")
    transcription_json = None
# %%
# Save transcription JSON to S3 bucket
try:
    # Save transcription JSON to your bucket
    s3.put_object(
        Bucket=bucket_name,  
        Key="transcriptions/ElonLexInterview_1YearAgo.json",
        Body=json.dumps(transcription_json),
        ContentType="application/json"
    )
    print("\n✅ Transcription saved to S3 bucket:")
    print(f"- s3://ceu-asset-de2/transcriptions/ElonLexInterview_1YearAgo.json")

except Exception as e:
    print(f"❌ Error saving transcription to S3: {str(e)}")
# %%
S3_KEY = "Elon_2_years_ago.mp3"
# %%
try:
    # Define transcription job parameters
    job_name = "ElonLexInterview_2YearsAgo"
    media_uri = f"s3://{S3_BUCKET}/{S3_KEY}"
    language_code = "en-US"  # Specify the language of the audio file

    # Start transcription job with speaker diarization enabled
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": media_uri},
        MediaFormat="mp3",  # Specify audio format
        LanguageCode=language_code,
        Settings={
            "ShowSpeakerLabels": True,  # Enable speaker diarization
            "MaxSpeakerLabels": 2       # Maximum number of speakers to detect
        }
    )

    print("\n🚀 Transcription job started:")
    print(f"- Job Name: {job_name}")
    print(f"- Media URI: {media_uri}")

    # Wait for the transcription job to complete
    print("\n⏳ Waiting for transcription to complete...")
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        job_status = status['TranscriptionJob']['TranscriptionJobStatus']

        if job_status in ["COMPLETED", "FAILED"]:
            break

        print(f"🕒 Current status: {job_status} (checking again in 10 seconds...)")
        time.sleep(10)

    if job_status == "COMPLETED":
        print("\n✅ Transcription complete!")
        transcription_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
    # Fetch transcription JSON
        try:
            print("\n📥 Downloading transcription...")
            response = requests.get(transcription_url)
            response.raise_for_status()  # Ensure no HTTP errors
            transcription_json = response.json()
            print("\n✅ Transcription downloaded successfully!")
        except Exception as e:
            print(f"❌ Error fetching transcription: {str(e)}")
            transcription_json = None
    # Save transcription JSON to S3 bucket
        try:
            # Save transcription JSON to your bucket
            s3.put_object(
             Bucket=bucket_name,  
             Key="transcriptions/ElonLexInterview_2YearsAgo.json",
             Body=json.dumps(transcription_json),
             ContentType="application/json"
            )
            print("\n✅ Transcription saved to S3 bucket:")
            print(f"- s3://ceu-asset-de2/transcriptions/ElonLexInterview_2YearsAgo.json")

        except Exception as e:
            print(f"❌ Error saving transcription to S3: {str(e)}")

except Exception as e:
    print(f"❌ Error transcribing audio file: {str(e)}")
# %%
S3_KEY = "Elon_5_years_ago_1.mp3"
# %%
try:
    # Define transcription job parameters
    job_name = "ElonLexInterview_5YearsAgo1"
    media_uri = f"s3://{S3_BUCKET}/{S3_KEY}"
    language_code = "en-US"  # Specify the language of the audio file

    # Start transcription job with speaker diarization enabled
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": media_uri},
        MediaFormat="mp3",  # Specify audio format
        LanguageCode=language_code,
        Settings={
            "ShowSpeakerLabels": True,  # Enable speaker diarization
            "MaxSpeakerLabels": 2       # Maximum number of speakers to detect
        }
    )

    print("\n🚀 Transcription job started:")
    print(f"- Job Name: {job_name}")
    print(f"- Media URI: {media_uri}")

    # Wait for the transcription job to complete
    print("\n⏳ Waiting for transcription to complete...")
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        job_status = status['TranscriptionJob']['TranscriptionJobStatus']

        if job_status in ["COMPLETED", "FAILED"]:
            break

        print(f"🕒 Current status: {job_status} (checking again in 10 seconds...)")
        time.sleep(10)

    if job_status == "COMPLETED":
        print("\n✅ Transcription complete!")
        transcription_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
    # Fetch transcription JSON
        try:
            print("\n📥 Downloading transcription...")
            response = requests.get(transcription_url)
            response.raise_for_status()  # Ensure no HTTP errors
            transcription_json = response.json()
            print("\n✅ Transcription downloaded successfully!")
        except Exception as e:
            print(f"❌ Error fetching transcription: {str(e)}")
            transcription_json = None
    # Save transcription JSON to S3 bucket
        try:
            # Save transcription JSON to your bucket
            s3.put_object(
             Bucket=bucket_name,  
             Key="transcriptions/ElonLexInterview_5YearsAgo1.json",
             Body=json.dumps(transcription_json),
             ContentType="application/json"
            )
            print("\n✅ Transcription saved to S3 bucket:")
            print(f"- s3://ceu-asset-de2/transcriptions/ElonLexInterview_5YearsAgo1.json")

        except Exception as e:
            print(f"❌ Error saving transcription to S3: {str(e)}")

except Exception as e:
    print(f"❌ Error transcribing audio file: {str(e)}")
# %%
S3_KEY = "Elon_5_years_ago_2.mp3"
# %%
try:
    # Define transcription job parameters
    job_name = "ElonLexInterview_5YearsAgo2"
    media_uri = f"s3://{S3_BUCKET}/{S3_KEY}"
    language_code = "en-US"  # Specify the language of the audio file

    # Start transcription job with speaker diarization enabled
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": media_uri},
        MediaFormat="mp3",  # Specify audio format
        LanguageCode=language_code,
        Settings={
            "ShowSpeakerLabels": True,  # Enable speaker diarization
            "MaxSpeakerLabels": 2       # Maximum number of speakers to detect
        }
    )

    print("\n🚀 Transcription job started:")
    print(f"- Job Name: {job_name}")
    print(f"- Media URI: {media_uri}")

    # Wait for the transcription job to complete
    print("\n⏳ Waiting for transcription to complete...")
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        job_status = status['TranscriptionJob']['TranscriptionJobStatus']

        if job_status in ["COMPLETED", "FAILED"]:
            break

        print(f"🕒 Current status: {job_status} (checking again in 10 seconds...)")
        time.sleep(10)

    if job_status == "COMPLETED":
        print("\n✅ Transcription complete!")
        transcription_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
    # Fetch transcription JSON
        try:
            print("\n📥 Downloading transcription...")
            response = requests.get(transcription_url)
            response.raise_for_status()  # Ensure no HTTP errors
            transcription_json = response.json()
            print("\n✅ Transcription downloaded successfully!")
        except Exception as e:
            print(f"❌ Error fetching transcription: {str(e)}")
            transcription_json = None
    # Save transcription JSON to S3 bucket
        try:
            # Save transcription JSON to your bucket
            s3.put_object(
             Bucket=bucket_name,  
             Key="transcriptions/ElonLexInterview_5YearsAgo2.json",
             Body=json.dumps(transcription_json),
             ContentType="application/json"
            )
            print("\n✅ Transcription saved to S3 bucket:")
            print(f"- s3://ceu-asset-de2/transcriptions/ElonLexInterview_5YearsAgo2.json")

        except Exception as e:
            print(f"❌ Error saving transcription to S3: {str(e)}")

except Exception as e:
    print(f"❌ Error transcribing audio file: {str(e)}")
# %% 
# Create Comprehend client
comprehend = boto3.client(service_name="comprehend", region_name="eu-west-1")
# %%
# Load the JSON file
json_file_path = "/workspaces/Data-Engineering-2-Cloud-Computing/ElonLexInterview_1YearAgo.json"

# Load the JSON data
with open(json_file_path) as f:
    data = json.load(f)
# %%
try:
    # Load the JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Extract the full transcript text
    full_transcript = " ".join([item['transcript'] for item in data['results']['transcripts']])

    # Determine chunk size as 5% of total text size
    chunk_size = int(len(full_transcript) * 0.05)

    # Split the text into manageable chunks
    chunks = [full_transcript[i:i+chunk_size] for i in range(0, len(full_transcript), chunk_size)]

    # Analyze sentiment for each chunk and collect sentiment scores
    sentiment_scores_1 = []
    for chunk in chunks:
        try:
            response = comprehend.detect_sentiment(Text=chunk, LanguageCode='en')
            sentiment_scores_1.append(response['SentimentScore'])
        except Exception as e:
            print(f"Error analyzing sentiment for chunk: {str(e)}")

    # Compute the average sentiment scores
    if sentiment_scores_1:
        avg_sentiment_score = {
            'Positive': np.mean([score['Positive'] for score in sentiment_scores_1]),
            'Negative': np.mean([score['Negative'] for score in sentiment_scores_1]),
            'Neutral': np.mean([score['Neutral'] for score in sentiment_scores_1]),
            'Mixed': np.mean([score['Mixed'] for score in sentiment_scores_1]),
        }
        print("\nAverage Sentiment Score across all chunks:")
        print(avg_sentiment_score)
    else:
        print("No sentiment scores to compute the average.")

except FileNotFoundError:
    print(f"Error: The file at path {json_file_path} could not be found.")
except json.JSONDecodeError:
    print(f"Error: Failed to decode JSON from the file at path {json_file_path}.")
except Exception as e:
    print(f"An error occurred: {str(e)}")


# %%
# Load the JSON file
json_file_path = "/workspaces/Data-Engineering-2-Cloud-Computing/ElonLexInterview_2YearsAgo.json"

# Load the JSON data
with open(json_file_path) as f:
    data = json.load(f)
# %%
try:
    # Load the JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Extract the full transcript text
    full_transcript = " ".join([item['transcript'] for item in data['results']['transcripts']])

    # Determine chunk size as 5% of total text size
    chunk_size = int(len(full_transcript) * 0.05)
    chunks = [full_transcript[i:i+chunk_size] for i in range(0, len(full_transcript), chunk_size)]

    # Analyze sentiment for each chunk and collect sentiment scores
    sentiment_scores_2 = []
    for chunk in chunks:
        try:
            response = comprehend.detect_sentiment(Text=chunk, LanguageCode='en')
            sentiment_scores_2.append(response['SentimentScore'])
        except Exception as e:
            print(f"Error analyzing sentiment for chunk: {str(e)}")

    # Compute the average sentiment scores
    if sentiment_scores_2:
        avg_sentiment_score_1 = {
            'Positive': np.mean([score['Positive'] for score in sentiment_scores_2]),
            'Negative': np.mean([score['Negative'] for score in sentiment_scores_2]),
            'Neutral': np.mean([score['Neutral'] for score in sentiment_scores_2]),
            'Mixed': np.mean([score['Mixed'] for score in sentiment_scores_2]),
        }
        print("\nAverage Sentiment Score across all chunks:")
        print(avg_sentiment_score_1)
    else:
        print("No sentiment scores to compute the average.")

except FileNotFoundError:
    print(f"Error: The file at path {json_file_path} could not be found.")
except json.JSONDecodeError:
    print(f"Error: Failed to decode JSON from the file at path {json_file_path}.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
# %%
# Load the JSON file
json_file_path = "/workspaces/Data-Engineering-2-Cloud-Computing/ElonLexInterview_5YearsAgo1.json"

# Load the JSON data
with open(json_file_path) as f:
    data = json.load(f)
# %%
try:
    # Load the JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Extract the full transcript text
    full_transcript = " ".join([item['transcript'] for item in data['results']['transcripts']])

    # Split the text into manageable chunks
    chunk_size = int(len(full_transcript) * 0.05)
    chunks = [full_transcript[i:i+chunk_size] for i in range(0, len(full_transcript), chunk_size)]

    # Analyze sentiment for each chunk and collect sentiment scores
    sentiment_scores_5_1 = []
    for chunk in chunks:
        try:
            response = comprehend.detect_sentiment(Text=chunk, LanguageCode='en')
            sentiment_scores_5_1.append(response['SentimentScore'])
        except Exception as e:
            print(f"Error analyzing sentiment for chunk: {str(e)}")

    # Compute the average sentiment scores
    if sentiment_scores_5_1:
        avg_sentiment_score_5_1 = {
            'Positive': np.mean([score['Positive'] for score in sentiment_scores_5_1]),
            'Negative': np.mean([score['Negative'] for score in sentiment_scores_5_1]),
            'Neutral': np.mean([score['Neutral'] for score in sentiment_scores_5_1]),
            'Mixed': np.mean([score['Mixed'] for score in sentiment_scores_5_1]),
        }
        print("\nAverage Sentiment Score across all chunks:")
        print(avg_sentiment_score_5_1)
    else:
        print("No sentiment scores to compute the average.")

except FileNotFoundError:
    print(f"Error: The file at path {json_file_path} could not be found.")
except json.JSONDecodeError:
    print(f"Error: Failed to decode JSON from the file at path {json_file_path}.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
# %%
# Load the JSON file
json_file_path = "/workspaces/Data-Engineering-2-Cloud-Computing/ElonLexInterview_5YearsAgo2.json"

# Load the JSON data
with open(json_file_path) as f:
    data = json.load(f)
# %%
try:
    # Load the JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Extract the full transcript text
    full_transcript = " ".join([item['transcript'] for item in data['results']['transcripts']])

    # Determine chunk size as 5% of total text size
    chunk_size = int(len(full_transcript) * 0.05)
    chunks = [full_transcript[i:i+chunk_size] for i in range(0, len(full_transcript), chunk_size)]

    # Analyze sentiment for each chunk and collect sentiment scores
    sentiment_scores_5_2 = []
    for chunk in chunks:
        try:
            response = comprehend.detect_sentiment(Text=chunk, LanguageCode='en')
            sentiment_scores_5_2.append(response['SentimentScore'])
        except Exception as e:
            print(f"Error analyzing sentiment for chunk: {str(e)}")

    # Compute the average sentiment scores
    if sentiment_scores_5_2:
        avg_sentiment_score_5_2 = {
            'Positive': np.mean([score['Positive'] for score in sentiment_scores_5_2]),
            'Negative': np.mean([score['Negative'] for score in sentiment_scores_5_2]),
            'Neutral': np.mean([score['Neutral'] for score in sentiment_scores_5_2]),
            'Mixed': np.mean([score['Mixed'] for score in sentiment_scores_5_2]),
        }
        print("\nAverage Sentiment Score across all chunks:")
        print(avg_sentiment_score_5_2)
    else:
        print("No sentiment scores to compute the average.")

except FileNotFoundError:
    print(f"Error: The file at path {json_file_path} could not be found.")
except json.JSONDecodeError:
    print(f"Error: Failed to decode JSON from the file at path {json_file_path}.")
except Exception as e:
    print(f"An error occurred: {str(e)}")
# %%
# Create DataFrame
sentiment = pd.DataFrame([
    avg_sentiment_score_5_2,
    avg_sentiment_score_5_1,
    avg_sentiment_score_1,
    avg_sentiment_score
], index=['Elon_5_years_ago', 'Elon_5_years_ago_2', 'Elon_2_years_ago', 'Elon_1_year_ago'])

# Display the DataFrame
print(sentiment)
sentiment = pd.DataFrame([
    avg_sentiment_score_5_2,
    avg_sentiment_score_5_1,
    avg_sentiment_score_1,
    avg_sentiment_score
], index=['Elon_5_years_ago', 'Elon_5_years_ago_2', 'Elon_2_years_ago', 'Elon_1_year_ago'])

# Display the DataFrame
print(sentiment)
# %%
# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Create a grouped bar plot
sentiment.plot(kind='bar', stacked=True, ax=ax, color=['#4CAF50', '#FF5722', '#2196F3', '#FFC107'])

# Add labels and title
plt.title('Average Sentiment Scores Across Lex Fridman Interviews with Elon Musk')
plt.xlabel('Averages')
plt.ylabel('Sentiment Scores')
plt.xticks(rotation=0)  # Rotate x-axis labels
plt.legend(title='Sentiment', loc='upper right')

# Show the plot
plt.tight_layout()
plt.show()

# %%
# Calculate average proportions for each sentiment type
# Sort the average proportions
avg_proportions = sentiment.mean().sort_values(ascending=True)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Bar plot for average proportions
avg_proportions.plot(kind='bar', color=['#4CAF50', '#FF5722', '#2196F3', '#FFC107'], ax=ax)

# Add labels and title
plt.title('Average Proportion of Each Sentiment Across Lex Fridman Interviews with Elon Musk')
plt.xlabel('Sentiments')
plt.ylabel('Average Proportion')
plt.xticks(rotation=0)  # Rotate x-axis labels
plt.legend(title='Averages', loc='upper right')

# Show the plot
plt.tight_layout()
plt.show()
# %%
# Extract positive sentiment scores
positive_scores = [score['Positive'] for score in sentiment_scores_1]

# Time points (e.g., chunk indices or other time-related variable)
time_points = range(1, len(positive_scores) + 1)  # 1 to 7 for 7 chunks

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time_points, positive_scores, marker='o', linestyle='-', color='#4CAF50')
plt.title('Positive Sentiment Score Over Time')
plt.xlabel('Time (Chunk Index)')
plt.ylabel('Positive Sentiment Score')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()

# %%
# Extract positive sentiment scores for each set
positive_scores_1 = [score['Positive'] for score in sentiment_scores_1]
positive_scores_2 = [score['Positive'] for score in sentiment_scores_2]
positive_scores_3 = [score['Positive'] for score in sentiment_scores_5_1]
positive_scores_4 = [score['Positive'] for score in sentiment_scores_5_2]

# Time points (chunk indices)
time_points = np.arange(1, len(positive_scores_1) + 1)

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(10, 18))

# Plot for sentiment scores 1
axs[0].plot(time_points, positive_scores_1, marker='o', linestyle='-', color='#4CAF50')
axs[0].set_title('Sentiment Analysis Over Time: Positive Sentiment for Lex Fridman Interview with Elon Musk (1 Year Ago)')
axs[0].set_xlabel('Time (Chunk Index)')
axs[0].set_ylabel('Positive Sentiment Score')
axs[0].grid(True)

# Plot for sentiment scores 2
axs[1].plot(time_points, positive_scores_2, marker='o', linestyle='-', color='#FF5722')
axs[1].set_title('Sentiment Analysis Over Time: Positive Sentiment for Lex Fridman Interview with Elon Musk (2 Years Ago)')
axs[1].set_xlabel('Time (Chunk Index)')
axs[1].set_ylabel('Positive Sentiment Score')
axs[1].grid(True)

# Plot for sentiment scores 3
axs[2].plot(time_points, positive_scores_3, marker='o', linestyle='-', color='#2196F3')
axs[2].set_title('Sentiment Analysis Over Time: Positive Sentiment for Lex Fridman Interview with Elon Musk (5(1) Years Ago)')
axs[2].set_xlabel('Time (Chunk Index)')
axs[2].set_ylabel('Positive Sentiment Score')
axs[2].grid(True)

# Plot for sentiment scores 4
axs[3].plot(time_points, positive_scores_4, marker='o', linestyle='-', color='#FFC107')
axs[3].set_title('Sentiment Analysis Over Time: Positive Sentiment for Lex Fridman Interview with Elon Musk (5(2) Years Ago)')
axs[3].set_xlabel('Time (Chunk Index)')
axs[3].set_ylabel('Positive Sentiment Score')
axs[3].grid(True)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
# %%
